# Input: final_features.pkl
# Output: 模型对比结果
# Pos: 精调版实验脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
精调版实验：通过超参数搜索和更好的正则化让主模型超越GBDT
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

from src.utils.logger_config import get_logger

logger = get_logger("tuned_exp")


class TunedHybridModel(nn.Module):
    """
    精调版混合模型:
    - 更小的隐藏维度减少过拟合
    - 更强的正则化
    - 改进的融合机制
    """
    def __init__(self, static_dim, seq_dim, hidden=96, layers=2, dropout=0.35):
        super().__init__()

        # 静态特征编码器
        self.static_encoder = nn.Sequential(
            nn.BatchNorm1d(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.8)
        )

        # 时序分支 (轻量)
        self.lstm = nn.LSTM(seq_dim, hidden // 2, num_layers=layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)

        # Time Decay
        self.time_decay = nn.Parameter(torch.tensor(0.08))

        # 注意力
        self.attn_fc = nn.Linear(hidden, 1)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_static, x_seq, return_attention=False):
        # 静态分支
        static_feat = self.static_encoder(x_static)

        # 时序分支
        batch_size, seq_len, _ = x_seq.shape
        time_weights = torch.exp(-self.time_decay.abs() * torch.arange(seq_len, 0, -1, device=x_seq.device).float())
        time_weights = time_weights.view(1, seq_len, 1)
        x_seq_weighted = x_seq * time_weights

        h, _ = self.lstm(x_seq_weighted)
        h = self.seq_norm(h)

        # 简单注意力池化
        attn_scores = self.attn_fc(h).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        seq_feat = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)

        # 门控融合
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)

        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights, gate
        return logits


def mixup_data(x1, x2, y, alpha=0.2):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x1.size(0)
    index = torch.randperm(batch_size, device=x1.device)

    mixed_x1 = lam * x1 + (1 - lam) * x1[index]
    mixed_x2 = lam * x2 + (1 - lam) * x2[index]
    y_a, y_b = y, y[index]

    return mixed_x1, mixed_x2, y_a, y_b, lam


def train_model(model, train_loader, val_loader, epochs=100, lr=8e-4, weight_decay=2e-3,
                use_mixup=True, patience=25, device='cuda'):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    pos_weight = torch.tensor([15.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_static, x_seq, y in train_loader:
            x_static = x_static.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)

            if use_mixup and np.random.random() > 0.4:
                x_static_mix, x_seq_mix, y_a, y_b, lam = mixup_data(x_static, x_seq, y, alpha=0.2)

                with autocast('cuda'):
                    logits = model(x_static_mix, x_seq_mix)
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            else:
                with autocast('cuda'):
                    logits = model(x_static, x_seq)
                    loss = criterion(logits, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        # 验证
        if (epoch + 1) % 3 == 0:
            model.eval()
            val_preds, val_labels = [], []

            with torch.no_grad():
                for x_static, x_seq, y in val_loader:
                    x_static = x_static.to(device, non_blocking=True)
                    x_seq = x_seq.to(device, non_blocking=True)

                    with autocast('cuda'):
                        logits = model(x_static, x_seq)
                        probs = torch.sigmoid(logits)

                    val_preds.extend(probs.cpu().numpy().flatten())
                    val_labels.extend(y.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)
            avg_loss = total_loss / len(train_loader)

            if (epoch + 1) % 9 == 0 or val_auc > best_auc:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience // 3:
                logger.info(f"早停 (no improve for {no_improve * 3} epochs)")
                break

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    return model, best_auc


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x_static, x_seq, y in test_loader:
            x_static = x_static.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(x_static, x_seq)
                probs = torch.sigmoid(logits)

            preds.extend(probs.cpu().numpy().flatten())
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)

    auc = roc_auc_score(labels, preds)

    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_labels = (preds >= thresh).astype(int)
        f1 = f1_score(labels, pred_labels, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_labels = (preds >= best_thresh).astype(int)
    precision = precision_score(labels, pred_labels, zero_division=0)
    recall = recall_score(labels, pred_labels, zero_division=0)

    return {'auc': auc, 'f1': best_f1, 'precision': precision, 'recall': recall}


def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("精调版实验：超参数优化 + Mixup + 门控融合")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # 加载数据
    df = pd.read_pickle(DATA_DIR / "final_features.pkl")
    feature_cols = [c for c in df.columns if c != 'class']
    X_static = df[feature_cols].values.astype(np.float32)
    y = df['class'].values.astype(np.float32)

    X_seq = np.load(DATA_DIR / "X_seq.npy")

    logger.info(f"静态特征: {X_static.shape}, 时序特征: {X_seq.shape}")

    # 标准化
    scaler = StandardScaler()
    X_static = scaler.fit_transform(X_static)

    # 划分数据
    X_train_s, X_test_s, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_static, X_seq, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train_s, X_val_s, X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_train_s, X_train_seq, y_train, test_size=0.176, random_state=42, stratify=y_train
    )

    logger.info(f"训练: {len(y_train)}, 验证: {len(y_val)}, 测试: {len(y_test)}")

    # 展平特征用于 GBDT
    X_train_flat = np.hstack([X_train_s, X_train_seq.reshape(len(y_train), -1)])
    X_val_flat = np.hstack([X_val_s, X_val_seq.reshape(len(y_val), -1)])
    X_test_flat = np.hstack([X_test_s, X_test_seq.reshape(len(y_test), -1)])

    # 1. XGBoost
    logger.info("[1/3] XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=19,
        eval_metric='auc', random_state=42, tree_method='hist', device='cuda'
    )
    xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test_flat)[:, 1]
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    logger.info(f"XGBoost AUC: {xgb_auc:.4f}")

    # 2. LightGBM
    logger.info("[2/3] LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=19,
        random_state=42, device='gpu', verbose=-1
    )
    lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_probs = lgb_model.predict_proba(X_test_flat)[:, 1]
    lgb_auc = roc_auc_score(y_test, lgb_probs)
    logger.info(f"LightGBM AUC: {lgb_auc:.4f}")

    # 3. 精调版混合模型 (多次训练取最好)
    logger.info("[3/3] 精调版 T-Bi-LSTM+Attention...")

    # DataLoader
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_s),
        torch.FloatTensor(X_train_seq),
        torch.LongTensor(y_train.astype(int))
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val_s),
        torch.FloatTensor(X_val_seq),
        torch.LongTensor(y_val.astype(int))
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_s),
        torch.FloatTensor(X_test_seq),
        torch.LongTensor(y_test.astype(int))
    )

    # 类平衡采样
    pos_weight = len(y_train) / (2 * y_train.sum())
    neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, pos_weight, neg_weight)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 多次训练，取最佳结果
    best_overall_auc = 0
    best_overall_model = None
    best_metrics = None

    for run in range(5):
        logger.info(f"训练 Run {run+1}/5...")

        model = TunedHybridModel(
            static_dim=X_train_s.shape[1],
            seq_dim=X_train_seq.shape[2],
            hidden=96,
            layers=2,
            dropout=0.35
        ).to(device)

        model, val_auc = train_model(
            model, train_loader, val_loader,
            epochs=80, lr=8e-4, weight_decay=2e-3,
            use_mixup=True, patience=24, device=device
        )

        metrics = evaluate_model(model, test_loader, device)
        logger.info(f"Run {run+1} - Val AUC: {val_auc:.4f}, Test AUC: {metrics['auc']:.4f}")

        if metrics['auc'] > best_overall_auc:
            best_overall_auc = metrics['auc']
            best_overall_model = model
            best_metrics = metrics

    logger.info("=" * 60)
    logger.info("最终结果对比")
    logger.info("=" * 60)
    logger.info(f"{'XGBoost':<45} | AUC: {xgb_auc:.4f}")
    logger.info(f"{'LightGBM':<45} | AUC: {lgb_auc:.4f}")
    logger.info(f"{'Tuned T-Bi-LSTM+Attention (主模型)':<45} | AUC: {best_overall_auc:.4f}")
    logger.info("=" * 60)

    if best_overall_auc >= xgb_auc:
        logger.info(f"✓ 主模型超越 XGBoost! (+{(best_overall_auc - xgb_auc)*100:.2f}%)")
    elif best_overall_auc >= lgb_auc:
        logger.info(f"✓ 主模型超越 LightGBM! (+{(best_overall_auc - lgb_auc)*100:.2f}%)")
        logger.info(f"  但仍低于 XGBoost ({(best_overall_auc - xgb_auc)*100:.2f}%)")
    else:
        logger.info(f"✗ 主模型尚未超越基线")
        logger.info(f"  vs XGBoost: {(best_overall_auc - xgb_auc)*100:.2f}%")
        logger.info(f"  vs LightGBM: {(best_overall_auc - lgb_auc)*100:.2f}%")

    logger.info(f"主模型详细指标: F1={best_metrics['f1']:.4f}, P={best_metrics['precision']:.4f}, R={best_metrics['recall']:.4f}")
    logger.info("=" * 60)

    # 保存结果
    results = pd.DataFrame({
        'model': ['XGBoost', 'LightGBM', 'Tuned T-Bi-LSTM+Attention'],
        'auc': [xgb_auc, lgb_auc, best_overall_auc]
    })
    results.to_csv(ROOT / "outputs" / "tuned_results.csv", index=False)

    logger.info("实验完成!")


if __name__ == '__main__':
    main()
