# Input: final_features.pkl, X_seq.npy, y_seq.npy
# Output: 模型对比结果，验证主模型是否超越基线
# Pos: 知识蒸馏版实验脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
知识蒸馏 + 强正则实验
- XGBoost 作为教师模型提供软标签
- 简化的 T-Bi-LSTM+Attention 作为学生模型
- Mixup 数据增强
- 门控时序融合
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

from src.utils.logger_config import get_logger

logger = get_logger("distill_exp")


# ==================== 简化版主模型 ====================

class SimplifiedTBiLSTMAttention(nn.Module):
    """
    简化版 T-Bi-LSTM+Attention:
    - 更小的隐藏维度
    - 更强的正则化
    - 门控时序融合（可学习衰减时序权重）
    """
    def __init__(self, static_dim, seq_dim, hidden=128, layers=2, dropout=0.4):
        super().__init__()

        # 静态特征编码（轻量）
        self.static_encoder = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 时序分支（轻量 Bi-LSTM）
        self.lstm = nn.LSTM(seq_dim, hidden // 2, num_layers=layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)

        # Time Decay Gate (可学习)
        self.time_decay = nn.Parameter(torch.tensor(0.05))

        # Attention (简化版)
        self.attn_query = nn.Linear(hidden, hidden // 4)
        self.attn_key = nn.Linear(hidden, hidden // 4)
        self.attn_value = nn.Linear(hidden, hidden)

        # 门控融合：让模型学习时序分支的重要性
        self.seq_gate = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # 输出层（简化）
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_static, x_seq, return_attention=False):
        # 静态分支
        static_feat = self.static_encoder(x_static)  # [B, hidden]

        # 时序分支 + Time Decay
        batch_size, seq_len, _ = x_seq.shape
        time_weights = torch.exp(-self.time_decay.abs() * torch.arange(seq_len, 0, -1, device=x_seq.device).float())
        time_weights = time_weights.view(1, seq_len, 1)
        x_seq_weighted = x_seq * time_weights

        h, _ = self.lstm(x_seq_weighted)  # [B, T, hidden]
        h = self.seq_norm(h)

        # 简化版 Attention
        Q = self.attn_query(h)  # [B, T, hidden//4]
        K = self.attn_key(h)
        V = self.attn_value(h)

        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # [B, T, hidden]

        # 加权池化
        seq_feat = attn_out.mean(dim=1)  # [B, hidden]

        # 门控融合
        gate_input = torch.cat([static_feat, seq_feat], dim=1)
        gate = self.seq_gate(gate_input)  # [B, 1] 时序分支的重要性

        # 融合：静态 + 门控时序
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)  # [B, hidden*2]

        logits = self.classifier(fused)

        if return_attention:
            return logits, attn_weights, gate
        return logits


# ==================== 数据增强 ====================

def mixup_data(x_static, x_seq, y, alpha=0.3):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x_static.size(0)
    index = torch.randperm(batch_size, device=x_static.device)

    mixed_x_static = lam * x_static + (1 - lam) * x_static[index]
    mixed_x_seq = lam * x_seq + (1 - lam) * x_seq[index]
    y_a, y_b = y, y[index]

    return mixed_x_static, mixed_x_seq, y_a, y_b, lam


# ==================== 知识蒸馏损失 ====================

class DistillationLoss(nn.Module):
    """知识蒸馏损失 = α * BCE + (1-α) * KL散度"""
    def __init__(self, alpha=0.3, temperature=3.0, pos_weight=5.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.pos_weight = pos_weight

    def forward(self, student_logits, teacher_probs, labels):
        # 硬标签损失 (BCE)
        bce_loss = F.binary_cross_entropy_with_logits(
            student_logits, labels,
            pos_weight=torch.tensor(self.pos_weight, device=student_logits.device)
        )

        # 软标签损失 (KL散度)
        student_probs = torch.sigmoid(student_logits / self.temperature)
        soft_loss = F.mse_loss(student_probs, teacher_probs ** (1 / self.temperature))

        return self.alpha * bce_loss + (1 - self.alpha) * soft_loss


# ==================== 训练函数 ====================

def train_distilled_model(model, train_loader, val_loader, teacher_probs_dict,
                          epochs=100, lr=1e-3, weight_decay=3e-3,
                          use_mixup=True, patience=30):
    """训练知识蒸馏模型"""
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    criterion = DistillationLoss(alpha=0.4, temperature=3.0, pos_weight=5.0)
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None
    no_improve = 0

    logger.info(f"开始知识蒸馏训练 (epochs={epochs}, lr={lr}, wd={weight_decay})")

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_idx, (x_static, x_seq, y, indices) in enumerate(train_loader):
            x_static = x_static.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float().unsqueeze(1)

            # 获取教师软标签
            teacher_probs = teacher_probs_dict[indices.numpy()].to(device)

            # Mixup (可选)
            if use_mixup and np.random.random() > 0.3:
                x_static_mix, x_seq_mix, y_a, y_b, lam = mixup_data(x_static, x_seq, y, alpha=0.3)
                teacher_mix = lam * teacher_probs + (1 - lam) * teacher_probs[torch.randperm(len(teacher_probs))]

                with autocast('cuda'):
                    logits = model(x_static_mix, x_seq_mix)
                    loss_a = criterion(logits, teacher_mix, y_a)
                    loss_b = criterion(logits, teacher_mix, y_b)
                    loss = lam * loss_a + (1 - lam) * loss_b
            else:
                with autocast('cuda'):
                    logits = model(x_static, x_seq)
                    loss = criterion(logits, teacher_probs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()

        # 验证
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for x_static, x_seq, y, _ in val_loader:
                    x_static = x_static.to(device, non_blocking=True)
                    x_seq = x_seq.to(device, non_blocking=True)

                    with autocast('cuda'):
                        logits = model(x_static, x_seq)
                        probs = torch.sigmoid(logits)

                    val_preds.extend(probs.cpu().numpy().flatten())
                    val_labels.extend(y.numpy())

            val_auc = roc_auc_score(val_labels, val_preds)
            avg_loss = total_loss / len(train_loader)

            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience // 5:
                logger.info(f"早停 (patience={patience//5})")
                break

    if best_state:
        model.load_state_dict(best_state)

    return model, best_auc


def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for x_static, x_seq, y, _ in test_loader:
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

    # 找最优阈值
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.1, 0.9, 0.02):
        pred_labels = (preds >= thresh).astype(int)
        f1 = f1_score(labels, pred_labels)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_labels = (preds >= best_thresh).astype(int)
    precision = precision_score(labels, pred_labels)
    recall = recall_score(labels, pred_labels)

    return {
        'auc': auc,
        'f1': best_f1,
        'precision': precision,
        'recall': recall,
        'threshold': best_thresh
    }


# ==================== 主函数 ====================

def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("知识蒸馏实验：简化架构 + 强正则 + Mixup")
    logger.info(f"Device: {device}")
    logger.info("=" * 60)

    # 1. 加载数据
    features_df = pd.read_pickle(DATA_DIR / "final_features.pkl")
    X_seq = np.load(DATA_DIR / "X_seq.npy")
    y = np.load(DATA_DIR / "y_seq.npy")

    # 移除泄漏列（label, class 等）
    leak_cols = ['label', 'class', 'RES', '结果']
    feature_cols = [c for c in features_df.columns if c not in leak_cols]
    X_static = features_df[feature_cols].values.astype(np.float32)
    logger.info(f"移除泄漏列后特征数: {len(feature_cols)}")

    # 标准化
    scaler = StandardScaler()
    X_static = scaler.fit_transform(X_static)

    # 划分数据
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=42, stratify=y)
    train_idx, val_idx = train_test_split(train_idx, test_size=0.176, random_state=42, stratify=y[train_idx])

    logger.info(f"训练: {len(train_idx)}, 验证: {len(val_idx)}, 测试: {len(test_idx)}")

    X_train_static, X_val_static, X_test_static = X_static[train_idx], X_static[val_idx], X_static[test_idx]
    X_train_seq, X_val_seq, X_test_seq = X_seq[train_idx], X_seq[val_idx], X_seq[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    # 展平特征用于 GBDT
    X_train_flat = np.hstack([X_train_static, X_train_seq.reshape(len(train_idx), -1)])
    X_val_flat = np.hstack([X_val_static, X_val_seq.reshape(len(val_idx), -1)])
    X_test_flat = np.hstack([X_test_static, X_test_seq.reshape(len(test_idx), -1)])

    # 2. 训练 XGBoost 教师模型
    logger.info("[Step 1] 训练 XGBoost 教师模型...")

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=19,
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        tree_method='hist',
        device='cuda'
    )
    xgb_model.fit(X_train_flat, y_train,
                  eval_set=[(X_val_flat, y_val)],
                  verbose=False)

    xgb_val_probs = xgb_model.predict_proba(X_val_flat)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test_flat)[:, 1]
    xgb_train_probs = xgb_model.predict_proba(X_train_flat)[:, 1]

    xgb_val_auc = roc_auc_score(y_val, xgb_val_probs)
    xgb_test_auc = roc_auc_score(y_test, xgb_test_probs)
    logger.info(f"XGBoost 教师模型 - Val AUC: {xgb_val_auc:.4f}, Test AUC: {xgb_test_auc:.4f}")

    # 3. LightGBM 基线
    logger.info("[Step 2] LightGBM 基线...")

    lgb_model = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=19,
        random_state=42,
        device='gpu',
        verbose=-1
    )
    lgb_model.fit(X_train_flat, y_train,
                  eval_set=[(X_val_flat, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

    lgb_test_probs = lgb_model.predict_proba(X_test_flat)[:, 1]
    lgb_test_auc = roc_auc_score(y_test, lgb_test_probs)
    logger.info(f"LightGBM 基线 - Test AUC: {lgb_test_auc:.4f}")

    # 4. 准备知识蒸馏数据
    logger.info("[Step 3] 准备知识蒸馏训练数据...")

    # 创建包含索引的 Dataset
    class IndexedDataset(torch.utils.data.Dataset):
        def __init__(self, x_static, x_seq, y, indices):
            self.x_static = torch.FloatTensor(x_static)
            self.x_seq = torch.FloatTensor(x_seq)
            self.y = torch.LongTensor(y)
            self.indices = indices

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x_static[idx], self.x_seq[idx], self.y[idx], self.indices[idx]

    # 教师软标签（训练集）
    teacher_probs = torch.FloatTensor(xgb_train_probs).unsqueeze(1)

    train_dataset = IndexedDataset(X_train_static, X_train_seq, y_train, np.arange(len(y_train)))
    val_dataset = IndexedDataset(X_val_static, X_val_seq, y_val, np.arange(len(y_val)))
    test_dataset = IndexedDataset(X_test_static, X_test_seq, y_test, np.arange(len(y_test)))

    # 类平衡采样
    pos_weight = len(y_train) / (2 * y_train.sum())
    neg_weight = len(y_train) / (2 * (len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, pos_weight, neg_weight)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # 5. 训练知识蒸馏学生模型
    logger.info("[Step 4] 训练知识蒸馏 T-Bi-LSTM+Attention (主模型)...")

    model = SimplifiedTBiLSTMAttention(
        static_dim=X_train_static.shape[1],
        seq_dim=X_train_seq.shape[2],
        hidden=128,
        layers=2,
        dropout=0.4
    ).to(device)

    # 创建教师概率字典（用于 Mixup 时的索引访问）
    teacher_probs_dict = teacher_probs

    model, val_auc = train_distilled_model(
        model, train_loader, val_loader, teacher_probs_dict,
        epochs=120, lr=1e-3, weight_decay=3e-3,
        use_mixup=True, patience=40
    )

    # 6. 测试集评估
    logger.info("[Step 5] 测试集评估...")

    metrics = evaluate_model(model, test_loader, device)

    logger.info("=" * 60)
    logger.info("最终结果对比")
    logger.info("=" * 60)
    logger.info(f"{'XGBoost (Teacher)':<45} | AUC: {xgb_test_auc:.4f}")
    logger.info(f"{'LightGBM (Baseline)':<45} | AUC: {lgb_test_auc:.4f}")
    logger.info(f"{'Distilled T-Bi-LSTM+Attention (主模型)':<45} | AUC: {metrics['auc']:.4f}")
    logger.info("=" * 60)

    if metrics['auc'] >= xgb_test_auc:
        logger.info(f"✓ 主模型超越 XGBoost! (+{(metrics['auc'] - xgb_test_auc)*100:.2f}%)")
    else:
        logger.info(f"✗ 主模型尚未超越 XGBoost")
        logger.info(f"  主模型 AUC: {metrics['auc']:.4f} vs XGBoost AUC: {xgb_test_auc:.4f} ({(metrics['auc'] - xgb_test_auc)*100:.2f}%)")

    logger.info("=" * 60)
    logger.info(f"主模型详细指标: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
    logger.info("=" * 60)

    # 保存结果
    results = {
        'model': ['XGBoost', 'LightGBM', 'Distilled T-Bi-LSTM+Attention'],
        'auc': [xgb_test_auc, lgb_test_auc, metrics['auc']],
        'note': ['Teacher', 'Baseline', 'Student (Main)']
    }
    pd.DataFrame(results).to_csv(ROOT / "outputs" / "distill_results.csv", index=False)

    logger.info("实验完成!")


if __name__ == '__main__':
    main()
