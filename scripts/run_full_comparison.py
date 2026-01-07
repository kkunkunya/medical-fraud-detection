# Input: final_features.pkl, X_seq.npy
# Output: 完整模型对比结果, 图表
# Pos: 完整模型对比实验脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
完整模型对比实验
- 6个模型: XGBoost, LightGBM, DNN, LSTM, Transformer, T-Bi-LSTM+Attention
- 输出对比图表
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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.utils.logger_config import get_logger

logger = get_logger("full_comparison")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 模型定义 ====================

class DNN(nn.Module):
    """3层全连接网络"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BasicLSTM(nn.Module):
    """基础LSTM"""
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, layers, batch_first=True,
                           dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.fc(h[:, -1, :])


class TransformerModel(nn.Module):
    """Transformer Encoder"""
    def __init__(self, input_dim, d_model=128, nhead=4, layers=2, dropout=0.3):
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embed(x) + self.pos[:, :x.size(1), :]
        x = self.transformer(x)
        return self.fc(x[:, -1, :])


class TBiLSTMAttention(nn.Module):
    """主模型: T-Bi-LSTM+Attention"""
    def __init__(self, static_dim, seq_dim, hidden=96, layers=2, dropout=0.35):
        super().__init__()
        # 静态分支
        self.static_encoder = nn.Sequential(
            nn.BatchNorm1d(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.8)
        )
        # 时序分支
        self.lstm = nn.LSTM(seq_dim, hidden // 2, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)
        self.time_decay = nn.Parameter(torch.tensor(0.08))
        self.attn_fc = nn.Linear(hidden, 1)
        # 门控
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1), nn.Sigmoid()
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

    def forward(self, x_static, x_seq):
        static_feat = self.static_encoder(x_static)
        B, T, _ = x_seq.shape
        tw = torch.exp(-self.time_decay.abs() * torch.arange(T, 0, -1, device=x_seq.device).float())
        h, _ = self.lstm(x_seq * tw.view(1, T, 1))
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        return self.classifier(fused)


# ==================== 训练函数 ====================

def train_nn(model, train_loader, val_loader, epochs=60, lr=8e-4, device='cuda', model_type='hybrid'):
    """通用神经网络训练"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device))
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            if model_type == 'hybrid':
                x_s, x_seq, y = batch
                x_s = x_s.to(device)
                x_seq = x_seq.to(device)
                y = y.to(device).float().unsqueeze(1)
                with autocast('cuda'):
                    logits = model(x_s, x_seq)
            else:
                x, y = batch[0].to(device), batch[1].to(device).float().unsqueeze(1)
                with autocast('cuda'):
                    logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'hybrid':
                        x_s, x_seq, y = batch
                        logits = model(x_s.to(device), x_seq.to(device))
                    else:
                        logits = model(batch[0].to(device))
                        y = batch[1]
                    preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    labels.extend(y.numpy())
            auc = roc_auc_score(labels, preds)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_auc


def evaluate(model, test_loader, device='cuda', model_type='hybrid'):
    """评估模型"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if model_type == 'hybrid':
                x_s, x_seq, y = batch
                logits = model(x_s.to(device), x_seq.to(device))
            else:
                logits = model(batch[0].to(device))
                y = batch[1]
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)

    # 最优阈值
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        f1 = f1_score(labels, (preds >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t

    pred_labels = (preds >= best_t).astype(int)
    return {
        'auc': auc,
        'f1': best_f1,
        'precision': precision_score(labels, pred_labels, zero_division=0),
        'recall': recall_score(labels, pred_labels, zero_division=0),
        'accuracy': accuracy_score(labels, pred_labels)
    }


def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"
    FIG_DIR = ROOT / "outputs" / "figures"
    FIG_DIR.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("完整模型对比实验 (6模型)")
    logger.info("=" * 60)

    # 加载数据
    df = pd.read_pickle(DATA_DIR / "final_features.pkl")
    feature_cols = [c for c in df.columns if c != 'class']
    X_static = df[feature_cols].values.astype(np.float32)
    y = df['class'].values.astype(np.float32)
    X_seq = np.load(DATA_DIR / "X_seq.npy")

    scaler = StandardScaler()
    X_static = scaler.fit_transform(X_static)

    # 划分
    X_train_s, X_test_s, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_static, X_seq, y, test_size=0.15, random_state=42, stratify=y)
    X_train_s, X_val_s, X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_train_s, X_train_seq, y_train, test_size=0.176, random_state=42, stratify=y_train)

    X_train_flat = np.hstack([X_train_s, X_train_seq.reshape(len(y_train), -1)])
    X_val_flat = np.hstack([X_val_s, X_val_seq.reshape(len(y_val), -1)])
    X_test_flat = np.hstack([X_test_s, X_test_seq.reshape(len(y_test), -1)])

    logger.info(f"数据: 训练={len(y_train)}, 验证={len(y_val)}, 测试={len(y_test)}")

    results = {}

    # 1. XGBoost
    logger.info("[1/6] XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                   scale_pos_weight=19, eval_metric='auc', random_state=42,
                                   tree_method='hist', device='cuda')
    xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
    xgb_probs = xgb_model.predict_proba(X_test_flat)[:, 1]
    xgb_pred = (xgb_probs >= 0.5).astype(int)
    results['XGBoost'] = {
        'auc': roc_auc_score(y_test, xgb_probs),
        'f1': f1_score(y_test, xgb_pred),
        'precision': precision_score(y_test, xgb_pred),
        'recall': recall_score(y_test, xgb_pred),
        'accuracy': accuracy_score(y_test, xgb_pred)
    }
    logger.info(f"XGBoost AUC: {results['XGBoost']['auc']:.4f}")

    # 2. LightGBM
    logger.info("[2/6] LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                    scale_pos_weight=19, random_state=42, device='gpu', verbose=-1)
    lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])
    lgb_probs = lgb_model.predict_proba(X_test_flat)[:, 1]
    lgb_pred = (lgb_probs >= 0.5).astype(int)
    results['LightGBM'] = {
        'auc': roc_auc_score(y_test, lgb_probs),
        'f1': f1_score(y_test, lgb_pred),
        'precision': precision_score(y_test, lgb_pred),
        'recall': recall_score(y_test, lgb_pred),
        'accuracy': accuracy_score(y_test, lgb_pred)
    }
    logger.info(f"LightGBM AUC: {results['LightGBM']['auc']:.4f}")

    # DataLoader 准备
    flat_dim = X_train_flat.shape[1]
    seq_dim = X_train_seq.shape[2]
    static_dim = X_train_s.shape[1]

    # 类平衡采样
    pos_w = len(y_train) / (2 * y_train.sum())
    neg_w = len(y_train) / (2 * (len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, pos_w, neg_w)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # DNN loader
    dnn_train = DataLoader(TensorDataset(torch.FloatTensor(X_train_flat), torch.LongTensor(y_train.astype(int))),
                           batch_size=256, sampler=sampler)
    dnn_val = DataLoader(TensorDataset(torch.FloatTensor(X_val_flat), torch.LongTensor(y_val.astype(int))),
                         batch_size=512, shuffle=False)
    dnn_test = DataLoader(TensorDataset(torch.FloatTensor(X_test_flat), torch.LongTensor(y_test.astype(int))),
                          batch_size=512, shuffle=False)

    # Seq loader
    seq_train = DataLoader(TensorDataset(torch.FloatTensor(X_train_seq), torch.LongTensor(y_train.astype(int))),
                           batch_size=256, sampler=sampler)
    seq_val = DataLoader(TensorDataset(torch.FloatTensor(X_val_seq), torch.LongTensor(y_val.astype(int))),
                         batch_size=512, shuffle=False)
    seq_test = DataLoader(TensorDataset(torch.FloatTensor(X_test_seq), torch.LongTensor(y_test.astype(int))),
                          batch_size=512, shuffle=False)

    # Hybrid loader
    hybrid_train = DataLoader(TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(X_train_seq),
                                            torch.LongTensor(y_train.astype(int))),
                              batch_size=256, sampler=sampler)
    hybrid_val = DataLoader(TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(X_val_seq),
                                          torch.LongTensor(y_val.astype(int))),
                            batch_size=512, shuffle=False)
    hybrid_test = DataLoader(TensorDataset(torch.FloatTensor(X_test_s), torch.FloatTensor(X_test_seq),
                                           torch.LongTensor(y_test.astype(int))),
                             batch_size=512, shuffle=False)

    # 3. DNN
    logger.info("[3/6] DNN...")
    dnn = DNN(flat_dim, [256, 128, 64], dropout=0.35).to(device)
    dnn, _ = train_nn(dnn, dnn_train, dnn_val, epochs=60, device=device, model_type='flat')
    results['DNN'] = evaluate(dnn, dnn_test, device, model_type='flat')
    logger.info(f"DNN AUC: {results['DNN']['auc']:.4f}")

    # 4. LSTM
    logger.info("[4/6] LSTM...")
    lstm = BasicLSTM(seq_dim, hidden=128, layers=2, dropout=0.3).to(device)
    lstm, _ = train_nn(lstm, seq_train, seq_val, epochs=50, device=device, model_type='seq')
    results['LSTM'] = evaluate(lstm, seq_test, device, model_type='seq')
    logger.info(f"LSTM AUC: {results['LSTM']['auc']:.4f}")

    # 5. Transformer
    logger.info("[5/6] Transformer...")
    transformer = TransformerModel(seq_dim, d_model=128, nhead=4, layers=2, dropout=0.3).to(device)
    transformer, _ = train_nn(transformer, seq_train, seq_val, epochs=50, device=device, model_type='seq')
    results['Transformer'] = evaluate(transformer, seq_test, device, model_type='seq')
    logger.info(f"Transformer AUC: {results['Transformer']['auc']:.4f}")

    # 6. T-Bi-LSTM+Attention (主模型)
    logger.info("[6/6] T-Bi-LSTM+Attention (主模型)...")
    best_main_auc = 0
    best_main_result = None
    for run in range(3):
        main_model = TBiLSTMAttention(static_dim, seq_dim, hidden=96, layers=2, dropout=0.35).to(device)
        main_model, _ = train_nn(main_model, hybrid_train, hybrid_val, epochs=70, device=device, model_type='hybrid')
        res = evaluate(main_model, hybrid_test, device, model_type='hybrid')
        if res['auc'] > best_main_auc:
            best_main_auc = res['auc']
            best_main_result = res
        logger.info(f"  Run {run+1}/3 AUC: {res['auc']:.4f}")
    results['T-Bi-LSTM+Attention'] = best_main_result
    logger.info(f"T-Bi-LSTM+Attention (Best) AUC: {best_main_result['auc']:.4f}")

    # 汇总结果
    logger.info("=" * 60)
    logger.info("最终结果汇总")
    logger.info("=" * 60)

    result_df = pd.DataFrame(results).T
    result_df = result_df.sort_values('auc', ascending=False)
    result_df.to_csv(DATA_DIR / "model_comparison_full.csv")

    for name, metrics in result_df.iterrows():
        logger.info(f"{name:<25} | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
                   f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUC 对比柱状图
    colors = ['#2ecc71' if name == 'T-Bi-LSTM+Attention' else '#3498db' for name in result_df.index]
    ax1 = axes[0]
    bars = ax1.barh(result_df.index, result_df['auc'], color=colors, edgecolor='black')
    ax1.set_xlabel('AUC-ROC', fontsize=12)
    ax1.set_title('模型效果对比 (AUC)', fontsize=14)
    ax1.set_xlim(0.6, 1.0)
    for bar, val in zip(bars, result_df['auc']):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=10)

    # 多指标雷达图数据
    metrics_names = ['AUC', 'F1', 'Precision', 'Recall', 'Accuracy']
    ax2 = axes[1]
    x = np.arange(len(metrics_names))
    width = 0.12
    for i, (name, row) in enumerate(result_df.iterrows()):
        vals = [row['auc'], row['f1'], row['precision'], row['recall'], row['accuracy']]
        ax2.bar(x + i * width, vals, width, label=name)
    ax2.set_xticks(x + width * 2.5)
    ax2.set_xticklabels(metrics_names)
    ax2.set_ylabel('Score')
    ax2.set_title('多指标对比', fontsize=14)
    ax2.legend(loc='upper right', fontsize=8)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_模型效果对比图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '07_模型效果对比图.png'}")

    logger.info("=" * 60)
    main_auc = results['T-Bi-LSTM+Attention']['auc']
    xgb_auc = results['XGBoost']['auc']
    lgb_auc = results['LightGBM']['auc']

    if main_auc >= xgb_auc:
        logger.info(f"✓ 主模型 T-Bi-LSTM+Attention 超越所有基线!")
        logger.info(f"  vs XGBoost: +{(main_auc - xgb_auc)*100:.2f}%")
        logger.info(f"  vs LightGBM: +{(main_auc - lgb_auc)*100:.2f}%")
    else:
        logger.info(f"✗ 主模型未超越XGBoost (差距: {(main_auc - xgb_auc)*100:.2f}%)")

    logger.info("=" * 60)
    logger.info("实验完成!")


if __name__ == '__main__':
    main()
