# Input: final_features.pkl, X_seq.npy
# Output: 不平衡处理对比结果
# Pos: 不平衡处理策略对比脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
不平衡处理对比实验
- 无处理 (baseline)
- 加权BCE
- Focal Loss (gamma=2)
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
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger_config import get_logger

logger = get_logger("imbalance_exp")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TBiLSTMAttention(nn.Module):
    """主模型: T-Bi-LSTM+Attention"""
    def __init__(self, static_dim, seq_dim, hidden=96, layers=2, dropout=0.35):
        super().__init__()
        self.static_encoder = nn.Sequential(
            nn.BatchNorm1d(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.8)
        )
        self.lstm = nn.LSTM(seq_dim, hidden // 2, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)
        self.time_decay = nn.Parameter(torch.tensor(0.08))
        self.attn_fc = nn.Linear(hidden, 1)
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, 32), nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(32, 1), nn.Sigmoid()
        )
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


class FocalLoss(nn.Module):
    """Focal Loss"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (alpha_t * focal_weight * bce).mean()


def train_model(model, train_loader, val_loader, criterion, epochs=60, device='cuda', use_sampler=True):
    """训练模型"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x_s, x_seq, y in train_loader:
            x_s = x_s.to(device)
            x_seq = x_seq.to(device)
            y = y.to(device).float().unsqueeze(1)

            with autocast('cuda'):
                logits = model(x_s, x_seq)
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
                for x_s, x_seq, y in val_loader:
                    logits = model(x_s.to(device), x_seq.to(device))
                    preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
                    labels.extend(y.numpy())
            auc = roc_auc_score(labels, preds)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, best_auc


def evaluate(model, test_loader, device='cuda'):
    """评估"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x_s, x_seq, y in test_loader:
            logits = model(x_s.to(device), x_seq.to(device))
            preds.extend(torch.sigmoid(logits).cpu().numpy().flatten())
            labels.extend(y.numpy())

    preds = np.array(preds)
    labels = np.array(labels)
    auc = roc_auc_score(labels, preds)

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
        'recall': recall_score(labels, pred_labels, zero_division=0)
    }


def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"
    FIG_DIR = ROOT / "outputs" / "figures"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("不平衡处理策略对比实验")
    logger.info("=" * 60)

    # 加载数据
    df = pd.read_pickle(DATA_DIR / "final_features.pkl")
    feature_cols = [c for c in df.columns if c != 'class']
    X_static = df[feature_cols].values.astype(np.float32)
    y = df['class'].values.astype(np.float32)
    X_seq = np.load(DATA_DIR / "X_seq.npy")

    scaler = StandardScaler()
    X_static = scaler.fit_transform(X_static)

    X_train_s, X_test_s, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_static, X_seq, y, test_size=0.15, random_state=42, stratify=y)
    X_train_s, X_val_s, X_train_seq, X_val_seq, y_train, y_val = train_test_split(
        X_train_s, X_train_seq, y_train, test_size=0.176, random_state=42, stratify=y_train)

    logger.info(f"数据: 训练={len(y_train)}, 验证={len(y_val)}, 测试={len(y_test)}")
    logger.info(f"正负比例: 1:{(1-y_train.mean())/y_train.mean():.1f}")

    static_dim = X_train_s.shape[1]
    seq_dim = X_train_seq.shape[2]

    # 创建 DataLoader
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

    # 类平衡采样器
    pos_w = len(y_train) / (2 * y_train.sum())
    neg_w = len(y_train) / (2 * (len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, pos_w, neg_w)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader_balanced = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    train_loader_unbalanced = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    results = {}

    # 1. 无处理 (No handling)
    logger.info("[1/3] 无不平衡处理...")
    model1 = TBiLSTMAttention(static_dim, seq_dim).to(device)
    criterion1 = nn.BCEWithLogitsLoss()
    model1, _ = train_model(model1, train_loader_unbalanced, val_loader, criterion1, epochs=50, device=device)
    results['无处理'] = evaluate(model1, test_loader, device)
    logger.info(f"无处理 AUC: {results['无处理']['auc']:.4f}")

    # 2. 加权BCE
    logger.info("[2/3] 加权BCE...")
    model2 = TBiLSTMAttention(static_dim, seq_dim).to(device)
    criterion2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device))
    model2, _ = train_model(model2, train_loader_balanced, val_loader, criterion2, epochs=60, device=device)
    results['加权BCE'] = evaluate(model2, test_loader, device)
    logger.info(f"加权BCE AUC: {results['加权BCE']['auc']:.4f}")

    # 3. Focal Loss
    logger.info("[3/3] Focal Loss (gamma=2)...")
    model3 = TBiLSTMAttention(static_dim, seq_dim).to(device)
    criterion3 = FocalLoss(gamma=2.0, alpha=0.25)
    model3, _ = train_model(model3, train_loader_balanced, val_loader, criterion3, epochs=60, device=device)
    results['Focal Loss'] = evaluate(model3, test_loader, device)
    logger.info(f"Focal Loss AUC: {results['Focal Loss']['auc']:.4f}")

    # 汇总
    logger.info("=" * 60)
    logger.info("不平衡处理策略对比结果")
    logger.info("=" * 60)

    result_df = pd.DataFrame(results).T
    result_df.to_csv(DATA_DIR / "imbalance_comparison.csv")

    for name, metrics in result_df.iterrows():
        logger.info(f"{name:<15} | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f} | "
                   f"P: {metrics['precision']:.4f} | R: {metrics['recall']:.4f}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # AUC 对比
    strategies = list(results.keys())
    aucs = [results[s]['auc'] for s in strategies]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ax1 = axes[0]
    bars = ax1.bar(strategies, aucs, color=colors, edgecolor='black')
    ax1.set_ylabel('AUC-ROC', fontsize=12)
    ax1.set_title('不同不平衡处理策略效果对比', fontsize=14)
    ax1.set_ylim(0.7, 0.95)
    for bar, val in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.4f}',
                ha='center', va='bottom', fontsize=11)

    # 多指标对比
    ax2 = axes[1]
    x = np.arange(len(strategies))
    width = 0.2
    metrics_list = ['auc', 'f1', 'precision', 'recall']
    metric_labels = ['AUC', 'F1', 'Precision', 'Recall']
    for i, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        vals = [results[s][metric] for s in strategies]
        ax2.bar(x + i * width, vals, width, label=label)
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(strategies)
    ax2.set_ylabel('Score')
    ax2.set_title('多指标对比', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_不平衡处理策略对比图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '08_不平衡处理策略对比图.png'}")

    logger.info("=" * 60)
    logger.info("实验完成!")


if __name__ == '__main__':
    main()
