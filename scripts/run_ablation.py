# Input: final_features.pkl, X_seq.npy
# Output: 消融实验结果
# Pos: 模块消融和特征消融实验脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
消融实验
A. 模块消融:
  - A1: T-Bi-LSTM+Attention (完整)
  - A2: Bi-LSTM+Attention (去除时间衰减)
  - A3: Bi-LSTM+Attention+标准位置编码
  - A4: Bi-LSTM+Attention+简单线性衰减

B. 特征消融:
  - B1: 完整特征(静态+时序)
  - B2: 仅静态特征
  - B3: 仅时序特征
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
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.logger_config import get_logger

logger = get_logger("ablation_exp")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ==================== 模块消融模型变体 ====================

class A1_Full(nn.Module):
    """A1: 完整模型 T-Bi-LSTM+Attention (带时间衰减门)"""
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
        self.time_decay = nn.Parameter(torch.tensor(0.08))  # 可学习时间衰减
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
        # 指数时间衰减
        tw = torch.exp(-self.time_decay.abs() * torch.arange(T, 0, -1, device=x_seq.device).float())
        h, _ = self.lstm(x_seq * tw.view(1, T, 1))
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        return self.classifier(fused)


class A2_NoTimeDecay(nn.Module):
    """A2: 去除时间衰减"""
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
        # 无时间衰减
        h, _ = self.lstm(x_seq)
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        return self.classifier(fused)


class A3_StdPosEncoding(nn.Module):
    """A3: 标准正弦位置编码"""
    def __init__(self, static_dim, seq_dim, hidden=96, layers=2, dropout=0.35, max_len=100):
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
        self.seq_embed = nn.Linear(seq_dim, hidden)
        # 标准位置编码
        pe = torch.zeros(max_len, hidden)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, hidden, 2).float() * (-np.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:hidden//2])
        self.register_buffer('pe', pe.unsqueeze(0))

        self.lstm = nn.LSTM(hidden, hidden // 2, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)
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
        # 嵌入 + 位置编码
        x_emb = self.seq_embed(x_seq) + self.pe[:, :T, :]
        h, _ = self.lstm(x_emb)
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        return self.classifier(fused)


class A4_LinearDecay(nn.Module):
    """A4: 简单线性衰减"""
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
        # 线性衰减: 1 - t/T
        tw = torch.linspace(1, 0, T, device=x_seq.device)
        h, _ = self.lstm(x_seq * tw.view(1, T, 1))
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        return self.classifier(fused)


# ==================== 特征消融模型 ====================

class StaticOnlyModel(nn.Module):
    """B2: 仅静态特征"""
    def __init__(self, static_dim, hidden=128, dropout=0.35):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x_static, x_seq=None):
        return self.net(x_static)


class SeqOnlyModel(nn.Module):
    """B3: 仅时序特征"""
    def __init__(self, seq_dim, hidden=96, layers=2, dropout=0.35):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden // 2, layers, batch_first=True,
                           bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden)
        self.time_decay = nn.Parameter(torch.tensor(0.08))
        self.attn_fc = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x_static, x_seq):
        B, T, _ = x_seq.shape
        tw = torch.exp(-self.time_decay.abs() * torch.arange(T, 0, -1, device=x_seq.device).float())
        h, _ = self.lstm(x_seq * tw.view(1, T, 1))
        h = self.seq_norm(h)
        attn = F.softmax(self.attn_fc(h).squeeze(-1), dim=1)
        seq_feat = torch.bmm(attn.unsqueeze(1), h).squeeze(1)
        return self.classifier(seq_feat)


# ==================== 训练和评估 ====================

def train_model(model, train_loader, val_loader, epochs=60, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([15.0]).to(device))
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x_s, x_seq, y in train_loader:
            x_s, x_seq = x_s.to(device), x_seq.to(device)
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
    best_f1 = max([f1_score(labels, (preds >= t).astype(int), zero_division=0)
                   for t in np.arange(0.1, 0.9, 0.02)])
    return {'auc': auc, 'f1': best_f1}


def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"
    FIG_DIR = ROOT / "outputs" / "figures"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("消融实验 (模块消融 + 特征消融)")
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

    static_dim = X_train_s.shape[1]
    seq_dim = X_train_seq.shape[2]

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_train_s), torch.FloatTensor(X_train_seq),
                                  torch.LongTensor(y_train.astype(int)))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_s), torch.FloatTensor(X_val_seq),
                                torch.LongTensor(y_val.astype(int)))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_s), torch.FloatTensor(X_test_seq),
                                 torch.LongTensor(y_test.astype(int)))

    pos_w = len(y_train) / (2 * y_train.sum())
    neg_w = len(y_train) / (2 * (len(y_train) - y_train.sum()))
    sample_weights = np.where(y_train == 1, pos_w, neg_w)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=256, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # A. 模块消融
    logger.info("=" * 40)
    logger.info("A. 模块消融实验")
    logger.info("=" * 40)

    module_results = {}

    logger.info("[A1] T-Bi-LSTM+Attention (完整)...")
    m1 = A1_Full(static_dim, seq_dim).to(device)
    m1, _ = train_model(m1, train_loader, val_loader, epochs=60, device=device)
    module_results['A1: 完整模型'] = evaluate(m1, test_loader, device)
    logger.info(f"A1 AUC: {module_results['A1: 完整模型']['auc']:.4f}")

    logger.info("[A2] Bi-LSTM+Attention (去除时间衰减)...")
    m2 = A2_NoTimeDecay(static_dim, seq_dim).to(device)
    m2, _ = train_model(m2, train_loader, val_loader, epochs=60, device=device)
    module_results['A2: 去除时间衰减'] = evaluate(m2, test_loader, device)
    logger.info(f"A2 AUC: {module_results['A2: 去除时间衰减']['auc']:.4f}")

    logger.info("[A3] Bi-LSTM+Attention+标准位置编码...")
    m3 = A3_StdPosEncoding(static_dim, seq_dim).to(device)
    m3, _ = train_model(m3, train_loader, val_loader, epochs=60, device=device)
    module_results['A3: 标准位置编码'] = evaluate(m3, test_loader, device)
    logger.info(f"A3 AUC: {module_results['A3: 标准位置编码']['auc']:.4f}")

    logger.info("[A4] Bi-LSTM+Attention+线性衰减...")
    m4 = A4_LinearDecay(static_dim, seq_dim).to(device)
    m4, _ = train_model(m4, train_loader, val_loader, epochs=60, device=device)
    module_results['A4: 线性衰减'] = evaluate(m4, test_loader, device)
    logger.info(f"A4 AUC: {module_results['A4: 线性衰减']['auc']:.4f}")

    # B. 特征消融
    logger.info("=" * 40)
    logger.info("B. 特征消融实验")
    logger.info("=" * 40)

    feature_results = {}

    logger.info("[B1] 完整特征 (静态+时序)...")
    feature_results['B1: 完整特征'] = module_results['A1: 完整模型']  # 同 A1
    logger.info(f"B1 AUC: {feature_results['B1: 完整特征']['auc']:.4f}")

    logger.info("[B2] 仅静态特征...")
    m_static = StaticOnlyModel(static_dim).to(device)
    m_static, _ = train_model(m_static, train_loader, val_loader, epochs=60, device=device)
    feature_results['B2: 仅静态特征'] = evaluate(m_static, test_loader, device)
    logger.info(f"B2 AUC: {feature_results['B2: 仅静态特征']['auc']:.4f}")

    logger.info("[B3] 仅时序特征...")
    m_seq = SeqOnlyModel(seq_dim).to(device)
    m_seq, _ = train_model(m_seq, train_loader, val_loader, epochs=60, device=device)
    feature_results['B3: 仅时序特征'] = evaluate(m_seq, test_loader, device)
    logger.info(f"B3 AUC: {feature_results['B3: 仅时序特征']['auc']:.4f}")

    # 汇总
    logger.info("=" * 60)
    logger.info("消融实验结果汇总")
    logger.info("=" * 60)

    all_results = {**module_results, **feature_results}
    result_df = pd.DataFrame(all_results).T
    result_df.to_csv(DATA_DIR / "ablation_results.csv")

    for name, metrics in result_df.iterrows():
        logger.info(f"{name:<25} | AUC: {metrics['auc']:.4f} | F1: {metrics['f1']:.4f}")

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 模块消融图
    ax1 = axes[0]
    names = list(module_results.keys())
    aucs = [module_results[n]['auc'] for n in names]
    colors = ['#2ecc71' if 'A1' in n else '#3498db' for n in names]
    bars = ax1.barh(names, aucs, color=colors, edgecolor='black')
    ax1.set_xlabel('AUC-ROC', fontsize=12)
    ax1.set_title('模块消融实验结果', fontsize=14)
    ax1.set_xlim(0.8, 0.92)
    for bar, val in zip(bars, aucs):
        ax1.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')

    # 特征消融图
    ax2 = axes[1]
    names = list(feature_results.keys())
    aucs = [feature_results[n]['auc'] for n in names]
    colors = ['#2ecc71' if 'B1' in n else '#3498db' for n in names]
    bars = ax2.barh(names, aucs, color=colors, edgecolor='black')
    ax2.set_xlabel('AUC-ROC', fontsize=12)
    ax2.set_title('特征消融实验结果', fontsize=14)
    ax2.set_xlim(0.6, 0.92)
    for bar, val in zip(bars, aucs):
        ax2.text(val + 0.003, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_消融实验结果图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '09_消融实验结果图.png'}")

    logger.info("=" * 60)
    logger.info("实验完成!")


if __name__ == '__main__':
    main()
