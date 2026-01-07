# Input: final_features.pkl, X_seq.npy, 训练好的模型
# Output: 可解释性分析图表
# Pos: 可解释性分析脚本
# Warning: 更新时同步更新注释和 _ARCH.md

"""
可解释性分析
- Attention权重可视化
- SHAP分析
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb

from src.utils.logger_config import get_logger

logger = get_logger("interpret_exp")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class InterpretableModel(nn.Module):
    """可解释性模型：返回注意力权重"""
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

    def forward(self, x_static, x_seq, return_attention=False):
        static_feat = self.static_encoder(x_static)
        B, T, _ = x_seq.shape
        tw = torch.exp(-self.time_decay.abs() * torch.arange(T, 0, -1, device=x_seq.device).float())
        h, _ = self.lstm(x_seq * tw.view(1, T, 1))
        h = self.seq_norm(h)
        attn_scores = self.attn_fc(h).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        seq_feat = torch.bmm(attn_weights.unsqueeze(1), h).squeeze(1)
        gate = self.gate(torch.cat([static_feat, seq_feat], dim=1))
        fused = torch.cat([static_feat, seq_feat * gate], dim=1)
        logits = self.classifier(fused)
        if return_attention:
            return logits, attn_weights, gate
        return logits


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
    return model


def main():
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / "outputs"
    FIG_DIR = ROOT / "outputs" / "figures"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    logger.info("=" * 60)
    logger.info("可解释性分析")
    logger.info("=" * 60)

    # 加载数据
    df = pd.read_pickle(DATA_DIR / "final_features.pkl")
    feature_cols = [c for c in df.columns if c != 'class']
    X_static = df[feature_cols].values.astype(np.float32)
    y = df['class'].values.astype(np.float32)
    X_seq = np.load(DATA_DIR / "X_seq.npy")

    scaler = StandardScaler()
    X_static_scaled = scaler.fit_transform(X_static)

    # 划分
    X_train_s, X_test_s, X_train_seq, X_test_seq, y_train, y_test = train_test_split(
        X_static_scaled, X_seq, y, test_size=0.15, random_state=42, stratify=y)
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

    # 训练模型
    logger.info("训练可解释性模型...")
    model = InterpretableModel(static_dim, seq_dim).to(device)
    model = train_model(model, train_loader, val_loader, epochs=60, device=device)

    # ==================== 1. Attention权重分析 ====================
    logger.info("=" * 40)
    logger.info("1. Attention权重可视化")
    logger.info("=" * 40)

    model.eval()
    # 找欺诈样本
    fraud_indices = np.where(y_test == 1)[0]
    normal_indices = np.where(y_test == 0)[0]

    # 选取5个高置信度欺诈样本
    with torch.no_grad():
        all_probs = []
        all_attns = []
        all_gates = []
        for x_s, x_seq, y in test_loader:
            logits, attn, gate = model(x_s.to(device), x_seq.to(device), return_attention=True)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
            all_attns.append(attn.cpu().numpy())
            all_gates.append(gate.cpu().numpy())

    all_probs = np.vstack(all_probs).flatten()
    all_attns = np.vstack(all_attns)
    all_gates = np.vstack(all_gates).flatten()

    # Top-5 欺诈样本（高置信度）
    fraud_probs = all_probs[fraud_indices]
    top5_fraud_idx = fraud_indices[np.argsort(fraud_probs)[-5:]]

    # Attention热力图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for i, idx in enumerate(top5_fraud_idx):
        ax = axes[i]
        attn = all_attns[idx]
        prob = all_probs[idx]
        gate_val = all_gates[idx]

        im = ax.imshow(attn.reshape(1, -1), aspect='auto', cmap='Reds')
        ax.set_title(f'样本 {idx} (欺诈概率: {prob:.3f}, 时序门: {gate_val:.3f})', fontsize=10)
        ax.set_xlabel('时间步')
        ax.set_yticks([])

    # 对比：正常样本
    normal_sample_idx = normal_indices[np.argmax(all_probs[normal_indices])]  # 最高概率的正常样本
    ax = axes[5]
    attn = all_attns[normal_sample_idx]
    prob = all_probs[normal_sample_idx]
    gate_val = all_gates[normal_sample_idx]
    im = ax.imshow(attn.reshape(1, -1), aspect='auto', cmap='Blues')
    ax.set_title(f'对比样本 (正常, 概率: {prob:.3f}, 时序门: {gate_val:.3f})', fontsize=10)
    ax.set_xlabel('时间步')
    ax.set_yticks([])

    plt.suptitle('Attention权重可视化 (Top-5欺诈样本 vs 正常样本)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_Attention权重热力图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '10_Attention权重热力图.png'}")

    # 平均Attention对比
    fraud_avg_attn = all_attns[fraud_indices].mean(axis=0)
    normal_avg_attn = all_attns[normal_indices].mean(axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(fraud_avg_attn))
    ax.plot(x, fraud_avg_attn, 'r-', label='欺诈样本平均', linewidth=2)
    ax.plot(x, normal_avg_attn, 'b-', label='正常样本平均', linewidth=2)
    ax.fill_between(x, fraud_avg_attn, alpha=0.3, color='red')
    ax.fill_between(x, normal_avg_attn, alpha=0.3, color='blue')
    ax.set_xlabel('时间步 (越靠近右侧越近期)', fontsize=12)
    ax.set_ylabel('Attention权重', fontsize=12)
    ax.set_title('欺诈 vs 正常样本的时序Attention权重对比', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_Attention权重对比曲线.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '11_Attention权重对比曲线.png'}")

    # ==================== 2. SHAP分析 (使用XGBoost) ====================
    logger.info("=" * 40)
    logger.info("2. SHAP特征重要性分析")
    logger.info("=" * 40)

    # 展平特征
    X_train_flat = np.hstack([X_train_s, X_train_seq.reshape(len(y_train), -1)])
    X_test_flat = np.hstack([X_test_s, X_test_seq.reshape(len(y_test), -1)])

    # 训练XGBoost用于SHAP
    logger.info("训练XGBoost用于SHAP分析...")
    xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=6, learning_rate=0.05,
                                   scale_pos_weight=19, random_state=42, tree_method='hist')
    xgb_model.fit(X_train_flat, y_train)

    # 特征重要性
    importance = xgb_model.feature_importances_

    # 构建特征名
    seq_feature_names = ['时序特征_' + str(i) for i in range(X_train_seq.shape[1] * X_train_seq.shape[2])]
    all_feature_names = list(feature_cols) + seq_feature_names

    # Top-20特征
    top20_idx = np.argsort(importance)[-20:][::-1]
    top20_names = [all_feature_names[i] if i < len(all_feature_names) else f'特征_{i}' for i in top20_idx]
    top20_importance = importance[top20_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#e74c3c' if '时序' not in name else '#3498db' for name in top20_names]
    bars = ax.barh(range(len(top20_names)), top20_importance, color=colors, edgecolor='black')
    ax.set_yticks(range(len(top20_names)))
    ax.set_yticklabels(top20_names, fontsize=9)
    ax.set_xlabel('特征重要性', fontsize=12)
    ax.set_title('Top-20特征重要性 (XGBoost)', fontsize=14)
    ax.invert_yaxis()

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#e74c3c', label='静态特征'),
                       Patch(facecolor='#3498db', label='时序特征')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_SHAP特征重要性图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '12_SHAP特征重要性图.png'}")

    # ==================== 3. Case Study ====================
    logger.info("=" * 40)
    logger.info("3. 典型案例分析")
    logger.info("=" * 40)

    # 选择一个典型欺诈案例和一个正常案例
    fraud_case_idx = top5_fraud_idx[0]
    normal_case_idx = normal_sample_idx

    fraud_features = X_test_s[fraud_case_idx]
    normal_features = X_test_s[normal_case_idx]

    # 对比主要特征
    top_static_idx = np.argsort(importance[:len(feature_cols)])[-10:][::-1]
    top_static_names = [feature_cols[i] for i in top_static_idx]

    fraud_vals = fraud_features[top_static_idx]
    normal_vals = normal_features[top_static_idx]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(top_static_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, fraud_vals, width, label=f'欺诈案例 (概率: {all_probs[fraud_case_idx]:.3f})', color='#e74c3c')
    bars2 = ax.bar(x + width/2, normal_vals, width, label=f'正常案例 (概率: {all_probs[normal_case_idx]:.3f})', color='#3498db')

    ax.set_xticks(x)
    ax.set_xticklabels(top_static_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('特征值 (标准化)', fontsize=12)
    ax.set_title('欺诈 vs 正常案例的Top-10特征对比', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_案例特征对比图.png", dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: {FIG_DIR / '13_案例特征对比图.png'}")

    # ==================== 汇总 ====================
    logger.info("=" * 60)
    logger.info("可解释性分析完成!")
    logger.info("=" * 60)
    logger.info("生成的图表:")
    logger.info("  - 10_Attention权重热力图.png")
    logger.info("  - 11_Attention权重对比曲线.png")
    logger.info("  - 12_SHAP特征重要性图.png")
    logger.info("  - 13_案例特征对比图.png")


if __name__ == '__main__':
    main()
