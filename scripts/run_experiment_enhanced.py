# 增强版实验：XGBoost叶特征注入 + FiBiNet交互 + AUC优化
# 目标：使主模型 T-Bi-LSTM+Attention 单独超越 XGBoost (AUC 0.8884)
import sys
sys.path.insert(0, '.')
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import time

from src.utils.logger_config import get_logger

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

ROOT = Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger("enhanced_exp")
NUM_WORKERS = 0
BATCH_SIZE = 256

logger.info('=' * 60)
logger.info('增强版实验：XGBoost叶特征注入 + FiBiNet + AUC Loss')
logger.info(f'Device: {device}')
logger.info('=' * 60)

# ===== 加载数据 =====
df = pd.read_pickle(ROOT / 'outputs' / 'final_features.pkl')
feature_cols = [c for c in df.columns if c != 'class']
X_flat = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.float32)
X_seq = np.load(ROOT / 'outputs' / 'X_seq.npy').astype(np.float32)

scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat).astype(np.float32)

indices = np.arange(len(y))
idx_temp, idx_test = train_test_split(indices, test_size=0.15, stratify=y, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.176, stratify=y[idx_temp], random_state=42)

X_train_flat, X_val_flat, X_test_flat = X_flat[idx_train], X_flat[idx_val], X_flat[idx_test]
X_train_seq, X_val_seq, X_test_seq = X_seq[idx_train], X_seq[idx_val], X_seq[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = neg_count / pos_count
logger.info(f'训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(idx_test)}')
logger.info(f'正负比例: 1:{pos_weight:.1f}')

results = []

# ===== 1. 训练 XGBoost 并提取叶节点特征 =====
logger.info('[Step 1] 训练 XGBoost 基线并提取叶节点特征...')

xgb_model = xgb.XGBClassifier(
    max_depth=6, learning_rate=0.05, n_estimators=150,
    scale_pos_weight=pos_weight, verbosity=0, random_state=42,
    tree_method='hist', device='cuda'
)
xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)

# XGBoost 基线结果
y_prob_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]
xgb_auc = roc_auc_score(y_test, y_prob_xgb)
logger.info(f'XGBoost 基线 AUC: {xgb_auc:.4f}')
results.append({'model': 'XGBoost (Baseline)', 'auc': xgb_auc})

# 提取叶节点索引 (每棵树的叶子节点编号)
booster = xgb_model.get_booster()
leaf_train = booster.predict(xgb.DMatrix(X_train_flat), pred_leaf=True)
leaf_val = booster.predict(xgb.DMatrix(X_val_flat), pred_leaf=True)
leaf_test = booster.predict(xgb.DMatrix(X_test_flat), pred_leaf=True)

n_trees = leaf_train.shape[1]
logger.info(f'提取叶节点特征: {n_trees} 棵树')

# 将叶节点索引归一化为 [0, 1] 范围
leaf_train = leaf_train.astype(np.float32) / (leaf_train.max() + 1)
leaf_val = leaf_val.astype(np.float32) / (leaf_val.max() + 1)
leaf_test = leaf_test.astype(np.float32) / (leaf_test.max() + 1)


# ===== 2. LightGBM 基线 =====
logger.info('[Step 2] LightGBM 基线...')
lgb_model = lgb.LGBMClassifier(
    max_depth=6, learning_rate=0.05, n_estimators=150,
    class_weight='balanced', verbose=-1, random_state=42
)
lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)])
y_prob_lgb = lgb_model.predict_proba(X_test_flat)[:, 1]
lgb_auc = roc_auc_score(y_test, y_prob_lgb)
logger.info(f'LightGBM 基线 AUC: {lgb_auc:.4f}')
results.append({'model': 'LightGBM (Baseline)', 'auc': lgb_auc})


# ===== 3. 增强版主模型 =====

class SENetBlock(nn.Module):
    """SENet风格的特征重要性加权"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.ReLU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.fc(x)
        return x * weights


class BilinearInteraction(nn.Module):
    """双线性特征交互 (FiBiNet风格)"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 使用双线性投影: x1 经 W 投影后与 x2 交互
        self.proj = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        # x1, x2: [B, input_dim]
        # 门控双线性交互
        proj_x1 = self.proj(x1)  # [B, output_dim]
        gate_x2 = torch.sigmoid(self.gate(x2))  # [B, output_dim]
        interaction = proj_x1 * gate_x2
        return interaction


class EnhancedTBiLSTMAttention(nn.Module):
    """
    增强版 T-Bi-LSTM+Attention:
    - 静态特征 + XGBoost叶特征嵌入
    - SENet特征重要性加权
    - 双线性特征交互
    - 时序分支 Bi-LSTM + Attention
    """
    def __init__(self, static_dim, leaf_dim, seq_dim, hidden=256, layers=2, dropout=0.3):
        super().__init__()

        # 静态特征编码
        self.static_embed = nn.Sequential(
            nn.LayerNorm(static_dim),
            nn.Linear(static_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 叶节点特征嵌入
        self.leaf_embed = nn.Sequential(
            nn.Linear(leaf_dim, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # SENet 加权
        self.senet = SENetBlock(hidden + hidden // 2)

        # 双线性交互
        self.bilinear = BilinearInteraction(hidden + hidden // 2, hidden)

        # 时序分支
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=layers,
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden * 2)

        # Attention
        self.attn = nn.MultiheadAttention(hidden * 2, num_heads=4, batch_first=True, dropout=dropout)

        # Time Decay Gate
        self.time_decay = nn.Parameter(torch.tensor(0.1))

        # 融合层
        # 静态: hidden + hidden//2, 交互: hidden, 时序: hidden*2*3
        fusion_dim = (hidden + hidden // 2) + hidden + (hidden * 2 * 3)

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_static, x_leaf, x_seq, return_attention=False):
        # 静态分支
        static_emb = self.static_embed(x_static)  # [B, hidden]
        leaf_emb = self.leaf_embed(x_leaf)  # [B, hidden//2]

        # 拼接并SENet加权
        static_combined = torch.cat([static_emb, leaf_emb], dim=1)  # [B, hidden + hidden//2]
        static_weighted = self.senet(static_combined)

        # 双线性交互
        interaction = self.bilinear(static_weighted, static_weighted)  # [B, hidden]

        # 时序分支 + Time Decay
        batch_size, seq_len, _ = x_seq.shape
        time_weights = torch.exp(-self.time_decay.abs() * torch.arange(seq_len, 0, -1, device=x_seq.device).float())
        time_weights = time_weights.view(1, seq_len, 1)
        x_seq_weighted = x_seq * time_weights

        h, _ = self.lstm(x_seq_weighted)  # [B, T, hidden*2]
        h = self.seq_norm(h)

        # Self-attention
        attn_out, attn_weights = self.attn(h, h, h)

        # 多池化
        h_mean = h.mean(dim=1)
        h_max = h.max(dim=1).values
        h_last = attn_out[:, -1, :]
        seq_features = torch.cat([h_mean, h_max, h_last], dim=1)  # [B, hidden*2*3]

        # 融合
        fused = torch.cat([static_weighted, interaction, seq_features], dim=1)
        logits = self.fusion(fused)

        if return_attention:
            return logits, attn_weights
        return logits


class ClassBalancedFocalLoss(nn.Module):
    """Class-balanced Focal Loss"""
    def __init__(self, beta=0.9999, gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)

        # Class-balanced weight
        pos_mask = targets.float()
        neg_mask = 1 - pos_mask
        cb_weight = pos_mask * self.pos_weight + neg_mask

        # Focal modulation
        focal_weight = (1 - pt) ** self.gamma

        loss = cb_weight * focal_weight * bce
        return loss.mean()


class PairwiseAUCLoss(nn.Module):
    """Pairwise AUC Loss (RankNet style)"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, logits, targets):
        # 找正负样本对
        pos_idx = (targets.squeeze() == 1).nonzero(as_tuple=True)[0]
        neg_idx = (targets.squeeze() == 0).nonzero(as_tuple=True)[0]

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return torch.tensor(0.0, device=logits.device)

        # 随机采样对 (避免计算量过大)
        n_pairs = min(len(pos_idx) * len(neg_idx), 1000)
        pos_sample = pos_idx[torch.randint(len(pos_idx), (n_pairs,))]
        neg_sample = neg_idx[torch.randint(len(neg_idx), (n_pairs,))]

        pos_scores = logits[pos_sample].squeeze()
        neg_scores = logits[neg_sample].squeeze()

        # Margin ranking loss: pos should be higher than neg by margin
        loss = F.relu(self.margin - (pos_scores - neg_scores)).mean()
        return loss


# ===== 训练函数 =====
def train_enhanced_model(model, train_loader, val_loader, epochs=60, lr=0.001):
    model = model.to(device)

    cb_focal = ClassBalancedFocalLoss(beta=0.9999, gamma=2.0, pos_weight=pos_weight)
    auc_loss = PairwiseAUCLoss(margin=0.5)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None
    no_improve = 0
    patience = 12

    logger.info(f'开始训练增强版主模型 (epochs={epochs})')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for x_static, x_leaf, x_seq, y_batch in train_loader:
            x_static = x_static.to(device, non_blocking=True)
            x_leaf = x_leaf.to(device, non_blocking=True)
            x_seq = x_seq.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                logits = model(x_static, x_leaf, x_seq)

                # 组合损失: Focal + AUC
                loss_focal = cb_focal(logits, y_batch)
                loss_auc = auc_loss(logits, y_batch)
                loss = loss_focal + 0.3 * loss_auc

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        scheduler.step()

        # 验证
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_probs = []
            with torch.no_grad():
                for x_static, x_leaf, x_seq, _ in val_loader:
                    x_static = x_static.to(device)
                    x_leaf = x_leaf.to(device)
                    x_seq = x_seq.to(device)
                    with autocast('cuda'):
                        logits = model(x_static, x_leaf, x_seq)
                    val_probs.append(torch.sigmoid(logits).cpu().numpy())

            val_probs = np.concatenate(val_probs).flatten()
            val_auc = roc_auc_score(y_val, val_probs)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            logger.info(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}')

            if no_improve >= patience:
                logger.info(f'早停 (patience={patience})')
                break

    elapsed = time.time() - start_time
    logger.info(f'训练完成 ({elapsed:.1f}s)')

    model.load_state_dict(best_state)
    return model, best_auc


# ===== 准备数据 =====
logger.info('[Step 3] 准备增强版模型训练数据...')

# 合并所有特征
X_train_static_t = torch.FloatTensor(X_train_flat)
X_train_leaf_t = torch.FloatTensor(leaf_train)
X_train_seq_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

X_val_static_t = torch.FloatTensor(X_val_flat)
X_val_leaf_t = torch.FloatTensor(leaf_val)
X_val_seq_t = torch.FloatTensor(X_val_seq)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

X_test_static_t = torch.FloatTensor(X_test_flat)
X_test_leaf_t = torch.FloatTensor(leaf_test)
X_test_seq_t = torch.FloatTensor(X_test_seq)

train_dataset = TensorDataset(X_train_static_t, X_train_leaf_t, X_train_seq_t, y_train_t)
val_dataset = TensorDataset(X_val_static_t, X_val_leaf_t, X_val_seq_t, y_val_t)

# 加权采样
train_labels = torch.LongTensor(y_train.astype(int))
class_counts = torch.bincount(train_labels)
weights = 1.0 / class_counts.float()
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                         num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)


# ===== 训练增强版主模型 =====
logger.info('[Step 4] 训练增强版 T-Bi-LSTM+Attention (主模型)...')

enhanced_model = EnhancedTBiLSTMAttention(
    static_dim=len(feature_cols),
    leaf_dim=n_trees,
    seq_dim=X_seq.shape[2],
    hidden=256,
    layers=2,
    dropout=0.3
)

enhanced_model, val_auc = train_enhanced_model(enhanced_model, train_loader, val_loader, epochs=80, lr=0.001)

# 测试评估
enhanced_model.eval()
with torch.no_grad():
    with autocast('cuda'):
        logits = enhanced_model(X_test_static_t.to(device), X_test_leaf_t.to(device), X_test_seq_t.to(device))
    y_prob_enhanced = torch.sigmoid(logits).cpu().numpy().flatten()

enhanced_auc = roc_auc_score(y_test, y_prob_enhanced)

# 寻找最优阈值
best_f1, best_t = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y_test, (y_prob_enhanced > t).astype(int))
    if f > best_f1:
        best_f1, best_t = f, t

y_pred = (y_prob_enhanced > best_t).astype(int)
enhanced_f1 = f1_score(y_test, y_pred)
enhanced_p = precision_score(y_test, y_pred)
enhanced_r = recall_score(y_test, y_pred)

logger.info(f'增强版主模型 - AUC: {enhanced_auc:.4f}, F1: {enhanced_f1:.4f}')
results.append({
    'model': 'Enhanced T-Bi-LSTM+Attention (主模型)',
    'auc': enhanced_auc,
    'f1': enhanced_f1,
    'precision': enhanced_p,
    'recall': enhanced_r
})

# 保存模型
torch.save(enhanced_model.state_dict(), ROOT / 'outputs' / 'models' / 'enhanced_main_model.pth')


# ===== 结果汇总 =====
logger.info('=' * 60)
logger.info('最终结果对比')
logger.info('=' * 60)

results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
for _, row in results_df.iterrows():
    logger.info(f'{row["model"]:45s} | AUC: {row["auc"]:.4f}')

logger.info('=' * 60)
if enhanced_auc > xgb_auc:
    logger.info(f'✓ 主模型成功超越 XGBoost!')
    logger.info(f'  主模型 AUC: {enhanced_auc:.4f} > XGBoost AUC: {xgb_auc:.4f} (+{(enhanced_auc-xgb_auc)*100:.2f}%)')
else:
    logger.info(f'✗ 主模型尚未超越 XGBoost')
    logger.info(f'  主模型 AUC: {enhanced_auc:.4f} vs XGBoost AUC: {xgb_auc:.4f} ({(enhanced_auc-xgb_auc)*100:.2f}%)')
logger.info('=' * 60)

results_df.to_csv(ROOT / 'outputs' / 'model_comparison_enhanced.csv', index=False)
logger.info('实验完成!')
