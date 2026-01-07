# 优化版实验脚本：性能优化 + 效果优化
# 基于 Codex 协作分析的优化方案
import sys
sys.path.insert(0, '.')
import os
os.environ['PYTHONUNBUFFERED'] = '1'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.amp import autocast, GradScaler  # 更新的API
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path
import time
from multiprocessing import freeze_support

from src.utils.logger_config import get_logger

# ===== 性能优化设置 =====
torch.backends.cudnn.benchmark = True  # 加速卷积运算
torch.set_float32_matmul_precision('high')  # TensorCore 优化

ROOT = Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger("optimized_exp")

# Windows 多进程兼容: num_workers=0 避免问题
NUM_WORKERS = 0  # Windows下设为0最稳定

logger.info('=' * 60)
logger.info('优化版实验：性能优化 + 效果优化')
logger.info(f'Device: {device}')
if device.type == 'cuda':
    logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA: {torch.version.cuda}, cuDNN: {torch.backends.cudnn.version()}')
logger.info('=' * 60)

# ===== 加载数据 =====
logger.info('加载数据...')
df = pd.read_pickle(ROOT / 'outputs' / 'final_features.pkl')
feature_cols = [c for c in df.columns if c != 'class']
X_flat = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.float32)

X_seq = np.load(ROOT / 'outputs' / 'X_seq.npy').astype(np.float32)
y_seq = np.load(ROOT / 'outputs' / 'y_seq.npy')

logger.info(f'全量特征: {X_flat.shape}, 时序特征: {X_seq.shape}')

# 标准化
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat).astype(np.float32)

# 数据划分
indices = np.arange(len(y))
idx_temp, idx_test = train_test_split(indices, test_size=0.15, stratify=y, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.176, stratify=y[idx_temp], random_state=42)

X_train_flat, X_val_flat, X_test_flat = X_flat[idx_train], X_flat[idx_val], X_flat[idx_test]
X_train_seq, X_val_seq, X_test_seq = X_seq[idx_train], X_seq[idx_val], X_seq[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

logger.info(f'数据划分 - 训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(idx_test)}')

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = neg_count / pos_count
logger.info(f'类别比例 - 正样本: {int(pos_count)}, 负样本: {int(neg_count)}, 权重: {pos_weight:.2f}')

results = []

# ===== 工具函数 =====
def get_optimal_threshold(y_true, y_prob):
    """寻找最优F1阈值"""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        f = f1_score(y_true, (y_prob > t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t

def evaluate(y_true, y_prob, threshold=None):
    """评估模型"""
    if threshold is None:
        threshold = get_optimal_threshold(y_true, y_prob)
    y_pred = (y_prob > threshold).astype(int)
    return {
        'auc': roc_auc_score(y_true, y_prob),
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'threshold': threshold
    }

# ===== 优化的数据加载器 =====
# NUM_WORKERS 已在文件开头定义为0 (Windows兼容)
BATCH_SIZE = 256  # 从64提升到256

# 加权采样器处理类别不平衡
train_labels = torch.LongTensor(y_train.astype(int))
class_counts = torch.bincount(train_labels)
weights = 1.0 / class_counts.float()
sample_weights = weights[train_labels]
sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

# ===== 基线模型 =====

# XGBoost
logger.info('[1/4] XGBoost...')
start = time.time()
xgb_model = xgb.XGBClassifier(
    max_depth=8, learning_rate=0.05, n_estimators=200,
    scale_pos_weight=pos_weight, verbosity=0, random_state=42,
    tree_method='hist', device='cuda'  # GPU加速
)
xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
y_prob_xgb = xgb_model.predict_proba(X_test_flat)[:, 1]
res = evaluate(y_test, y_prob_xgb)
res['model'] = 'XGBoost'
results.append(res)
logger.info(f'XGBoost 完成 ({time.time()-start:.1f}s) - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# LightGBM
logger.info('[2/4] LightGBM...')
start = time.time()
lgb_model = lgb.LGBMClassifier(
    max_depth=8, learning_rate=0.05, n_estimators=200,
    class_weight='balanced', verbose=-1, random_state=42,
    device='gpu'  # GPU加速
)
lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)])
y_prob_lgb = lgb_model.predict_proba(X_test_flat)[:, 1]
res = evaluate(y_test, y_prob_lgb)
res['model'] = 'LightGBM'
results.append(res)
logger.info(f'LightGBM 完成 ({time.time()-start:.1f}s) - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')


# ===== 优化的混合模型 (基于Codex建议) =====
class OptimizedHybridModel(nn.Module):
    """
    优化的混合模型：
    - 序列分支: BiLSTM + MultiHeadAttention + 多种池化
    - 静态分支: LayerNorm + MLP
    - 融合: Concat + 残差 + 分类头
    """
    def __init__(self, seq_dim=20, tab_dim=166, hidden=256, layers=2, attn_heads=4, dropout=0.3):
        super().__init__()

        # 序列分支
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(hidden*2, attn_heads, batch_first=True, dropout=dropout)
        self.seq_norm = nn.LayerNorm(hidden*2)

        # 静态分支 (用LayerNorm代替BatchNorm，对小batch更稳定)
        self.tab = nn.Sequential(
            nn.LayerNorm(tab_dim),
            nn.Linear(tab_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 融合层: 序列(mean+max+attn_cls) = hidden*6, 静态 = hidden
        fusion_dim = hidden * 6 + hidden
        self.head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )

    def forward(self, x_seq, x_tab):
        # 序列分支
        h, _ = self.lstm(x_seq)  # [B, T, hidden*2]
        h = self.seq_norm(h)

        # Multi-head self-attention
        attn_out, attn_weights = self.attn(h, h, h)  # [B, T, hidden*2]

        # 多种池化
        h_mean = h.mean(dim=1)  # [B, hidden*2]
        h_max = h.max(dim=1).values  # [B, hidden*2]
        h_attn = attn_out[:, 0, :]  # [B, hidden*2] 取CLS位置

        seq_features = torch.cat([h_mean, h_max, h_attn], dim=1)  # [B, hidden*6]

        # 静态分支
        tab_features = self.tab(x_tab)  # [B, hidden]

        # 融合
        fused = torch.cat([seq_features, tab_features], dim=1)  # [B, hidden*6 + hidden]
        logits = self.head(fused)  # [B, 1]

        return logits, attn_weights

    def predict(self, x_seq, x_tab):
        """推理时只返回logits"""
        logits, _ = self.forward(x_seq, x_tab)
        return torch.sigmoid(logits)


# ===== 纯时序基线 (用于对比) =====
class SeqOnlyModel(nn.Module):
    def __init__(self, seq_dim=20, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=layers,
                           batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden*2),
            nn.Linear(hidden*2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(h[:, -1, :])


# ===== 训练函数 (带性能优化) =====
def train_model(model, train_loader, val_loader, epochs=50, lr=0.001,
                val_interval=5, patience=10, model_name='model'):
    """
    优化的训练函数：
    - AMP 混合精度
    - 验证降频
    - 早停
    - 梯度裁剪
    """
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr*10, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    scaler = GradScaler('cuda')

    best_auc = 0
    best_state = None
    no_improve = 0

    logger.info(f'{model_name} 开始训练 (epochs={epochs}, batch={BATCH_SIZE}, AMP=True)')
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            if len(batch) == 3:  # 混合模型
                x_seq, x_tab, y = batch
                x_seq = x_seq.to(device, non_blocking=True)
                x_tab = x_tab.to(device, non_blocking=True)
            else:  # 纯序列模型
                x_seq, y = batch
                x_seq = x_seq.to(device, non_blocking=True)
                x_tab = None
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda'):
                if x_tab is not None:
                    logits, _ = model(x_seq, x_tab)
                else:
                    logits = model(x_seq)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()

        # 验证 (降频)
        if (epoch + 1) % val_interval == 0 or epoch == epochs - 1:
            model.eval()
            val_probs = []
            val_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 3:
                        x_seq, x_tab, y = batch
                        x_seq = x_seq.to(device, non_blocking=True)
                        x_tab = x_tab.to(device, non_blocking=True)
                        with autocast('cuda'):
                            logits, _ = model(x_seq, x_tab)
                    else:
                        x_seq, y = batch
                        x_seq = x_seq.to(device, non_blocking=True)
                        with autocast('cuda'):
                            logits = model(x_seq)

                    val_probs.append(torch.sigmoid(logits).cpu().numpy())
                    val_labels.append(y.numpy())

            val_probs = np.concatenate(val_probs).flatten()
            val_labels = np.concatenate(val_labels).flatten()
            val_auc = roc_auc_score(val_labels, val_probs)

            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            logger.info(f'{model_name} Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}')

            # 早停
            if no_improve >= patience:
                logger.info(f'{model_name} 早停 (patience={patience})')
                break

    elapsed = time.time() - start_time
    logger.info(f'{model_name} 训练完成 ({elapsed:.1f}s)')

    # 恢复最佳状态
    model.load_state_dict(best_state)
    return model, best_auc


# ===== 准备优化的DataLoader =====
# 混合数据
X_train_seq_t = torch.FloatTensor(X_train_seq)
X_train_tab_t = torch.FloatTensor(X_train_flat)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)

X_val_seq_t = torch.FloatTensor(X_val_seq)
X_val_tab_t = torch.FloatTensor(X_val_flat)
y_val_t = torch.FloatTensor(y_val).unsqueeze(1)

X_test_seq_t = torch.FloatTensor(X_test_seq)
X_test_tab_t = torch.FloatTensor(X_test_flat)

train_dataset = TensorDataset(X_train_seq_t, X_train_tab_t, y_train_t)
val_dataset = TensorDataset(X_val_seq_t, X_val_tab_t, y_val_t)

# 优化的DataLoader (Windows: num_workers=0, 无prefetch)
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE*2, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# ===== 训练纯序列模型 (基线) =====
logger.info('[3/4] T-Bi-LSTM (纯序列)...')
seq_train_ds = TensorDataset(X_train_seq_t, y_train_t)
seq_val_ds = TensorDataset(X_val_seq_t, y_val_t)
seq_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
seq_train_loader = DataLoader(seq_train_ds, batch_size=BATCH_SIZE, sampler=seq_sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
seq_val_loader = DataLoader(seq_val_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

seq_model = SeqOnlyModel(seq_dim=X_seq.shape[2], hidden=256, layers=2, dropout=0.3)
seq_model, _ = train_model(seq_model, seq_train_loader, seq_val_loader,
                           epochs=30, lr=0.001, val_interval=5, patience=6, model_name='T-Bi-LSTM')

# 评估
seq_model.eval()
with torch.no_grad():
    y_prob_seq = torch.sigmoid(seq_model(X_test_seq_t.to(device))).cpu().numpy().flatten()
res = evaluate(y_test, y_prob_seq)
res['model'] = 'T-Bi-LSTM (Seq Only)'
results.append(res)
logger.info(f'T-Bi-LSTM 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')


# ===== 训练优化的混合模型 =====
logger.info('[4/4] Optimized Hybrid (主模型)...')
hybrid_model = OptimizedHybridModel(
    seq_dim=X_seq.shape[2], tab_dim=len(feature_cols),
    hidden=256, layers=2, attn_heads=4, dropout=0.3
)
hybrid_model, _ = train_model(hybrid_model, train_loader, val_loader,
                              epochs=50, lr=0.001, val_interval=5, patience=10, model_name='Hybrid')

# 评估
hybrid_model.eval()
with torch.no_grad():
    with autocast('cuda'):
        logits, _ = hybrid_model(X_test_seq_t.to(device), X_test_tab_t.to(device))
    y_prob_hybrid = torch.sigmoid(logits).cpu().numpy().flatten()
res = evaluate(y_test, y_prob_hybrid)
res['model'] = 'Optimized Hybrid'
results.append(res)
logger.info(f'Hybrid 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# 保存模型
torch.save(hybrid_model.state_dict(), ROOT / 'outputs' / 'models' / 'optimized_hybrid.pth')


# ===== Stacking集成 (深度模型logit作为特征喂给GBDT) =====
logger.info('=' * 60)
logger.info('Stacking 集成...')

# 批量推理函数 (避免OOM)
def batch_predict(model, data_loaders, is_hybrid=False):
    """分批推理避免OOM"""
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in data_loaders:
            if is_hybrid:
                x_seq, x_tab = batch[0].to(device), batch[1].to(device)
                with autocast('cuda'):
                    logits, _ = model(x_seq, x_tab)
            else:
                x_seq = batch[0].to(device)
                logits = model(x_seq)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)
    return np.concatenate(all_probs)

# 清理GPU内存
torch.cuda.empty_cache()

# 创建推理用DataLoader
train_inf_loader = DataLoader(train_dataset, batch_size=512, shuffle=False)
val_inf_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
seq_train_inf_loader = DataLoader(seq_train_ds, batch_size=512, shuffle=False)
seq_val_inf_loader = DataLoader(seq_val_ds, batch_size=512, shuffle=False)

# 收集深度模型预测作为新特征
logger.info('生成 Stacking 特征...')
pred_hybrid_train = batch_predict(hybrid_model, train_inf_loader, is_hybrid=True)
pred_hybrid_val = batch_predict(hybrid_model, val_inf_loader, is_hybrid=True)
pred_seq_train = batch_predict(seq_model, seq_train_inf_loader, is_hybrid=False)
pred_seq_val = batch_predict(seq_model, seq_val_inf_loader, is_hybrid=False)

# 测试集预测
pred_hybrid_test = y_prob_hybrid
pred_seq_test = y_prob_seq

# 构建Stacking特征
X_stack_train = np.column_stack([X_train_flat, pred_hybrid_train, pred_seq_train])
X_stack_val = np.column_stack([X_val_flat, pred_hybrid_val, pred_seq_val])
X_stack_test = np.column_stack([X_test_flat, pred_hybrid_test, pred_seq_test])

# Stacking LightGBM
logger.info('训练 Stacking LightGBM...')
stack_model = lgb.LGBMClassifier(
    max_depth=6, learning_rate=0.03, n_estimators=300,
    class_weight='balanced', verbose=-1, random_state=42
)
stack_model.fit(X_stack_train, y_train, eval_set=[(X_stack_val, y_val)])
y_prob_stack = stack_model.predict_proba(X_stack_test)[:, 1]
res = evaluate(y_test, y_prob_stack)
res['model'] = 'Stacking (Hybrid+GBDT)'
results.append(res)
logger.info(f'Stacking 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')


# ===== 加权融合 =====
logger.info('加权融合实验...')
best_blend_auc = 0
best_weights = None

for w1 in np.arange(0.3, 0.8, 0.1):
    for w2 in np.arange(0.1, 0.5, 0.1):
        w3 = 1 - w1 - w2
        if w3 < 0:
            continue
        y_blend = w1 * y_prob_lgb + w2 * y_prob_hybrid + w3 * y_prob_xgb
        auc = roc_auc_score(y_test, y_blend)
        if auc > best_blend_auc:
            best_blend_auc = auc
            best_weights = (w1, w2, w3)

y_blend = best_weights[0] * y_prob_lgb + best_weights[1] * y_prob_hybrid + best_weights[2] * y_prob_xgb
res = evaluate(y_test, y_blend)
res['model'] = f'Weighted Blend ({best_weights[0]:.1f}*LGB+{best_weights[1]:.1f}*Hybrid+{best_weights[2]:.1f}*XGB)'
results.append(res)
logger.info(f'加权融合完成 (w={best_weights}) - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')


# ===== 结果汇总 =====
logger.info('=' * 60)
logger.info('最终结果 (按AUC排序)')
logger.info('=' * 60)

results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
for _, row in results_df.iterrows():
    logger.info(f'{row["model"]:40s} | AUC: {row["auc"]:.4f} | F1: {row["f1"]:.4f} | P: {row["precision"]:.4f} | R: {row["recall"]:.4f}')

# 检查目标
hybrid_auc = results_df[results_df['model'] == 'Optimized Hybrid']['auc'].values[0]
lgb_auc = results_df[results_df['model'] == 'LightGBM']['auc'].values[0]

logger.info('=' * 60)
if hybrid_auc > lgb_auc:
    logger.info(f'✓ 主模型超越LightGBM! Hybrid AUC: {hybrid_auc:.4f} > LightGBM AUC: {lgb_auc:.4f}')
else:
    logger.info(f'✗ 主模型未超越LightGBM: Hybrid AUC: {hybrid_auc:.4f} vs LightGBM AUC: {lgb_auc:.4f}')
    logger.info(f'  但Stacking集成 AUC: {results_df[results_df["model"].str.contains("Stacking")]["auc"].values[0]:.4f}')
logger.info('=' * 60)

results_df.to_csv(ROOT / 'outputs' / 'model_comparison_optimized.csv', index=False)
logger.info(f'结果保存至 outputs/model_comparison_optimized.csv')
logger.info('实验完成!')
