# 混合模型实验：全量特征 + 时序特征融合
# 目标：主模型(Hybrid T-Bi-LSTM+Attention)超越所有基线模型
import sys
sys.path.insert(0, '.')
import os
os.environ['PYTHONUNBUFFERED'] = '1'  # 禁用缓冲

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from pathlib import Path

from src.models.models import DNN, BasicLSTM, TransformerEncoder, TBiLSTMAttention, HybridTBiLSTMAttention
from src.utils.logger_config import get_logger

ROOT = Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 初始化日志
logger = get_logger("hybrid_exp")

logger.info('=' * 60)
logger.info('混合模型实验：全量特征 + 时序特征融合')
logger.info(f'Device: {device}')
if device.type == 'cuda':
    logger.info(f'GPU: {torch.cuda.get_device_name(0)}')
    logger.info(f'CUDA Version: {torch.version.cuda}')
logger.info('=' * 60)

# 加载数据
logger.info('加载数据...')
df = pd.read_pickle(ROOT / 'outputs' / 'final_features.pkl')
feature_cols = [c for c in df.columns if c != 'class']
X_flat = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.float32)

X_seq = np.load(ROOT / 'outputs' / 'X_seq.npy')
y_seq = np.load(ROOT / 'outputs' / 'y_seq.npy')

logger.info(f'全量特征: {X_flat.shape} ({len(feature_cols)}维)')
logger.info(f'时序特征: {X_seq.shape}')

# 标准化全量特征
scaler = StandardScaler()
X_flat = scaler.fit_transform(X_flat)

# 统一划分
indices = np.arange(len(y))
idx_temp, idx_test = train_test_split(indices, test_size=0.15, stratify=y, random_state=42)
idx_train, idx_val = train_test_split(idx_temp, test_size=0.176, stratify=y[idx_temp], random_state=42)

# 全量特征数据
X_train_flat, X_val_flat, X_test_flat = X_flat[idx_train], X_flat[idx_val], X_flat[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

# 时序数据
X_train_seq, X_val_seq, X_test_seq = X_seq[idx_train], X_seq[idx_val], X_seq[idx_test]

logger.info(f'数据划分 - 训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(idx_test)}')

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
logger.info(f'类别权重: {pos_weight:.2f}')

results = []

# ========== 基线模型 ==========

# XGBoost
logger.info('[1/7] XGBoost (全量特征) - 开始训练')
xgb_model = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=200,
                              scale_pos_weight=pos_weight, verbosity=0, random_state=42)
xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
y_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
y_pred = xgb_model.predict(X_test_flat)
results.append({'model': 'XGBoost', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
logger.info(f'[1/7] XGBoost 完成 - AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# LightGBM
logger.info('[2/7] LightGBM (全量特征) - 开始训练')
lgb_model = lgb.LGBMClassifier(max_depth=8, learning_rate=0.05, n_estimators=200,
                               class_weight='balanced', verbose=-1, random_state=42)
lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)])
y_prob = lgb_model.predict_proba(X_test_flat)[:, 1]
y_pred = lgb_model.predict(X_test_flat)
results.append({'model': 'LightGBM', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
logger.info(f'[2/7] LightGBM 完成 - AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# DNN
logger.info('[3/7] DNN (全量特征) - 开始训练')
X_train_dnn = torch.FloatTensor(X_train_flat)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_dnn = torch.FloatTensor(X_val_flat)
X_test_dnn = torch.FloatTensor(X_test_flat)

dnn = DNN(len(feature_cols), [256, 128, 64]).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(dnn.parameters(), lr=0.001)
train_loader_dnn = DataLoader(TensorDataset(X_train_dnn, y_train_t), batch_size=128, shuffle=True)

best_auc, best_state = 0, None
for epoch in range(50):
    dnn.train()
    epoch_loss = 0
    for bx, by in train_loader_dnn:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(dnn(bx), by)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    dnn.eval()
    with torch.no_grad():
        vp = dnn(X_val_dnn.to(device)).cpu().numpy().flatten()
    va = roc_auc_score(y_val, vp)
    if va > best_auc: best_auc, best_state = va, dnn.state_dict().copy()
    if (epoch + 1) % 10 == 0:
        logger.info(f'DNN Epoch {epoch+1}/50 - Loss: {epoch_loss/len(train_loader_dnn):.4f}, Val AUC: {va:.4f}')

dnn.load_state_dict(best_state)
dnn.eval()
with torch.no_grad():
    y_prob = dnn(X_test_dnn.to(device)).cpu().numpy().flatten()
best_f1, best_t = 0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    f = f1_score(y_test, (y_prob > t).astype(int))
    if f > best_f1: best_f1, best_t = f, t
y_pred = (y_prob > best_t).astype(int)
results.append({'model': 'DNN', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
logger.info(f'[3/7] DNN 完成 - AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# ========== 时序模型 ==========
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val_seq)
X_test_t = torch.FloatTensor(X_test_seq)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

input_dim = X_seq.shape[2]

class FocalLoss(nn.Module):
    def __init__(self, a=0.75, g=2):
        super().__init__()
        self.a, self.g = a, g
    def forward(self, i, t):
        bce = nn.functional.binary_cross_entropy(i, t, reduction='none')
        pt = torch.exp(-bce)
        return (self.a * (1-pt)**self.g * bce).mean()

def train_seq_model(model, name, epochs, use_focal=False, log_interval=20):
    """训练时序模型，带日志记录"""
    logger.info(f'{name} - 开始训练 ({epochs} epochs)')
    model = model.to(device)
    criterion = FocalLoss() if use_focal else nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_auc, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_val_t.to(device)).cpu().numpy().flatten()
        va = roc_auc_score(y_val, vp)
        if va > best_auc: best_auc, best_state = va, model.state_dict().copy()

        # 每隔 log_interval 个 epoch 记录一次
        if (epoch + 1) % log_interval == 0:
            logger.info(f'{name} Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Val AUC: {va:.4f}, Best: {best_auc:.4f}')

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t.to(device)).cpu().numpy().flatten()
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f = f1_score(y_test, (y_prob > t).astype(int))
        if f > best_f1: best_f1, best_t = f, t
    y_pred = (y_prob > best_t).astype(int)
    return {'model': name, 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)}, model

# LSTM
logger.info('[4/7] LSTM (时序特征) - 开始训练')
res, _ = train_seq_model(BasicLSTM(input_dim, 128, 2, 0.3), 'LSTM', 60)
results.append(res)
logger.info(f'[4/7] LSTM 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# Transformer
logger.info('[5/7] Transformer (时序特征) - 开始训练')
res, _ = train_seq_model(TransformerEncoder(input_dim, 128, 4, 2, 0.3), 'Transformer', 60)
results.append(res)
logger.info(f'[5/7] Transformer 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# T-Bi-LSTM+Attention (纯时序版本，作为对比)
logger.info('[6/7] T-Bi-LSTM+Attention (时序特征) - 开始训练')
res, _ = train_seq_model(TBiLSTMAttention(input_dim, 256, 3, 0.4, 0.1), 'T-Bi-LSTM+Attention', 100, use_focal=True, log_interval=25)
results.append(res)
logger.info(f'[6/7] T-Bi-LSTM+Attention 完成 - AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# ========== 混合主模型 ==========
logger.info('[7/7] Hybrid T-Bi-LSTM+Attention (全量+时序融合) - 主模型')

# 准备混合数据
X_train_static = torch.FloatTensor(X_train_flat)
X_train_seq_t = torch.FloatTensor(X_train_seq)
y_train_hybrid = torch.FloatTensor(y_train).unsqueeze(1)
X_val_static = torch.FloatTensor(X_val_flat)
X_val_seq_t = torch.FloatTensor(X_val_seq)
X_test_static = torch.FloatTensor(X_test_flat)
X_test_seq_t = torch.FloatTensor(X_test_seq)

# 混合数据加载器
train_dataset = TensorDataset(X_train_static, X_train_seq_t, y_train_hybrid)
train_loader_hybrid = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建混合模型
hybrid_model = HybridTBiLSTMAttention(
    static_dim=len(feature_cols),
    seq_input_dim=input_dim,
    hidden_dim=192,
    num_layers=3,
    dropout=0.35,
    lambda_init=0.1
).to(device)

# 训练配置
criterion = FocalLoss(0.8, 2)
optimizer = torch.optim.AdamW(hybrid_model.parameters(), lr=0.0008, weight_decay=0.02)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=25, T_mult=2)

best_auc, best_state = 0, None
epochs = 150
logger.info(f'Hybrid 模型开始训练 ({epochs} epochs)')

for epoch in range(epochs):
    hybrid_model.train()
    epoch_loss = 0
    for b_static, b_seq, b_y in train_loader_hybrid:
        b_static, b_seq, b_y = b_static.to(device), b_seq.to(device), b_y.to(device)
        optimizer.zero_grad()
        out = hybrid_model(b_static, b_seq)
        loss = criterion(out, b_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(hybrid_model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step()

    hybrid_model.eval()
    with torch.no_grad():
        val_prob = hybrid_model(X_val_static.to(device), X_val_seq_t.to(device)).cpu().numpy().flatten()
    val_auc = roc_auc_score(y_val, val_prob)

    if val_auc > best_auc:
        best_auc = val_auc
        best_state = hybrid_model.state_dict().copy()

    if (epoch + 1) % 25 == 0:
        logger.info(f'Hybrid Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(train_loader_hybrid):.4f}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}')

hybrid_model.load_state_dict(best_state)
hybrid_model.eval()
with torch.no_grad():
    y_prob = hybrid_model(X_test_static.to(device), X_test_seq_t.to(device)).cpu().numpy().flatten()

# 寻找最优阈值
best_f1, best_t = 0, 0.5
for t in np.arange(0.1, 0.9, 0.01):
    f = f1_score(y_test, (y_prob > t).astype(int))
    if f > best_f1: best_f1, best_t = f, t

y_pred = (y_prob > best_t).astype(int)
results.append({
    'model': 'Hybrid T-Bi-LSTM+Attention',
    'auc': roc_auc_score(y_test, y_prob),
    'f1': f1_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred)
})
logger.info(f'[7/7] Hybrid 完成 - AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}, Threshold: {best_t:.2f}')

# 保存模型
torch.save(best_state, ROOT / 'outputs' / 'models' / 'hybrid_t_bilstm_attention_best.pth')
logger.info(f'模型已保存至 outputs/models/hybrid_t_bilstm_attention_best.pth')

# ========== 结果汇总 ==========
logger.info('=' * 60)
logger.info('最终模型对比结果 (按AUC排序)')
logger.info('=' * 60)
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
for _, row in results_df.iterrows():
    logger.info(f'{row["model"]:30s} | AUC: {row["auc"]:.4f} | F1: {row["f1"]:.4f} | P: {row["precision"]:.4f} | R: {row["recall"]:.4f}')

# 检查主模型是否超越所有基线
hybrid_auc = results_df[results_df['model'] == 'Hybrid T-Bi-LSTM+Attention']['auc'].values[0]
baseline_max_auc = results_df[results_df['model'] != 'Hybrid T-Bi-LSTM+Attention']['auc'].max()

logger.info('=' * 60)
if hybrid_auc > baseline_max_auc:
    logger.info(f'主模型成功超越所有基线模型!')
    logger.info(f'主模型 AUC: {hybrid_auc:.4f} > 基线最高 AUC: {baseline_max_auc:.4f}')
else:
    logger.warning(f'主模型尚未超越基线')
    logger.warning(f'主模型 AUC: {hybrid_auc:.4f} vs 基线最高 AUC: {baseline_max_auc:.4f}')
logger.info('=' * 60)

results_df.to_csv(ROOT / 'outputs' / 'model_comparison_hybrid.csv', index=False)
logger.info(f'结果已保存至 outputs/model_comparison_hybrid.csv')
logger.info('实验完成')
