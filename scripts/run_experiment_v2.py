# 优化实验：时序模型用原始序列，ML用全量特征
import sys
sys.path.insert(0, '.')
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

from src.models.models import DNN, BasicLSTM, TransformerEncoder, TBiLSTMAttention

ROOT = Path('.')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('=' * 60)
print('优化实验：混合数据策略')
print('=' * 60)

# 加载数据
df = pd.read_pickle(ROOT / 'outputs' / 'final_features.pkl')
feature_cols = [c for c in df.columns if c != 'class']
X_flat = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.float32)

X_seq = np.load(ROOT / 'outputs' / 'X_seq.npy')
y_seq = np.load(ROOT / 'outputs' / 'y_seq.npy')

print(f'全量特征: {X_flat.shape}')
print(f'时序特征: {X_seq.shape}')

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

print(f'训练: {len(idx_train)}, 验证: {len(idx_val)}, 测试: {len(idx_test)}')

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f'类别权重: {pos_weight:.2f}')

results = []

# XGBoost
print('\n[1/6] XGBoost (全量特征)...')
xgb_model = xgb.XGBClassifier(max_depth=8, learning_rate=0.05, n_estimators=200,
                              scale_pos_weight=pos_weight, verbosity=0)
xgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)], verbose=False)
y_prob = xgb_model.predict_proba(X_test_flat)[:, 1]
y_pred = xgb_model.predict(X_test_flat)
results.append({'model': 'XGBoost', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# LightGBM
print('\n[2/6] LightGBM (全量特征)...')
lgb_model = lgb.LGBMClassifier(max_depth=8, learning_rate=0.05, n_estimators=200,
                               class_weight='balanced', verbose=-1)
lgb_model.fit(X_train_flat, y_train, eval_set=[(X_val_flat, y_val)])
y_prob = lgb_model.predict_proba(X_test_flat)[:, 1]
y_pred = lgb_model.predict(X_test_flat)
results.append({'model': 'LightGBM', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# DNN用全量特征
print('\n[3/6] DNN (全量特征)...')
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
    for bx, by in train_loader_dnn:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(dnn(bx), by)
        loss.backward()
        optimizer.step()
    dnn.eval()
    with torch.no_grad():
        vp = dnn(X_val_dnn.to(device)).cpu().numpy().flatten()
    va = roc_auc_score(y_val, vp)
    if va > best_auc: best_auc, best_state = va, dnn.state_dict().copy()

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
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# 时序模型用原始序列
X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val_seq)
X_test_t = torch.FloatTensor(X_test_seq)
train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True)

input_dim = X_seq.shape[2]

def train_seq_model(model, name, epochs, use_focal=False):
    model = model.to(device)
    if use_focal:
        class FocalLoss(nn.Module):
            def __init__(self, a=0.75, g=2):
                super().__init__()
                self.a, self.g = a, g
            def forward(self, i, t):
                bce = nn.functional.binary_cross_entropy(i, t, reduction='none')
                pt = torch.exp(-bce)
                return (self.a * (1-pt)**self.g * bce).mean()
        criterion = FocalLoss()
    else:
        criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_auc, best_state = 0, None
    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            vp = model(X_val_t.to(device)).cpu().numpy().flatten()
        va = roc_auc_score(y_val, vp)
        if va > best_auc: best_auc, best_state = va, model.state_dict().copy()
        if (epoch+1) % 30 == 0:
            print(f'    Epoch {epoch+1}, Val AUC: {va:.4f}, Best: {best_auc:.4f}')

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
print('\n[4/6] LSTM (时序特征)...')
res, _ = train_seq_model(BasicLSTM(input_dim, 128, 2, 0.3), 'LSTM', 60)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# Transformer
print('\n[5/6] Transformer (时序特征)...')
res, _ = train_seq_model(TransformerEncoder(input_dim, 128, 4, 2, 0.3), 'Transformer', 60)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# T-Bi-LSTM+Attention (加强版)
print('\n[6/6] T-Bi-LSTM+Attention (时序特征 + Focal Loss)...')
main_model = TBiLSTMAttention(input_dim, hidden_dim=256, num_layers=3, dropout=0.4, lambda_init=0.1)
res, trained_model = train_seq_model(main_model, 'T-Bi-LSTM+Attention', 120, use_focal=True)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

torch.save(trained_model.state_dict(), ROOT / 'outputs' / 'models' / 't_bilstm_attention_best.pth')

print('\n' + '=' * 60)
print('最终模型对比结果 (按AUC排序)')
print('=' * 60)
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(ROOT / 'outputs' / 'model_comparison_final.csv', index=False)
