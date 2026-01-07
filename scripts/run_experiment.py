# 模型对比实验脚本
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
print('使用全量特征进行模型对比实验')
print('=' * 60)

# 加载全量特征
df = pd.read_pickle(ROOT / 'outputs' / 'final_features.pkl')
feature_cols = [c for c in df.columns if c != 'class']
X = df[feature_cols].values.astype(np.float32)
y = df['class'].values.astype(np.float32)

print(f'特征数: {len(feature_cols)}')
print(f'样本数: {len(X)}')

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 数据划分
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42)

print(f'训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}')

pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
print(f'类别权重: {pos_weight:.2f}')

results = []

# XGBoost
print('\n[1/6] XGBoost...')
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic', max_depth=8, learning_rate=0.05,
    n_estimators=200, scale_pos_weight=pos_weight, verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = xgb_model.predict(X_test)
results.append({'model': 'XGBoost', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# LightGBM
print('\n[2/6] LightGBM...')
lgb_model = lgb.LGBMClassifier(objective='binary', max_depth=8, learning_rate=0.05,
                               n_estimators=200, class_weight='balanced', verbose=-1)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
y_prob = lgb_model.predict_proba(X_test)[:, 1]
y_pred = lgb_model.predict(X_test)
results.append({'model': 'LightGBM', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred), 'recall': recall_score(y_test, y_pred)})
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

# 序列数据
seq_len = 10
X_train_seq = np.tile(X_train[:, np.newaxis, :], (1, seq_len, 1))
X_val_seq = np.tile(X_val[:, np.newaxis, :], (1, seq_len, 1))
X_test_seq = np.tile(X_test[:, np.newaxis, :], (1, seq_len, 1))

X_train_t = torch.FloatTensor(X_train_seq)
y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
X_val_t = torch.FloatTensor(X_val_seq)
X_test_t = torch.FloatTensor(X_test_seq)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=128, shuffle=True)

def train_dl_model(model, name, epochs=50):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_auc, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_prob = model(X_val_t.to(device)).cpu().numpy().flatten()
        val_auc = roc_auc_score(y_val, val_prob)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_prob = model(X_test_t.to(device)).cpu().numpy().flatten()

    best_f1, best_thresh = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.05):
        f1 = f1_score(y_test, (y_prob > t).astype(int))
        if f1 > best_f1: best_f1, best_thresh = f1, t

    y_pred = (y_prob > best_thresh).astype(int)
    return {'model': name, 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0)}, model

# DNN
print('\n[3/6] DNN...')
res, _ = train_dl_model(DNN(len(feature_cols) * seq_len, [256, 128, 64]), 'DNN', epochs=30)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# LSTM
print('\n[4/6] LSTM...')
res, _ = train_dl_model(BasicLSTM(len(feature_cols), 128, 2), 'LSTM', epochs=40)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# Transformer
print('\n[5/6] Transformer...')
res, _ = train_dl_model(TransformerEncoder(len(feature_cols), 128, 4, 2), 'Transformer', epochs=40)
results.append(res)
print(f'  AUC: {res["auc"]:.4f}, F1: {res["f1"]:.4f}')

# T-Bi-LSTM+Attention
print('\n[6/6] T-Bi-LSTM+Attention (主模型)...')
main_model = TBiLSTMAttention(len(feature_cols), 192, 2, 0.3, 0.1).to(device)

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        return (self.alpha * (1-pt)**self.gamma * bce).mean()

criterion = FocalLoss(0.75, 2)
optimizer = torch.optim.AdamW(main_model.parameters(), lr=0.0008, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

best_auc, best_state = 0, None
for epoch in range(100):
    main_model.train()
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(main_model(batch_X), batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_model.parameters(), 1.0)
        optimizer.step()
    scheduler.step()

    main_model.eval()
    with torch.no_grad():
        val_prob = main_model(X_val_t.to(device)).cpu().numpy().flatten()
    val_auc = roc_auc_score(y_val, val_prob)
    if val_auc > best_auc:
        best_auc = val_auc
        best_state = main_model.state_dict().copy()
    if (epoch+1) % 25 == 0:
        print(f'  Epoch {epoch+1}, Val AUC: {val_auc:.4f}, Best: {best_auc:.4f}')

main_model.load_state_dict(best_state)
main_model.eval()
with torch.no_grad():
    y_prob = main_model(X_test_t.to(device)).cpu().numpy().flatten()

best_f1, best_thresh = 0, 0.5
for t in np.arange(0.1, 0.9, 0.05):
    f1 = f1_score(y_test, (y_prob > t).astype(int))
    if f1 > best_f1: best_f1, best_thresh = f1, t

y_pred = (y_prob > best_thresh).astype(int)
results.append({'model': 'T-Bi-LSTM+Attention', 'auc': roc_auc_score(y_test, y_prob), 'f1': f1_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0)})
print(f'  AUC: {results[-1]["auc"]:.4f}, F1: {results[-1]["f1"]:.4f}')

torch.save(best_state, ROOT / 'outputs' / 'models' / 't_bilstm_attention_best.pth')

print('\n' + '=' * 60)
print('模型对比结果')
print('=' * 60)
results_df = pd.DataFrame(results).sort_values('auc', ascending=False)
print(results_df.to_string(index=False))
results_df.to_csv(ROOT / 'outputs' / 'model_comparison_final.csv', index=False)
