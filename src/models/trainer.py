# Input: 模型定义, 特征数据
# Output: 训练好的模型, 评估结果
# Pos: 模型训练模块，统一训练和评估接口
# Warning: 更新时同步更新注释和 _ARCH.md

"""
模型训练与评估模块
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .models import get_model


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced data"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class ModelTrainer:
    """统一的模型训练器"""

    def __init__(self, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        print(f'Using device: {self.device}')

    def prepare_data(self, X: np.ndarray, y: np.ndarray,
                     test_size: float = 0.15, val_size: float = 0.15) -> Dict:
        """划分数据集"""
        # 先划分训练+验证 和 测试
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )

        # 再划分训练和验证
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
        )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

    def train_xgboost(self, data: Dict, class_weight: Optional[float] = None) -> Tuple:
        """训练XGBoost"""
        X_train = data['X_train'].reshape(len(data['X_train']), -1)
        X_val = data['X_val'].reshape(len(data['X_val']), -1)
        X_test = data['X_test'].reshape(len(data['X_test']), -1)

        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'use_label_encoder': False,
            'verbosity': 0
        }

        if class_weight:
            params['scale_pos_weight'] = class_weight

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, data['y_train'],
                  eval_set=[(X_val, data['y_val'])],
                  verbose=False)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return model, y_pred, y_prob

    def train_lightgbm(self, data: Dict, class_weight: Optional[str] = None) -> Tuple:
        """训练LightGBM"""
        X_train = data['X_train'].reshape(len(data['X_train']), -1)
        X_val = data['X_val'].reshape(len(data['X_val']), -1)
        X_test = data['X_test'].reshape(len(data['X_test']), -1)

        params = {
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'verbose': -1
        }

        if class_weight:
            params['class_weight'] = class_weight

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, data['y_train'],
                  eval_set=[(X_val, data['y_val'])])

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        return model, y_pred, y_prob

    def train_deep_model(self, model: nn.Module, data: Dict,
                         epochs: int = 50, batch_size: int = 64,
                         lr: float = 0.001, loss_type: str = 'bce',
                         class_weight: Optional[float] = None) -> Tuple:
        """训练深度学习模型"""
        model = model.to(self.device)

        # 准备数据加载器
        X_train = torch.FloatTensor(data['X_train'])
        y_train = torch.FloatTensor(data['y_train']).unsqueeze(1)
        X_val = torch.FloatTensor(data['X_val'])
        y_val = torch.FloatTensor(data['y_val']).unsqueeze(1)
        X_test = torch.FloatTensor(data['X_test'])
        y_test = torch.FloatTensor(data['y_test']).unsqueeze(1)

        train_loader = DataLoader(TensorDataset(X_train, y_train),
                                  batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val),
                                batch_size=batch_size)

        # 损失函数
        if loss_type == 'focal':
            criterion = FocalLoss(gamma=2.0)
        elif loss_type == 'weighted_bce' and class_weight:
            pos_weight = torch.tensor([class_weight]).to(self.device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_auc = 0
        best_model_state = None

        for epoch in range(epochs):
            # 训练
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                if loss_type == 'weighted_bce':
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # 验证
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(self.device)
                    outputs = model(batch_X)
                    val_preds.extend(outputs.cpu().numpy())

            val_auc = roc_auc_score(data['y_val'], val_preds)
            scheduler.step(val_auc)

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}')

        # 加载最佳模型
        model.load_state_dict(best_model_state)

        # 测试
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(self.device)
            y_prob = model(X_test_device).cpu().numpy().flatten()
            y_pred = (y_prob > 0.5).astype(int)

        return model, y_pred, y_prob

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
        """计算评估指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_prob)
        }


def run_all_models(X: np.ndarray, y: np.ndarray, output_dir: Path) -> pd.DataFrame:
    """运行所有模型并返回对比结果"""
    trainer = ModelTrainer()
    data = trainer.prepare_data(X, y)

    # 计算类别权重
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    class_weight = neg_count / pos_count

    print(f'\n数据划分: 训练={len(data["X_train"])}, 验证={len(data["X_val"])}, 测试={len(data["X_test"])}')
    print(f'类别权重: {class_weight:.2f}')

    results = []
    input_dim = X.shape[2]
    seq_len = X.shape[1]

    # 1. XGBoost
    print('\n[1/6] 训练 XGBoost...')
    _, y_pred, y_prob = trainer.train_xgboost(data)
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'XGBoost'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 2. LightGBM
    print('\n[2/6] 训练 LightGBM...')
    _, y_pred, y_prob = trainer.train_lightgbm(data)
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'LightGBM'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 3. DNN
    print('\n[3/6] 训练 DNN...')
    model = get_model('dnn', input_dim, seq_len)
    _, y_pred, y_prob = trainer.train_deep_model(model, data, epochs=30)
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'DNN'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 4. LSTM
    print('\n[4/6] 训练 LSTM...')
    model = get_model('lstm', input_dim)
    _, y_pred, y_prob = trainer.train_deep_model(model, data, epochs=30)
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'LSTM'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 5. Transformer
    print('\n[5/6] 训练 Transformer...')
    model = get_model('transformer', input_dim)
    _, y_pred, y_prob = trainer.train_deep_model(model, data, epochs=30)
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'Transformer'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 6. T-Bi-LSTM+Attention (主模型)
    print('\n[6/6] 训练 T-Bi-LSTM+Attention (主模型)...')
    model = get_model('t_bilstm_attention', input_dim)
    trained_model, y_pred, y_prob = trainer.train_deep_model(
        model, data, epochs=50, loss_type='focal'
    )
    metrics = trainer.evaluate(data['y_test'], y_pred, y_prob)
    metrics['model'] = 'T-Bi-LSTM+Attention'
    results.append(metrics)
    print(f'  AUC: {metrics["auc"]:.4f}, F1: {metrics["f1"]:.4f}')

    # 保存主模型
    torch.save(trained_model.state_dict(), output_dir / 'models' / 't_bilstm_attention.pth')

    # 转为DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']]
    results_df = results_df.sort_values('auc', ascending=False).reset_index(drop=True)

    return results_df, data, trained_model


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent

    X = np.load(ROOT / 'outputs' / 'X_seq.npy')
    y = np.load(ROOT / 'outputs' / 'y_seq.npy')

    print(f'数据形状: X={X.shape}, y={y.shape}')

    results_df, data, model = run_all_models(X, y, ROOT / 'outputs')
    print('\n' + '='*60)
    print('模型对比结果:')
    print(results_df.to_string(index=False))
