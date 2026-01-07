# Input: 预处理后的数据, 特征工程结果
# Output: 时序窗口数据 (X_seq, y), LSTM梯度重要性
# Pos: 时序数据准备模块，为LSTM模型提供序列输入
# Warning: 更新时同步更新注释和 _ARCH.md

"""
时序窗口构建与特征筛选模块
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class SequenceBuilder:
    """构建时序窗口数据"""

    def __init__(self, df: pd.DataFrame, seq_len: int = 50):
        """
        Args:
            df: 预处理后的就诊记录数据
            seq_len: 序列长度
        """
        self.df = df.copy()
        self.seq_len = seq_len
        self.id_col = '个人编码'
        self.time_col = '交易时间'
        self.label_col = 'class'
        self.scaler = StandardScaler()

    def prepare_features(self) -> List[str]:
        """准备用于序列的特征列"""
        # 选择数值型费用特征
        feature_cols = []
        for col in self.df.columns:
            if col in [self.id_col, self.time_col, self.label_col, '顺序号', '医院编码']:
                continue
            if self.df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                feature_cols.append(col)

        return feature_cols[:30]  # 限制特征数量

    def build_sequences(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """构建用户级别的时序窗口"""
        print('构建时序窗口...')

        feature_cols = self.prepare_features()
        print(f'  使用特征数: {len(feature_cols)}')

        # 确保时间排序
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors='coerce')
        self.df = self.df.sort_values([self.id_col, self.time_col])

        # 获取用户列表和标签
        user_labels = self.df.groupby(self.id_col)[self.label_col].first()
        users = user_labels.index.tolist()

        # 标准化特征
        self.df[feature_cols] = self.df[feature_cols].fillna(0)
        self.df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])

        # 构建序列
        sequences = []
        labels = []

        for user_id in users:
            user_data = self.df[self.df[self.id_col] == user_id][feature_cols].values

            if len(user_data) >= self.seq_len:
                # 取最近的 seq_len 条记录
                seq = user_data[-self.seq_len:]
            else:
                # 零填充
                padding = np.zeros((self.seq_len - len(user_data), len(feature_cols)))
                seq = np.vstack([padding, user_data])

            sequences.append(seq)
            labels.append(user_labels[user_id])

        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)

        print(f'  序列形状: {X.shape}')
        print(f'  标签形状: {y.shape}')
        print(f'  正样本数: {int(y.sum())}')

        return X, y, feature_cols


class SimpleLSTM(nn.Module):
    """简单LSTM用于梯度分析"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(out)


class LSTMGradientAnalyzer:
    """LSTM梯度分析用于特征筛选"""

    def __init__(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
        self.feature_names = feature_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_gradient_importance(self, epochs: int = 10) -> Dict[str, float]:
        """计算特征梯度重要性"""
        print(f'\n计算LSTM梯度重要性 (device: {self.device})...')

        # 划分数据
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, stratify=self.y, random_state=42
        )

        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        # 模型
        model = SimpleLSTM(self.X.shape[2]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # 训练
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')

        # 计算梯度重要性
        model.train()  # 保持训练模式以支持LSTM梯度计算
        X_sample = self.X[:500].to(self.device)
        X_sample.requires_grad = True

        outputs = model(X_sample)
        loss = outputs.sum()
        loss.backward()

        # 梯度绝对值的平均
        gradients = X_sample.grad.abs().mean(dim=(0, 1)).cpu().numpy()

        importance = dict(zip(self.feature_names, gradients))
        importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

        print(f'  Top 10 重要特征:')
        for i, (name, score) in enumerate(list(importance.items())[:10]):
            print(f'    {i+1}. {name}: {score:.4f}')

        return importance


def build_final_dataset(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                        importance: Dict[str, float], top_k: int = 20) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """基于重要性筛选特征，构建最终数据集"""
    print(f'\n筛选Top {top_k}特征...')

    # 选择top_k个特征
    top_features = list(importance.keys())[:top_k]
    top_indices = [feature_names.index(f) for f in top_features if f in feature_names]

    X_filtered = X[:, :, top_indices]
    print(f'  筛选后形状: {X_filtered.shape}')

    return X_filtered, y, top_features


if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent.parent
    df = pd.read_pickle(ROOT / 'outputs' / 'preprocessed_data.pkl')

    # 构建序列
    builder = SequenceBuilder(df, seq_len=50)
    X, y, feature_names = builder.build_sequences()

    # LSTM梯度分析
    analyzer = LSTMGradientAnalyzer(X, y, feature_names)
    importance = analyzer.compute_gradient_importance(epochs=10)

    # 筛选特征
    X_final, y_final, final_features = build_final_dataset(X, y, feature_names, importance, top_k=20)

    # 保存
    np.save(ROOT / 'outputs' / 'X_seq.npy', X_final)
    np.save(ROOT / 'outputs' / 'y_seq.npy', y_final)

    import json
    with open(ROOT / 'outputs' / 'seq_features.json', 'w') as f:
        json.dump({'features': final_features, 'importance': importance}, f, indent=2)

    print(f'\n数据已保存:')
    print(f'  X_seq.npy: {X_final.shape}')
    print(f'  y_seq.npy: {y_final.shape}')
