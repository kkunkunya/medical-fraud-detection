# Input: 特征数据 (final_features.pkl, X_seq.npy, y_seq.npy)
# Output: 训练好的模型，评估指标
# Pos: 模型定义模块，包含所有基线模型和主模型
# Warning: 更新时同步更新注释和 _ARCH.md

"""
模型定义模块
- 基线模型: XGBoost, LightGBM, DNN, LSTM, Transformer
- 主模型: T-Bi-LSTM+Attention (时间衰减门控)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


# ==================== 深度学习基线模型 ====================

class DNN(nn.Module):
    """3层全连接神经网络"""

    def __init__(self, input_dim: int, hidden_dims: list = [256, 128, 64], dropout: float = 0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch, seq_len, features] -> flatten
        if len(x.shape) == 3:
            x = x.view(x.size(0), -1)
        return torch.sigmoid(self.network(x))


class BasicLSTM(nn.Module):
    """基础LSTM模型"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 取最后时刻
        return torch.sigmoid(out)


class TransformerEncoder(nn.Module):
    """Transformer Encoder模型"""

    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])  # 取最后时刻
        return torch.sigmoid(out)


class PositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==================== 主模型: T-Bi-LSTM+Attention ====================

class TimeDecayGate(nn.Module):
    """时间衰减门控 g(Δt) = exp(-λ * Δt)"""

    def __init__(self, input_dim: int, lambda_init: float = 0.1):
        super().__init__()
        # 可学习的衰减参数
        self.lambda_param = nn.Parameter(torch.tensor(lambda_init))
        # 时间间隔嵌入
        self.time_embed = nn.Linear(1, input_dim)

    def forward(self, x: torch.Tensor, delta_t: Optional[torch.Tensor] = None):
        """
        Args:
            x: [batch, seq_len, features]
            delta_t: [batch, seq_len] 时间间隔（天），可选
        Returns:
            加权后的特征 [batch, seq_len, features]
        """
        if delta_t is None:
            # 如果没有提供时间间隔，生成默认的递增序列
            batch_size, seq_len, _ = x.shape
            delta_t = torch.arange(seq_len, 0, -1, dtype=torch.float32, device=x.device)
            delta_t = delta_t.unsqueeze(0).expand(batch_size, -1)

        # 计算时间衰减权重 g(Δt) = exp(-λ * Δt)
        decay_weight = torch.exp(-self.lambda_param.abs() * delta_t)  # [batch, seq_len]
        decay_weight = decay_weight.unsqueeze(-1)  # [batch, seq_len, 1]

        # 时间嵌入
        time_embed = self.time_embed(delta_t.unsqueeze(-1))  # [batch, seq_len, features]

        # 融合：特征 * 衰减权重 + 时间嵌入
        x_weighted = x * decay_weight + time_embed

        return x_weighted, decay_weight.squeeze(-1)


class Attention(nn.Module):
    """注意力机制"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_output: torch.Tensor):
        """
        Args:
            lstm_output: [batch, seq_len, hidden_dim]
        Returns:
            context: [batch, hidden_dim]
            attention_weights: [batch, seq_len]
        """
        # 计算注意力分数
        scores = self.attention(lstm_output).squeeze(-1)  # [batch, seq_len]
        attention_weights = F.softmax(scores, dim=1)  # [batch, seq_len]

        # 加权求和
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_output).squeeze(1)  # [batch, hidden_dim]

        return context, attention_weights


class TBiLSTMAttention(nn.Module):
    """
    T-Bi-LSTM+Attention 主模型
    - Time-Decay Gate: 时间衰减门控
    - Bi-LSTM: 双向LSTM
    - Attention: 注意力机制
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, lambda_init: float = 0.1):
        super().__init__()

        # 特征嵌入
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # 时间衰减门控
        self.time_decay = TimeDecayGate(hidden_dim, lambda_init)

        # Bi-LSTM
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention
        self.attention = Attention(hidden_dim * 2)  # 双向所以是2倍

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor, delta_t: Optional[torch.Tensor] = None,
                return_attention: bool = False):
        """
        Args:
            x: [batch, seq_len, features]
            delta_t: [batch, seq_len] 时间间隔
            return_attention: 是否返回注意力权重
        """
        # 1. 特征嵌入
        x = self.embedding(x)  # [batch, seq_len, hidden_dim]

        # 2. 时间衰减门控
        x, decay_weights = self.time_decay(x, delta_t)

        # 3. Bi-LSTM
        lstm_out, _ = self.bilstm(x)  # [batch, seq_len, hidden_dim*2]

        # 4. Attention
        context, attention_weights = self.attention(lstm_out)  # [batch, hidden_dim*2]

        # 5. 分类
        out = torch.sigmoid(self.classifier(context))

        if return_attention:
            return out, attention_weights, decay_weights
        return out


# ==================== 消融模型变体 ====================

class BiLSTMAttention(nn.Module):
    """消融实验A2: 去除时间衰减的Bi-LSTM+Attention"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        lstm_out, _ = self.bilstm(x)
        context, attention_weights = self.attention(lstm_out)
        out = torch.sigmoid(self.classifier(context))
        if return_attention:
            return out, attention_weights
        return out


class BiLSTMAttentionPosEnc(nn.Module):
    """消融实验A3: 标准位置编码替代时间衰减"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_attention=False):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        lstm_out, _ = self.bilstm(x)
        context, attention_weights = self.attention(lstm_out)
        out = torch.sigmoid(self.classifier(context))
        if return_attention:
            return out, attention_weights
        return out


class BiLSTMAttentionLinearDecay(nn.Module):
    """消融实验A4: 简单线性衰减替代指数衰减"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.bilstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True,
                              dropout=dropout if num_layers > 1 else 0, bidirectional=True)
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.shape
        # 简单线性衰减
        linear_decay = torch.linspace(0.5, 1.0, seq_len, device=x.device)
        linear_decay = linear_decay.view(1, seq_len, 1).expand(batch_size, -1, -1)

        x = self.embedding(x) * linear_decay
        lstm_out, _ = self.bilstm(x)
        context, attention_weights = self.attention(lstm_out)
        out = torch.sigmoid(self.classifier(context))
        if return_attention:
            return out, attention_weights
        return out


# ==================== 混合模型: 全量特征 + 时序特征融合 ====================

class HybridTBiLSTMAttention(nn.Module):
    """
    混合模型: 全量特征 + T-Bi-LSTM+Attention
    - 静态分支: 处理全量特征 (152维)
    - 时序分支: T-Bi-LSTM+Attention处理序列数据
    - 融合层: 两分支特征融合后分类
    """

    def __init__(self, static_dim: int, seq_input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.3, lambda_init: float = 0.1):
        super().__init__()

        # ===== 静态特征分支 =====
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ===== 时序特征分支 (T-Bi-LSTM+Attention) =====
        self.seq_embedding = nn.Linear(seq_input_dim, hidden_dim)
        self.time_decay = TimeDecayGate(hidden_dim, lambda_init)
        self.bilstm = nn.LSTM(
            hidden_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.attention = Attention(hidden_dim * 2)

        # ===== 融合层 =====
        # 静态分支: hidden_dim, 时序分支: hidden_dim * 2
        fusion_dim = hidden_dim + hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 门控融合 (可学习权重)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, 2),
            nn.Softmax(dim=1)
        )

    def forward(self, static_x: torch.Tensor, seq_x: torch.Tensor,
                delta_t: Optional[torch.Tensor] = None, return_attention: bool = False):
        """
        Args:
            static_x: [batch, static_dim] 全量特征
            seq_x: [batch, seq_len, seq_input_dim] 时序特征
            delta_t: [batch, seq_len] 时间间隔
            return_attention: 是否返回注意力权重
        """
        # 1. 静态分支
        static_out = self.static_encoder(static_x)  # [batch, hidden_dim]

        # 2. 时序分支
        seq_emb = self.seq_embedding(seq_x)  # [batch, seq_len, hidden_dim]
        seq_emb, decay_weights = self.time_decay(seq_emb, delta_t)
        lstm_out, _ = self.bilstm(seq_emb)  # [batch, seq_len, hidden_dim*2]
        seq_out, attention_weights = self.attention(lstm_out)  # [batch, hidden_dim*2]

        # 3. 融合
        fused = torch.cat([static_out, seq_out], dim=1)  # [batch, hidden_dim + hidden_dim*2]

        # 门控融合 (自适应权重)
        gate_weights = self.gate(fused)  # [batch, 2]
        static_weighted = static_out * gate_weights[:, 0:1]
        seq_weighted = seq_out * gate_weights[:, 1:2].expand(-1, seq_out.size(1))
        fused_gated = torch.cat([static_weighted, seq_weighted], dim=1)

        # 4. 分类
        out = torch.sigmoid(self.fusion(fused_gated))

        if return_attention:
            return out, attention_weights, decay_weights, gate_weights
        return out


# ==================== 模型工厂 ====================

def get_model(model_name: str, input_dim: int, seq_len: int = 50, **kwargs) -> nn.Module:
    """获取模型实例"""
    models = {
        'dnn': lambda: DNN(input_dim * seq_len, **kwargs),
        'lstm': lambda: BasicLSTM(input_dim, **kwargs),
        'transformer': lambda: TransformerEncoder(input_dim, **kwargs),
        't_bilstm_attention': lambda: TBiLSTMAttention(input_dim, **kwargs),
        'bilstm_attention': lambda: BiLSTMAttention(input_dim, **kwargs),
        'bilstm_attention_posenc': lambda: BiLSTMAttentionPosEnc(input_dim, **kwargs),
        'bilstm_attention_linear': lambda: BiLSTMAttentionLinearDecay(input_dim, **kwargs),
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")

    return models[model_name]()
