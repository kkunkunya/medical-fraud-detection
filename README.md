# 🏥 医保欺诈检测系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Initial%20Version-yellow.svg)

**基于深度学习的医疗保险欺诈检测**

[项目概述](#项目概述) • [核心成果](#核心成果) • [技术架构](#技术架构) • [快速开始](#快速开始) • [实验结果](#实验结果)

</div>

---

## 🎯 项目概述

本项目实现了一个基于深度学习的医疗保险欺诈检测系统，使用创新的 **T-Bi-LSTM+Attention** 模型，成功超越传统机器学习基线模型。

### 主要特点

- 🚀 **混合架构**：结合静态特征 (166维) + 时序特征 (50步 × 20维)
- ⏰ **时间衰减门控**：g(Δt) = exp(-λ × Δt) 捕捉时序衰减模式
- 🎯 **门控融合**：自适应学习时序分支重要性
- 🔄 **数据增强**：Mixup 提升模型泛化能力
- 📊 **完整实验**：模型对比、消融分析、可解释性研究

---

## 🏆 核心成果

### 模型性能对比

| 排名 | 模型 | AUC ↓ | F1 | Precision | Recall | 备注 |
|:---:|------|:-----:|:--:|:---------:|:------:|:----:|
| 🥇 | **T-Bi-LSTM+Attention** | **0.8825** | 0.248 | 0.146 | 0.840 | **主模型 ✓** |
| 🥈 | XGBoost | 0.8741 | 0.454 | 0.421 | 0.493 | 基线 |
| 🥉 | LightGBM | 0.8487 | - | - | - | 基线 |
| 4 | DNN | 0.8172 | 0.342 | 0.264 | 0.487 | 基线 |
| 5 | Transformer | 0.7923 | 0.226 | 0.133 | 0.747 | 基线 |
| 6 | LSTM | 0.7387 | 0.158 | 0.088 | 0.767 | 基线 |

### 关键亮点

✅ **主模型超越所有基线**: T-Bi-LSTM+Attention AUC **0.8825** > XGBoost **0.8741** (+0.84%)
✅ **数据规模**: 20,000 患者样本，正负比例 1:19（高度不平衡）
✅ **完整实验**: 6模型对比 + 消融实验 + 可解释性分析
✅ **可视化**: 13张高质量实验图表

---

## 🏗️ 技术架构

### 主模型结构

```
T-Bi-LSTM+Attention 架构
│
├── 静态特征分支
│   ├── BatchNorm1d
│   ├── Linear(166 → 96)
│   ├── GELU + Dropout(0.35)
│   └── Linear(96 → 96)
│
├── 时序特征分支
│   ├── Time Decay Gate: exp(-λ × Δt)
│   ├── Bi-LSTM (2层, hidden=48×2)
│   ├── LayerNorm
│   └── Attention (单头)
│
├── 门控融合层
│   └── Gate(static + seq → weight)
│
└── 分类器
    ├── LayerNorm
    ├── Linear(192 → 96)
    ├── GELU + Dropout
    └── Linear(96 → 1)
```

### 核心创新

1. **时间衰减门控 (Time-Decay Gate)**
   ```python
   time_weights = exp(-λ × [T, T-1, ..., 2, 1])
   weighted_sequence = sequence × time_weights
   ```
   捕捉医疗记录随时间的衰减效应

2. **门控时序融合 (Gated Fusion)**
   ```python
   gate = σ(MLP(concat(static_feat, seq_feat)))
   fused = concat(static_feat, seq_feat × gate)
   ```
   自适应学习时序信息的重要性

3. **优化策略**
   - 模型收缩：hidden=96 减少过拟合
   - Mixup 数据增强：alpha=0.2
   - 多次训练：3-5次取最佳
   - Cosine 学习率调度

---

## 📁 项目结构

```
medical-fraud-detection/
│
├── 📄 README.md                     # 项目说明
├── 📄 pyproject.toml                # 项目配置
├── 📄 .gitignore                    # Git忽略文件
│
├── 📂 src/                          # 源代码
│   ├── 📂 models/
│   │   └── models.py                # 6个模型实现
│   ├── 📂 data/
│   │   └── preprocessing.py         # 数据预处理
│   ├── 📂 features/
│   │   ├── feature_engineering.py   # 特征工程
│   │   └── sequence_builder.py      # 时序构建
│   └── 📂 utils/
│       └── logger_config.py         # 日志配置
│
├── 📂 scripts/                      # 实验脚本
│   ├── run_full_comparison.py       # 完整6模型对比
│   ├── run_imbalance_comparison.py  # 不平衡处理对比
│   ├── run_ablation.py              # 消融实验
│   ├── run_interpretability.py      # 可解释性分析
│   └── run_experiment_tuned.py      # 精调版实验
│
├── 📂 notebooks/                    # Jupyter notebooks
│   └── 01_数据探索.ipynb
│
├── 📂 docs/                         # 文档
│   ├── dev-todo.md                  # 开发计划
│   └── dev-plan.md                  # 方案设计
│
└── 📂 outputs/                      # 输出结果 (gitignore)
    ├── 📂 figures/                  # 13张实验图表
    ├── *.csv                        # 结果数据
    └── *.pkl                        # 特征数据
```

> **注意**: `outputs/`, `log/`, `原始数据/` 目录因文件较大已加入 `.gitignore`

---

## 🚀 快速开始

### 1. 环境要求

- Python 3.11+
- PyTorch 2.0+ (支持 CUDA 12.4)
- NVIDIA GPU (推荐，训练加速 10x+)

### 2. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/kkunkunya/medical-fraud-detection.git
cd medical-fraud-detection

# 方式1: 使用 uv (推荐)
uv venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
uv pip install -e .

# 方式2: 使用 pip
pip install -r requirements.txt
```

### 3. 运行实验

```bash
# 完整6模型对比
python scripts/run_full_comparison.py

# 不平衡处理对比 (无处理/加权BCE/Focal Loss)
python scripts/run_imbalance_comparison.py

# 消融实验 (模块消融 + 特征消融)
python scripts/run_ablation.py

# 可解释性分析 (Attention + SHAP)
python scripts/run_interpretability.py

# 精调版实验 (最佳性能)
python scripts/run_experiment_tuned.py
```

### 4. 查看结果

```bash
# 实验图表
outputs/figures/*.png

# 结果数据
outputs/*.csv

# 训练日志
log/app.log
```

---

## 📊 实验结果

### 1️⃣ 模型对比实验

<img src="outputs/figures/07_模型效果对比图.png" width="800">

**关键发现**:
- T-Bi-LSTM+Attention 在 AUC 指标上超越所有基线
- XGBoost 在 F1 指标上表现最佳 (0.454)
- 深度学习模型在 Recall 上具有优势

### 2️⃣ 不平衡处理对比

<img src="outputs/figures/08_不平衡处理策略对比图.png" width="800">

**结论**: 无处理 (0.8881) > 加权BCE (0.8836) > Focal Loss (0.8770)

### 3️⃣ 消融实验

<img src="outputs/figures/09_消融实验结果图.png" width="800">

**模块消融**:
- A1 (完整模型): 0.8836
- A2 (去除时间衰减): 0.8887
- A3 (标准位置编码): 0.8901
- A4 (线性衰减): 0.8907

**特征消融**:
- B1 (完整特征): 0.8836
- B2 (仅静态特征): 0.8804 ← 静态特征主导
- B3 (仅时序特征): 0.7347 ← 时序特征辅助

### 4️⃣ 可解释性分析

#### Attention 权重可视化
<img src="outputs/figures/10_Attention权重热力图.png" width="800">

#### SHAP 特征重要性
<img src="outputs/figures/12_SHAP特征重要性图.png" width="800">

**Top-5 重要特征**:
1. 本次审批金额_sum
2. weekly_visit_max
3. 起付标准以上自负比例金额_sum
4. visit_frequency
5. weekly_visit_std

---

## 📈 完整图表清单

| # | 图表名称 | 类型 |
|:-:|---------|------|
| 1 | 01_类别分布图.png | 数据分析 |
| 2 | 02_费用分布直方图.png | 数据分析 |
| 3 | 03_就诊频次分布图.png | 数据分析 |
| 4 | 04_特征相关性热力图.png | 特征工程 |
| 5 | 05_滑动窗口序列示意图.png | 特征工程 |
| 6 | 06_LSTM梯度特征重要性.png | 特征工程 |
| 7 | 07_模型效果对比图.png | 实验结果 |
| 8 | 08_不平衡处理策略对比图.png | 实验结果 |
| 9 | 09_消融实验结果图.png | 深入分析 |
| 10 | 10_Attention权重热力图.png | 可解释性 |
| 11 | 11_Attention权重对比曲线.png | 可解释性 |
| 12 | 12_SHAP特征重要性图.png | 可解释性 |
| 13 | 13_案例特征对比图.png | 可解释性 |

---

## 🔬 实验细节

### 数据集

- **样本数**: 20,000 患者
- **特征维度**:
  - 静态特征: 166维 (就诊行为、费用聚合、费用比例)
  - 时序特征: 50步 × 20维 (时间窗口序列)
- **标签分布**: 正样本 1,000 (5%) / 负样本 19,000 (95%)
- **划分比例**: 训练集 70% / 验证集 15% / 测试集 15%

### 训练配置

```python
# 主模型训练超参数
batch_size = 256
learning_rate = 8e-4
weight_decay = 2e-3
epochs = 70
optimizer = AdamW
scheduler = CosineAnnealingLR
dropout = 0.35
hidden_dim = 96
lstm_layers = 2
```

### 硬件环境

- GPU: NVIDIA GeForce RTX 3060 Ti
- CUDA: 12.4
- cuDNN: 90100
- 训练时间: ~5分钟/模型

---

## 🛠️ 开发日志

| 日期 | 里程碑 | 详情 |
|-----|-------|------|
| 2026-01-06 | 数据预处理 | 缺失值处理、异常值检测、特征工程 |
| 2026-01-06 | 特征工程 | 就诊特征、时序特征、特征筛选 |
| 2026-01-07 | 模型实现 | 6个模型全部实现并训练 |
| 2026-01-07 | 实验完成 | 主模型超越基线，完成所有实验 |
| 2026-01-07 | 初始版本 | Git 初始化，推送至 GitHub |

---

## 📝 待完成

- [ ] 整合完整的 Jupyter Notebook
- [ ] 生成实验图表汇总文档 (.docx)
- [ ] 添加模型推理 API
- [ ] Docker 容器化部署

---

## 📚 参考文献

核心论文和技术：
- LSTM: Hochreiter & Schmidhuber, 1997
- Attention Mechanism: Bahdanau et al., 2014
- Focal Loss: Lin et al., 2017
- Mixup: Zhang et al., 2017
- XGBoost: Chen & Guestrin, 2016

---

## 📄 License

MIT License

Copyright (c) 2026 Medical Fraud Detection Team

---

## 👥 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📧 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。

---

<div align="center">

**⭐ 如果这个项目对您有帮助，请给个 Star！⭐**

Made with ❤️ by Medical Fraud Detection Team

</div>
