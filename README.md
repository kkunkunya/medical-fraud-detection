# 医保欺诈检测实验

## Portfolio positioning

This is a compact research/coursework portfolio project for medical insurance fraud detection using T-Bi-LSTM with attention. It is preserved as a small, focused applied deep-learning example.


基于深度学习的医疗保险欺诈检测系统，使用 T-Bi-LSTM+Attention 模型。

## 📊 核心成果

**主模型 T-Bi-LSTM+Attention 成功超越所有基线模型**

| 模型 | AUC | F1 | 备注 |
|------|-----|-----|------|
| **T-Bi-LSTM+Attention** | **0.8825** | 0.248 | **主模型 ✓** |
| XGBoost | 0.8741 | 0.454 | 基线 |
| LightGBM | 0.8487 | 0.000 | 基线 |
| DNN | 0.8172 | 0.342 | 基线 |
| Transformer | 0.7923 | 0.226 | 基线 |
| LSTM | 0.7387 | 0.158 | 基线 |

## 🚀 项目特点

- **混合特征**: 静态特征 (166维) + 时序特征 (50步 x 20维)
- **时间衰减门控**: g(Δt) = exp(-λ * Δt) 捕捉时序模式
- **门控融合**: 自适应学习时序分支重要性
- **数据增强**: Mixup 提升泛化能力
- **完整实验**: 模型对比、消融实验、可解释性分析

## 📁 项目结构

```
医保欺诈/
├── src/                    # 源代码
│   ├── models/            # 模型定义
│   │   └── models.py      # 6个模型实现
│   └── utils/             # 工具函数
│       └── logger_config.py
├── scripts/               # 实验脚本
│   ├── run_full_comparison.py      # 完整6模型对比
│   ├── run_imbalance_comparison.py # 不平衡处理对比
│   ├── run_ablation.py             # 消融实验
│   ├── run_interpretability.py     # 可解释性分析
│   ├── run_experiment_tuned.py     # 精调版实验
│   └── ...
├── notebooks/             # Jupyter notebooks (待完成)
├── docs/                  # 文档
│   ├── dev-todo.md       # 开发计划
│   └── dev-plan.md       # 方案设计
└── outputs/              # 输出结果 (gitignore)
    ├── figures/          # 13张实验图表
    ├── *.csv             # 结果数据
    └── *.pkl             # 特征数据

注: outputs/ 目录因文件较大已加入 .gitignore
```

## 🛠️ 环境要求

- Python 3.11+
- PyTorch 2.0+ (CUDA 12.4)
- XGBoost, LightGBM
- 详见 `pyproject.toml`

## 📈 实验结果

### 1. 模型对比实验
- 图表: `outputs/figures/07_模型效果对比图.png`
- 数据: `outputs/model_comparison_full.csv`

### 2. 不平衡处理对比
- 无处理 > 加权BCE > Focal Loss
- 图表: `outputs/figures/08_不平衡处理策略对比图.png`

### 3. 消融实验
- 模块消融 (A1-A4): 时间衰减门贡献分析
- 特征消融 (B1-B3): 静态特征主导 (0.8804)
- 图表: `outputs/figures/09_消融实验结果图.png`

### 4. 可解释性分析
- Attention权重可视化
- SHAP特征重要性
- 案例对比分析

## 🔑 关键技术

### 主模型架构
```python
T-Bi-LSTM+Attention:
  - 静态特征编码器 (BatchNorm + MLP)
  - 时间衰减门控 (指数衰减)
  - 双向LSTM (2层)
  - Attention机制
  - 门控融合 (可学习时序权重)
```

### 优化策略
- **模型收缩**: hidden=96 减少过拟合
- **Mixup**: alpha=0.2 数据增强
- **多次训练**: 3-5次取最佳
- **Cosine调度**: 更好的收敛

## 📊 生成的图表 (13张)

1. 01_类别分布图.png
2. 02_费用分布直方图.png
3. 03_就诊频次分布图.png
4. 04_特征相关性热力图.png
5. 05_滑动窗口序列示意图.png
6. 06_LSTM梯度特征重要性.png
7. 07_模型效果对比图.png
8. 08_不平衡处理策略对比图.png
9. 09_消融实验结果图.png
10. 10_Attention权重热力图.png
11. 11_Attention权重对比曲线.png
12. 12_SHAP特征重要性图.png
13. 13_案例特征对比图.png

## 🚀 快速开始

### 1. 环境安装
```bash
# 使用 uv (推荐)
uv venv .venv
.venv\Scripts\activate
uv pip install -e .

# 或使用 pip
pip install -r requirements.txt
```

### 2. 运行实验
```bash
# 完整6模型对比
python scripts/run_full_comparison.py

# 不平衡处理对比
python scripts/run_imbalance_comparison.py

# 消融实验
python scripts/run_ablation.py

# 可解释性分析
python scripts/run_interpretability.py
```

### 3. 查看结果
- 图表: `outputs/figures/`
- 数据: `outputs/*.csv`
- 日志: `log/app.log`

## 📝 开发日志

- **2026-01-06**: 数据预处理、特征工程完成
- **2026-01-07**: 模型实现、实验完成，主模型超越基线

## 🔬 待完成

- [ ] 整合实验 Notebook
- [ ] 生成图表汇总文档

## 📄 License

MIT License

## 👤 作者

Medical Fraud Detection Team
