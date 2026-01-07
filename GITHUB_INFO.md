# 项目说明

## 仓库信息

- **仓库地址**: https://github.com/kkunkunya/medical-fraud-detection
- **项目名称**: 医保欺诈检测系统
- **技术栈**: Python, PyTorch, XGBoost, LightGBM

## 核心亮点

### 1️⃣ 主模型超越基线
- T-Bi-LSTM+Attention: **AUC 0.8825**
- XGBoost 基线: AUC 0.8741
- 超越幅度: **+0.84%**

### 2️⃣ 创新技术
- **时间衰减门控**: g(Δt) = exp(-λ × Δt)
- **门控融合**: 自适应学习时序权重
- **Mixup 数据增强**: 提升泛化能力

### 3️⃣ 完整实验
- ✅ 6模型对比
- ✅ 不平衡处理对比
- ✅ 消融实验
- ✅ 可解释性分析
- ✅ 13张高质量图表

## 文件结构

```
已上传到 GitHub:
├── README.md (英文详细说明)
├── src/ (源代码)
├── scripts/ (实验脚本)
├── docs/ (开发文档)
└── notebooks/ (Jupyter notebook)

已忽略（未上传）:
├── outputs/ (~200MB 结果文件)
├── log/ (日志)
├── .venv/ (虚拟环境)
└── 原始数据/ (~2GB 数据集)
```

## 如何使用

### 克隆仓库
```bash
git clone https://github.com/kkunkunya/medical-fraud-detection.git
cd medical-fraud-detection
```

### 安装依赖
```bash
# 使用 uv (推荐)
uv venv .venv
.venv\Scripts\activate
uv pip install -e .
```

### 运行实验
```bash
# 完整6模型对比
python scripts/run_full_comparison.py

# 消融实验
python scripts/run_ablation.py

# 可解释性分析
python scripts/run_interpretability.py
```

## 注意事项

1. **数据文件未上传**: 由于文件较大，`outputs/` 和原始数据未包含在仓库中
2. **需要 GPU**: 推荐使用 NVIDIA GPU 加速训练（可选）
3. **Python 版本**: 需要 Python 3.11+

## 实验结果

所有实验结果详见 `README.md`，包括：
- 模型性能对比表
- 实验图表（13张）
- 技术架构图
- 训练配置

## GitHub 仓库链接

https://github.com/kkunkunya/medical-fraud-detection

---

**如有问题，请在 GitHub 提 Issue**
