# SUCF: 结构门控的跨模态融合网络

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)

> **S**tructurally **U**ncertainty-aware **C**ross-modal **F**usion

通过融合蛋白质的序列和结构信息来预测其生物活性的深度学习模型。

## 🎯 核心创新

- 🧬 **三阶段架构**: 序列编码 → 结构门控 → 跨模态融合
- 🔬 **pLDDT 门控 RGAT**: 利用 pLDDT 分数门控的关系图注意力网络处理结构
- ⚡ **Mamba 融合**: 使用 Mamba 模块高效融合跨模态特征
- 📊 **不确定性感知**: 通过 pLDDT 分数量化结构不确定性

## 📊 性能指标

| 数据集 | Accuracy | AUC-ROC | F1-Score |
|--------|----------|---------|----------|
| Benchmark 1 | 93.89% | 0.97 | 0.93 |
| Benchmark 2 | 91.24% | 0.95 | 0.91 |

## 🏗️ 项目结构

```
SUCF/
├── configs/                    # 配置文件
├── data_processing/            # 数据预处理脚本
├── models/                     # 模型定义
│   ├── sucf_model.py          # 主模型
│   ├── sucf_components.py     # 模型组件
│   ├── relational_gvp.py      # GVP 模块
│   └── relational_gatv3.py    # GATv3 模块
├── utils/                      # 工具函数
├── scripts/                    # 运行脚本
├── train_sucf.py               # 训练主脚本
└── requirements.txt            # 依赖列表
```

## 🚀 快速开始

### 环境配置

```bash
# 创建 Conda 环境
conda create -n sucf_env python=3.10 -y
conda activate sucf_env

# 安装依赖
pip install -r requirements.txt
```

### 数据预处理

```bash
# 准备数据并配置 scripts/run_preprocess.sh
bash scripts/run_preprocess.sh
```

### 模型训练

```bash
# 修改 configs/training_config.yaml
python train_sucf.py --config configs/training_config.yaml
```

## 🔬 技术细节

### 模型架构

1. **序列编码器**: 使用 ESM-2 提取序列特征
2. **结构编码器**: pLDDT 门控的 RGAT 处理结构图
3. **融合模块**: Mamba 模块高效融合跨模态特征
4. **预测头**: MLP 预测生物活性

### 关键组件

- **RelationalGVP**: 基于 Geometric Vector Perceptron 的关系图网络
- **RelationalGATv3**: 基于 GATv3 的关系图注意力网络
- **Mamba**: 高效的序列建模模块

## 📚 相关论文

- [ESM-2](https://www.science.org/doi/10.1126/science.ade2574)
- [GVP](https://arxiv.org/abs/2009.01411)
- [Mamba](https://arxiv.org/abs/2312.00752)

## 📄 许可证

MIT License

---

**⭐ If you find this project useful, please give it a star! ⭐**
