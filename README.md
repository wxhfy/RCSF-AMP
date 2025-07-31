# SUCF: 结构门控的跨模态融合网络 (Sructurally Uncertainty-aware Cross-modal Fusion)

本项目是一个先进的深度学习框架，旨在通过融合蛋白质的序列和结构信息来预测其生物活性。该模型的核心创新在于其三阶段架构，利用 pLDDT 分数门控的关系图注意力网络（RGAT）来处理结构，并使用 Mamba 模块高效地融合跨模态特征。

模型名称 **SUCF** 来源于其核心思想: **S**ructurally **U**ncertainty-aware **C**ross-modal **F**usion。


## 项目结构

```text
SUCF/
├── benchmark1_graph/           # Benchmark1 预处理数据
│   ├── embeddings/
│   ├── graphs/
│   ├── sequences_L_lt_inf.json
│   ├── test.txt
│   ├── train.txt
│   └── val.txt
├── benchmark2_graph/           # Benchmark2 预处理数据
│   └── ...
├── configs/                    # 配置文件
│   ├── __init__.py
│   └── training_config.yaml
├── data_processing/            # 数据预处理脚本
│   ├── esm_embedder.py
│   ├── feature_calculator.py
│   ├── graph_constructor.py
│   ├── main_preprocess.py
│   └── pdb_parser.py
├── models/                     # 模型定义
│   ├── sucf_model.py
│   ├── sucf_components.py
│   ├── relational_gvp.py
│   ├── relational_gatv3.py
│   └── ...
├── outputs/                    # 训练输出 (日志, 检查点)
│   ├── checkpoints/
│   └── logs/
├── scripts/                    # 数据预处理脚本
│   └── run_preprocess.sh
├── utils/                      # 工具函数
│   ├── config_utils.py
│   ├── datasets.py
│   ├── early_stopping.py
│   ├── metrics.py
│   └── sucf_losses.py
├── requirements.txt            # Python 环境依赖
└── train_sucf.py               # 模型训练主脚本
```

## 环境配置

建议使用 Conda 创建独立的虚拟环境来管理项目依赖。

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd SUCF
    ```

2.  **使用 Conda 创建并激活虚拟环境**
    (我们推荐使用 Python 3.10，因为项目的缓存文件 `*.pyc` 是基于该版本生成的)
    ```bash
    # 创建名为 sucf_env 的新环境
    conda create -n sucf_env python=3.10 -y

    # 激活环境
    conda activate sucf_env
    ```

3.  **安装依赖库**
    项目所需的所有依赖库都已在 `requirements.txt` 中列出。运行以下命令进行安装：
    ```bash
    pip install -r requirements.txt
    ```
    *注意: `torch` 和 `torch_geometric` 的安装可能与您的 CUDA 版本有关。如果遇到问题，请参考官方文档进行安装。*

## 使用方法

### 第一步：数据预处理

在模型训练之前，需要对原始数据（如 `.pdb.gz` 文件）进行预处理，以提取序列和结构特征并构建图数据。

1.  **准备数据**: 将您的原始 PDB 文件按数据集（如 `AMP_train`, `DECOY_train` 等）存放在指定的数据根目录中。
2.  **配置脚本**: 打开 `scripts/run_preprocess.sh` 文件。根据您的实际路径，修改 `data_root` (输入数据根目录)、`output_dir` (预处理结果输出目录) 和 `esm_model_base_path` (本地 ESM 模型路径，可选)。
3.  **运行预处理**:
    ```bash
    bash scripts/run_preprocess.sh
    ```
    该脚本会为 `benchmark1` 和 `benchmark2` 两个数据集分别生成图数据，并保存在 `--output_dir` 指定的目录中。

### 第二步：模型训练

数据预处理完成后，可以开始训练模型。

1.  **修改配置文件**: 打开 `configs/training_config.yaml`。
    * 将 `paths.data_root` 修改为您预处理好的图数据目录 (例如 `./benchmark1_graph/`)。
    * 根据需要调整 `training` 部分的参数，如 `batch_size`, `num_workers` 等。

2.  **开始训练**: 运行 `train_sucf.py` 脚本，并指定配置文件。
    ```bash
    python train_sucf.py --config configs/training_config.yaml
    ```
    训练脚本会自动执行两阶段的训练流程。训练过程中的日志、模型检查点和结果将保存在 `outputs` 目录下。

