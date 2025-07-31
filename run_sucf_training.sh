#!/bin/bash

# SGG-Net训练启动脚本
# 支持单GPU和多GPU分布式训练

set -e
export OMP_NUM_THREADS=16
# 默认配置
CONFIG_FILE="configs/training_config.yaml"
# CONFIG_FILE="/home/fyh0106/work/checkpoints_sucf_8_best/training_config.yaml"
NUM_GPUS=1
GPU_IDS="2"
TEST_MODE=false
RESUME_CHECKPOINT=""
DEBUG_MODE=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --resume)
            RESUME_CHECKPOINT="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --config FILE       指定配置文件 (默认: configs/training_config.yaml)"
            echo "  --gpus NUM          使用的GPU数量 (默认: 1)"
            echo "  --gpu-ids IDS       指定具体的GPU ID，用逗号分隔 (例如: 0,1,2)"
            echo "  --debug             启用分布式调试模式"
            echo "  --test-mode         启用测试模式（快速验证）"
            echo "  --resume FILE       从检查点恢复训练"
            echo "  --help, -h         显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                                      # 单GPU训练 (GPU 0)"
            echo "  $0 --gpus 4                           # 4卡分布式训练 (GPU 0-3)"
            echo "  $0 --gpu-ids 1,3,5                    # 指定GPU 1,3,5"
            echo "  $0 --gpus 2 --gpu-ids 2,3             # 使用GPU 2,3"
            echo "  $0 --gpus 8 --test-mode               # 8卡测试模式"
            echo "  $0 --resume checkpoints/best.pth      # 恢复训练"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

echo "=== SUCF 训练脚本 ==="
echo "时间: $(date)"
echo "工作目录: $(pwd)"

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"

# 处理GPU设置
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $AVAILABLE_GPUS 个可用GPU"

# 如果指定了GPU ID列表，优先使用
if [ -n "$GPU_IDS" ]; then
    # 验证GPU ID格式
    if [[ ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "错误: GPU ID格式不正确，应该用逗号分隔的数字 (例如: 0,1,2)"
        exit 1
    fi
    
    # 将逗号分隔的GPU ID转换为数组
    IFS=',' read -ra GPU_ARRAY <<< "$GPU_IDS"
    NUM_GPUS=${#GPU_ARRAY[@]}
    
    # 验证GPU ID是否有效
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if [ "$gpu_id" -ge "$AVAILABLE_GPUS" ]; then
            echo "错误: GPU ID $gpu_id 不存在，可用GPU范围: 0-$((AVAILABLE_GPUS-1))"
            exit 1
        fi
    done
    
    CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "使用指定的GPU: $GPU_IDS (共 $NUM_GPUS 个)"
else
    # 使用连续的GPU ID
    if ! [[ "$NUM_GPUS" =~ ^[1-8]$ ]]; then
        echo "错误: GPU数量必须是1-8之间的整数"
        exit 1
    fi
    
    if [ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]; then
        echo "错误: 请求 $NUM_GPUS 个GPU，但只有 $AVAILABLE_GPUS 个可用"
        exit 1
    fi
    
    # 生成GPU列表 0,1,2,...
    GPU_LIST=$(seq -s, 0 $((NUM_GPUS-1)))
    CUDA_VISIBLE_DEVICES="$GPU_LIST"
    echo "使用GPU: $GPU_LIST (共 $NUM_GPUS 个)"
fi

echo "GPU数量: $NUM_GPUS"
echo "GPU设备: $CUDA_VISIBLE_DEVICES"
echo "测试模式: $TEST_MODE"

# 检查Python环境
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA设备数: $(python -c 'import torch; print(torch.cuda.device_count())')"

# 创建必要的目录
mkdir -p checkpoints_sucf_amp_$(date +%Y%m%d_%H%M%S)
mkdir -p logs_sucf_amp_$(date +%Y%m%d_%H%M%S)

# 设置环境变量
export PYTHONPATH="${PYTHONPATH}:$(pwd)"


# 构建训练参数
TRAIN_ARGS="--config $CONFIG_FILE"
if [ "$TEST_MODE" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --test-mode"
fi
if [ -n "$RESUME_CHECKPOINT" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --resume $RESUME_CHECKPOINT"
fi

echo "=== 开始训练 ==="
echo "训练参数: $TRAIN_ARGS"

# 生成日志文件名
mkdir -p logs_sucf
LOG_FILE="logs_sucf/training_$(date +%Y%m%d_%H%M%S).log"

# 设置CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

# 根据GPU数量选择训练方式
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "单GPU训练模式"
    echo "使用GPU: $CUDA_VISIBLE_DEVICES"
    python scripts/train_sucf.py $TRAIN_ARGS 2>&1 | tee "$LOG_FILE"

    echo "=== 训练完成 ==="
    echo "检查点保存在: checkpoints_sucf_amp_$(date +%Y%m%d_%H%M%S)/"
    echo "日志保存在: logs_sucf_amp_$(date +%Y%m%d_%H%M%S)/"
fi



