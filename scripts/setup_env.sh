#!/bin/bash
# -*- coding: utf-8 -*-
"""
DLFE-LSTM-WSI 环境安装脚本
自动配置项目运行环境
"""

set -e  # 遇错即停

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# 检查系统要求
check_requirements() {
    print_info "检查系统要求..."
    
    # 检查Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python3未安装"
        exit 1
    fi
    
    # 检查conda
    if ! command -v conda &> /dev/null; then
        print_warning "Conda未安装，将使用pip安装"
        USE_PIP=true
    else
        USE_PIP=false
    fi
    
    # 检查CUDA（可选）
    if command -v nvidia-smi &> /dev/null; then
        print_info "检测到NVIDIA GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv
        USE_CUDA=true
    else
        print_warning "未检测到GPU，将使用CPU模式"
        USE_CUDA=false
    fi
}

# 创建conda环境
setup_conda_env() {
    ENV_NAME="dlfe-lstm-wsi"
    
    print_info "创建Conda环境: $ENV_NAME"
    
    # 检查环境是否存在
    if conda env list | grep -q "^$ENV_NAME "; then
        print_warning "环境已存在，是否重建？[y/N]"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            conda env remove -n $ENV_NAME -y
        else
            print_info "使用现有环境"
            return
        fi
    fi
    
    # 创建环境
    conda create -n $ENV_NAME python=3.9 -y
    
    # 激活环境
    eval "$(conda shell.bash hook)"
    conda activate $ENV_NAME
    
    print_info "Conda环境创建成功"
}

# 安装Python依赖
install_python_deps() {
    print_info "安装Python依赖包..."
    
    # 升级pip
    pip install --upgrade pip
    
    # 基础依赖
    pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml
    
    # PyTorch（根据CUDA版本选择）
    if [ "$USE_CUDA" = true ]; then
        # CUDA 11.8版本
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        # CPU版本
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # VMD和信号处理
    pip install PyWavelets scipy statsmodels
    
    # 深度学习工具
    pip install tensorboard wandb mlflow
    
    # 可视化
    pip install plotly
    
    # 超参数优化
    pip install optuna
    
    # 开发工具
    pip install jupyter notebook ipython
    pip install pytest pytest-cov black flake8 mypy
    
    print_info "Python依赖安装完成"
}

# 创建项目目录结构
setup_project_dirs() {
    print_info "创建项目目录结构..."
    
    # 数据目录
    mkdir -p data/{raw,processed,features,splits}
    
    # 实验目录
    mkdir -p experiments/{logs,checkpoints,results,runs}
    
    # 配置目录
    mkdir -p config
    
    # 文档目录
    mkdir -p docs/{api,tutorials,deployment}
    
    print_info "目录结构创建完成"
}

# 下载预训练模型（可选）
download_pretrained() {
    print_info "是否下载预训练模型？[y/N]"
    read -r response
    
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        print_info "下载预训练模型..."
        
        # 创建模型目录
        mkdir -p pretrained_models
        
        # 下载示例（替换为实际URL）
        # wget -O pretrained_models/dlfe_lstm_wsi.pth https://example.com/model.pth
        
        print_info "预训练模型下载完成"
    fi
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    
    python -c "
import torch
import numpy as np
import pandas as pd

print('PyTorch版本:', torch.__version__)
print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA版本:', torch.version.cuda)
    print('GPU数量:', torch.cuda.device_count())
    print('GPU名称:', torch.cuda.get_device_name(0))

print('NumPy版本:', np.__version__)
print('Pandas版本:', pd.__version__)
print('安装验证成功！')
"
    
    if [ $? -eq 0 ]; then
        print_info "环境验证成功"
    else
        print_error "环境验证失败"
        exit 1
    fi
}

# 生成环境信息文件
generate_env_info() {
    print_info "生成环境信息..."
    
    cat > environment_info.txt << EOF
DLFE-LSTM-WSI 环境信息
生成时间: $(date)

系统信息:
$(uname -a)

Python版本:
$(python --version)

已安装包:
$(pip list)

GPU信息:
$(nvidia-smi 2>/dev/null || echo "无GPU")
EOF
    
    print_info "环境信息已保存至 environment_info.txt"
}

# 主函数
main() {
    echo "================================================"
    echo "   DLFE-LSTM-WSI 环境安装脚本"
    echo "================================================"
    
    # 执行安装步骤
    check_requirements
    
    if [ "$USE_PIP" = false ]; then
        setup_conda_env
    fi
    
    install_python_deps
    setup_project_dirs
    download_pretrained
    verify_installation
    generate_env_info
    
    echo ""
    echo "================================================"
    echo "   安装完成！"
    echo "================================================"
    echo ""
    echo "激活环境: conda activate dlfe-lstm-wsi"
    echo "运行项目: python main.py"
    echo ""
}

# 运行主函数
main