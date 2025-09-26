#!/bin/bash
# DLFE-LSTM-WSI项目环境安装脚本

echo "开始安装DLFE-LSTM-WSI项目环境..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "错误: Conda未安装，请先安装Anaconda或Miniconda"
    exit 1
fi

# 创建conda环境
echo "创建conda环境: dlfe-lstm-wsi"
conda env create -f environment/environment.yml

# 激活环境
echo "激活环境..."
conda activate dlfe-lstm-wsi

# 验证安装
echo "验证核心依赖..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')"

echo "环境安装完成！"
echo "使用 'conda activate dlfe-lstm-wsi' 激活环境"