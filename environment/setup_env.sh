#!/bin/bash
# DLFE-LSTM-WSI项目环境安装脚本 (支持 GPU/CPU)

set -e

ENV_NAME="dlfe-lstm-wsi"
YAML_FILE="environment/environment.yml"
REQ_FILE="environment/requirements.txt"

usage() {
    echo "Usage: $0 [--cpu]"
    echo "  --cpu  使用CPU版本的PyTorch"
}

USE_CPU=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu)
            USE_CPU=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            usage
            exit 1
            ;;
    esac
done

if ! command -v conda &> /dev/null; then
    echo "错误: Conda未安装，请先安装Anaconda或Miniconda"
    exit 1
fi

echo "创建conda环境: ${ENV_NAME}"
conda env remove -n ${ENV_NAME} -y >/dev/null 2>&1 || true
conda env create -f ${YAML_FILE}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

if ${USE_CPU}; then
    echo "切换至CPU版本的PyTorch"
    pip uninstall -y torch torchvision torchaudio || true
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "安装额外pip依赖"
pip install -r ${REQ_FILE}
pip install pytest pytest-cov black flake8 mypy

echo "验证核心依赖"
python - <<'PY'
import torch, numpy, pandas, plotly
print(f"PyTorch版本: {torch.__version__}")
print(f"NumPy版本: {numpy.__version__}")
print(f"Pandas版本: {pandas.__version__}")
print(f"Plotly版本: {plotly.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
PY

echo "环境安装完成！"
echo "使用 'conda activate ${ENV_NAME}' 激活环境"