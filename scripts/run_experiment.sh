#!/bin/bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
CONFIG_FILE=${CONFIG_FILE:-"$PROJECT_ROOT/config/config.yaml"}
RUN_NAME=${RUN_NAME:-"run_$(date +%Y%m%d_%H%M%S)"}

echo "================================================"
echo "  DLFE-LSTM-WSI 实验流程"
echo "  运行名称: $RUN_NAME"
echo "================================================"

echo "[1/3] 准备特征缓存"
python "$PROJECT_ROOT/scripts/prepare_data.py" \
  --config "$CONFIG_FILE" \
  --run-name "$RUN_NAME"

echo "[2/3] 训练模型"
python "$PROJECT_ROOT/scripts/train_model.py" \
  --config "$CONFIG_FILE" \
  --run-name "$RUN_NAME"

echo "[3/3] 测试集评估"
python "$PROJECT_ROOT/scripts/evaluate_model.py" \
  --config "$CONFIG_FILE" \
  --run-name "$RUN_NAME"

echo "================================================"
echo "实验完成，结果位于 experiments/runs/$RUN_NAME"
echo "================================================"

