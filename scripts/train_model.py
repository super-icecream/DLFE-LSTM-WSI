#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""训练脚本包装

等价于 ``python main.py train``，方便保持既有命令习惯。
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import resolve_paths, run_train  # noqa: E402
from src import initialize_project  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 DLFE-LSTM-WSI 模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="自定义运行名称（默认使用时间戳）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="强制重新执行数据与特征流水线",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_name is None:
        args.run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    runtime = initialize_project(config_path=args.config)
    config = runtime["config"]
    logger = runtime["logger"]

    try:
        paths = resolve_paths(config, args.run_name)
        run_train(args, config, paths, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

