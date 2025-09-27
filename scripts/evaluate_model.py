#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""评估脚本包装

与 ``python main.py test`` 一致，用于复用现有命令行习惯。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import resolve_paths, run_test  # noqa: E402
from src import initialize_project  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在测试集上评估 DLFE-LSTM-WSI 模型",
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
        required=True,
        help="需要评估的运行名称（对应 checkpoints/results 目录）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="如有必要重建特征缓存",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = initialize_project(config_path=args.config)
    config = runtime["config"]
    logger = runtime["logger"]

    try:
        paths = resolve_paths(config, args.run_name)
        run_test(args, config, paths, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

