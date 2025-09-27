#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""特征准备脚本

包装新的 pipeline，方便单独执行“prepare”阶段：

    python scripts/prepare_data.py --config config/config.yaml

等价于运行 ``python main.py prepare``，支持选择配置文件、
强制重建特征缓存以及自定义运行名称。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import resolve_paths, run_prepare  # noqa: E402
from src import initialize_project  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="准备 DLFE-LSTM-WSI 数据与特征缓存",
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
        help="运行名称（用于生成 artifacts/runs 目录）",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="忽略已有缓存，强制重新执行流水线",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    runtime = initialize_project(config_path=args.config)
    config = runtime["config"]
    logger = runtime["logger"]

    try:
        paths = resolve_paths(config, args.run_name)
        run_prepare(args, config, paths, logger)
    finally:
        logger.close()


if __name__ == "__main__":
    main()

