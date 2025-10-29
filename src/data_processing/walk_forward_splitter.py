"""
WalkForwardSplitter
===================

基于配置动态构建 Walk-Forward 时序划分，支持可配置的训练 / 验证 / 测试窗口，
并默认采用半开区间策略避免数据泄露。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


logger = logging.getLogger(__name__)


class WalkForwardSplitter:
    """按照配置动态划分 Walk-Forward 数据集。"""

    def __init__(self, config: Dict):
        walk_cfg = config.get("walk_forward", {})
        self.n_folds = int(walk_cfg.get("n_folds", 0))
        if self.n_folds <= 0:
            raise ValueError("walk_forward.n_folds 必须为正整数")

        self.fold_definitions = walk_cfg.get("fold_definition", {})
        if len(self.fold_definitions) < self.n_folds:
            raise ValueError("fold_definition 中的定义数量不足以支撑指定的 n_folds")

        self.online_cfg = walk_cfg.get("online_learning", {})
        self.weight_cfg = walk_cfg.get("weight_inheritance", {})

    def create_folds(self, data: pd.DataFrame) -> List[Dict]:
        """根据配置创建所有 fold。"""

        if data.empty:
            raise ValueError("传入的数据集为空，无法执行 Walk-Forward 划分")

        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError("Walk-Forward 划分要求数据索引为 DatetimeIndex")

        data_sorted = data.sort_index()
        start_date = data_sorted.index.min()
        folds: List[Dict] = []

        for fold_id in range(1, self.n_folds + 1):
            fold_key = f"fold{fold_id}"
            fold_cfg = self.fold_definitions.get(fold_key)
            if fold_cfg is None:
                raise KeyError(f"未找到 {fold_key} 的配置")

            train_months = int(fold_cfg.get("train_months", 0))
            val_months = int(fold_cfg.get("val_months", 0))
            test_months = int(fold_cfg.get("test_months", 0))
            if min(train_months, val_months, test_months) <= 0:
                raise ValueError(f"{fold_key} 配置中的月份跨度必须为正值")

            train_end = start_date + pd.DateOffset(months=train_months)
            train_df = data_sorted.loc[start_date:train_end]
            if len(train_df) > 1:
                train_df = train_df.iloc[:-1]

            val_start = train_end
            val_end = val_start + pd.DateOffset(months=val_months)
            val_df = data_sorted.loc[val_start:val_end]
            if len(val_df) > 1:
                val_df = val_df.iloc[:-1]

            test_start = val_end
            test_end = test_start + pd.DateOffset(months=test_months)
            test_df = data_sorted.loc[test_start:test_end]
            if fold_id < self.n_folds and len(test_df) > 1:
                test_df = test_df.iloc[:-1]

            fold_payload = {
                "id": fold_id,
                "train": train_df,
                "val": val_df,
                "test": test_df,
                "time_ranges": {
                    "train": (
                        str(train_df.index.min()) if not train_df.empty else None,
                        str(train_df.index.max()) if not train_df.empty else None,
                    ),
                    "val": (
                        str(val_df.index.min()) if not val_df.empty else None,
                        str(val_df.index.max()) if not val_df.empty else None,
                    ),
                    "test": (
                        str(test_df.index.min()) if not test_df.empty else None,
                        str(test_df.index.max()) if not test_df.empty else None,
                    ),
                },
                "size": {
                    "train": len(train_df),
                    "val": len(val_df),
                    "test": len(test_df),
                },
            }
            folds.append(fold_payload)

            logger.info(
                "Fold %d 创建完成: train=%d, val=%d, test=%d",
                fold_id,
                len(train_df),
                len(val_df),
                len(test_df),
            )

        return folds

    @staticmethod
    def validate_folds(folds: List[Dict]) -> bool:
        """验证相邻 fold 之间是否存在时间重叠。"""

        for idx in range(len(folds) - 1):
            curr_test = folds[idx]["test"]
            next_val = folds[idx + 1]["val"]
            if curr_test.empty or next_val.empty:
                continue

            if curr_test.index.max() >= next_val.index.min():
                logger.error(
                    "Fold %d 测试集结束时间 %s 与 Fold %d 验证集开始时间 %s 存在重叠",
                    folds[idx]["id"],
                    curr_test.index.max(),
                    folds[idx + 1]["id"],
                    next_val.index.min(),
                )
                return False
        logger.info("Walk-Forward 划分校验通过")
        return True

    def save_fold_info(self, folds: List[Dict], output_path: Path) -> None:
        """将 fold 划分信息保存为 JSON 方便复现。"""

        payload = []
        for fold in folds:
            payload.append(
                {
                    "id": fold["id"],
                    "time_ranges": fold["time_ranges"],
                    "size": fold["size"],
                }
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=False, indent=2)
        logger.info("Walk-Forward 划分信息已保存至 %s", output_path)

