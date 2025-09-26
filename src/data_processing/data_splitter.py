"""
数据集划分模块
功能：按时间顺序严格划分数据集（70%训练、20%验证、10%测试）
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Union
from pathlib import Path
import logging
from datetime import datetime, timedelta

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplitter:
    """
    时序数据集划分器

    按照时间顺序严格划分数据集，确保训练集、验证集和测试集之间无时间重叠，
    防止数据泄露，保持时序连续性。

    Attributes:
        train_ratio (float): 训练集比例
        val_ratio (float): 验证集比例
        test_ratio (float): 测试集比例
        seed (int): 随机种子（用于可重复性）
    """

    def __init__(self, train_ratio: float = 0.7, val_ratio: float = 0.2,
                 test_ratio: float = 0.1, seed: int = 42):
        """
        初始化数据划分器

        Args:
            train_ratio: 训练集比例，默认0.7
            val_ratio: 验证集比例，默认0.2
            test_ratio: 测试集比例，默认0.1
            seed: 随机种子，默认42

        Raises:
            ValueError: 如果比例之和不等于1
        """
        # 验证比例
        total_ratio = train_ratio + val_ratio + test_ratio
        if not np.isclose(total_ratio, 1.0):
            raise ValueError(f"数据集比例之和必须为1，当前为: {total_ratio}")

        if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
            raise ValueError("所有数据集比例必须大于0")

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        # 设置随机种子
        np.random.seed(seed)

        # 存储划分信息
        self.split_info = {
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': seed,
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"数据划分器初始化: 训练集{train_ratio:.0%}, "
                   f"验证集{val_ratio:.0%}, 测试集{test_ratio:.0%}")

    def split_temporal(self, data: pd.DataFrame,
                       keep_continuity: bool = True,
                       gap_hours: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据集

        严格按照时间顺序进行划分，确保训练集在前，验证集居中，测试集在后。
        可选择在数据集之间添加时间间隔以减少数据泄露风险。

        Args:
            data: 输入数据，必须有时间索引
            keep_continuity: 是否保持每个数据集内部的时序连续性
            gap_hours: 数据集之间的时间间隔（小时）

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                (训练集, 验证集, 测试集)

        Raises:
            ValueError: 如果数据没有时间索引
        """
        # 验证时间索引
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("数据必须有DatetimeIndex时间索引")

        # 确保数据按时间排序
        data_sorted = data.sort_index()
        total_samples = len(data_sorted)

        logger.info(f"开始时序划分，总样本数: {total_samples}")
        logger.info(f"时间范围: {data_sorted.index[0]} 至 {data_sorted.index[-1]}")

        if gap_hours > 0 and keep_continuity:
            # 考虑时间间隔的划分
            train_data, val_data, test_data = self._split_with_gap(
                data_sorted, gap_hours
            )
        else:
            # 简单按比例划分
            train_end_idx = int(total_samples * self.train_ratio)
            val_end_idx = int(total_samples * (self.train_ratio + self.val_ratio))

            train_data = data_sorted.iloc[:train_end_idx]
            val_data = data_sorted.iloc[train_end_idx:val_end_idx]
            test_data = data_sorted.iloc[val_end_idx:]

        # 更新划分信息
        self.split_info.update({
            'total_samples': total_samples,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_time_range': [str(train_data.index[0]), str(train_data.index[-1])],
            'val_time_range': [str(val_data.index[0]), str(val_data.index[-1])],
            'test_time_range': [str(test_data.index[0]), str(test_data.index[-1])],
            'gap_hours': gap_hours
        })

        # 验证划分结果
        self._validate_split(train_data, val_data, test_data)

        # 日志输出
        logger.info(f"划分完成:")
        logger.info(f"  训练集: {len(train_data)} 样本 "
                   f"({train_data.index[0]} ~ {train_data.index[-1]})")
        logger.info(f"  验证集: {len(val_data)} 样本 "
                   f"({val_data.index[0]} ~ {val_data.index[-1]})")
        logger.info(f"  测试集: {len(test_data)} 样本 "
                   f"({test_data.index[0]} ~ {test_data.index[-1]})")

        return train_data, val_data, test_data

    def _split_with_gap(self, data: pd.DataFrame,
                       gap_hours: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        带时间间隔的数据划分

        在训练集、验证集和测试集之间添加时间间隔，
        避免相邻数据集之间的信息泄露。

        Args:
            data: 排序后的数据
            gap_hours: 间隔小时数

        Returns:
            划分后的三个数据集
        """
        gap = timedelta(hours=gap_hours)
        total_samples = len(data)

        # 计算有效样本数（扣除间隔）
        # 假设数据均匀分布，估算间隔占用的样本数
        time_range = data.index[-1] - data.index[0]
        avg_freq = time_range / total_samples
        gap_samples = int(gap / avg_freq)

        # 调整后的样本数
        effective_samples = total_samples - 2 * gap_samples
        train_samples = int(effective_samples * self.train_ratio)
        val_samples = int(effective_samples * self.val_ratio)

        # 划分索引
        train_end_idx = train_samples
        val_start_idx = train_end_idx + gap_samples
        val_end_idx = val_start_idx + val_samples
        test_start_idx = val_end_idx + gap_samples

        # 提取数据集
        train_data = data.iloc[:train_end_idx]
        val_data = data.iloc[val_start_idx:val_end_idx]
        test_data = data.iloc[test_start_idx:]

        return train_data, val_data, test_data

    def split_by_station(self, data: pd.DataFrame,
                        station_column: str = 'station') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
        """
        按站点分别划分数据

        对每个站点独立进行70/20/10划分，保持站点内的时序连续性。

        Args:
            data: 包含站点信息的数据
            station_column: 站点列名

        Returns:
            Dict: 站点名到(训练集, 验证集, 测试集)的映射
        """
        if station_column not in data.columns:
            raise ValueError(f"数据中不存在列: {station_column}")

        stations = data[station_column].unique()
        station_splits = {}

        logger.info(f"按站点划分，共 {len(stations)} 个站点")

        for station in stations:
            station_data = data[data[station_column] == station]

            # 对每个站点进行划分
            train, val, test = self.split_temporal(station_data)
            station_splits[station] = (train, val, test)

            logger.info(f"站点 {station} 划分完成: "
                       f"训练{len(train)}, 验证{len(val)}, 测试{len(test)}")

        return station_splits

    def split_with_sliding_window(self, data: pd.DataFrame,
                                 window_size: int = 144,
                                 stride: int = 1) -> Dict[str, pd.DataFrame]:
        """
        使用滑动窗口创建时序样本

        适用于LSTM等需要固定长度输入序列的模型。

        Args:
            data: 原始时序数据
            window_size: 窗口大小（时间步数）
            stride: 滑动步长

        Returns:
            Dict: 包含划分后的窗口数据
        """
        # 先进行基本的时序划分
        train_data, val_data, test_data = self.split_temporal(data)

        # 为每个数据集创建滑动窗口
        result = {
            'train_windows': self._create_windows(train_data, window_size, stride),
            'val_windows': self._create_windows(val_data, window_size, stride),
            'test_windows': self._create_windows(test_data, window_size, stride)
        }

        logger.info(f"滑动窗口创建完成: "
                   f"训练{len(result['train_windows'])}个, "
                   f"验证{len(result['val_windows'])}个, "
                   f"测试{len(result['test_windows'])}个")

        return result

    def _create_windows(self, data: pd.DataFrame,
                       window_size: int,
                       stride: int) -> List[pd.DataFrame]:
        """
        创建滑动窗口样本

        Args:
            data: 输入数据
            window_size: 窗口大小
            stride: 步长

        Returns:
            List[pd.DataFrame]: 窗口列表
        """
        windows = []
        for i in range(0, len(data) - window_size + 1, stride):
            window = data.iloc[i:i + window_size]
            windows.append(window)
        return windows

    def _validate_split(self, train: pd.DataFrame,
                       val: pd.DataFrame,
                       test: pd.DataFrame) -> None:
        """
        验证数据划分的正确性

        检查时间顺序、数据重叠等问题。

        Args:
            train: 训练集
            val: 验证集
            test: 测试集

        Raises:
            ValueError: 如果划分存在问题
        """
        # 检查是否为空
        if len(train) == 0 or len(val) == 0 or len(test) == 0:
            raise ValueError("存在空数据集，请检查数据量和划分比例")

        # 检查时间顺序
        if train.index[-1] > val.index[0]:
            logger.warning("训练集和验证集存在时间重叠")

        if val.index[-1] > test.index[0]:
            logger.warning("验证集和测试集存在时间重叠")

        # 检查比例
        total = len(train) + len(val) + len(test)
        actual_train_ratio = len(train) / total
        actual_val_ratio = len(val) / total
        actual_test_ratio = len(test) / total

        # 允许5%的误差
        tolerance = 0.05
        if abs(actual_train_ratio - self.train_ratio) > tolerance:
            logger.warning(f"训练集实际比例 {actual_train_ratio:.2%} "
                         f"与预期 {self.train_ratio:.0%} 相差较大")

    def save_split_info(self, output_path: Union[str, Path],
                       format: str = 'json') -> None:
        """
        保存数据划分信息

        Args:
            output_path: 输出文件路径
            format: 保存格式 ('json' 或 'pickle')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.split_info, f, indent=2, ensure_ascii=False)
            logger.info(f"划分信息已保存至: {output_path}")

        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(self.split_info, f)
            logger.info(f"划分信息已保存至: {output_path}")

        else:
            raise ValueError(f"不支持的格式: {format}")

    def load_split_info(self, info_path: Union[str, Path]) -> Dict:
        """
        加载已保存的划分信息

        Args:
            info_path: 信息文件路径

        Returns:
            Dict: 划分信息字典
        """
        info_path = Path(info_path)

        if info_path.suffix == '.json':
            with open(info_path, 'r', encoding='utf-8') as f:
                split_info = json.load(f)

        elif info_path.suffix == '.pkl':
            with open(info_path, 'rb') as f:
                split_info = pickle.load(f)

        else:
            raise ValueError(f"不支持的文件格式: {info_path.suffix}")

        self.split_info = split_info
        logger.info(f"已加载划分信息: {info_path}")

        return split_info

    def save_splits(self, train: pd.DataFrame,
                   val: pd.DataFrame,
                   test: pd.DataFrame,
                   output_dir: Union[str, Path]) -> None:
        """
        保存划分后的数据集到文件

        Args:
            train: 训练集
            val: 验证集
            test: 测试集
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存数据集
        train_path = output_dir / "train.csv"
        val_path = output_dir / "val.csv"
        test_path = output_dir / "test.csv"

        train.to_csv(train_path)
        val.to_csv(val_path)
        test.to_csv(test_path)

        # 保存划分信息
        info_path = output_dir / "split_info.json"
        self.save_split_info(info_path)

        logger.info(f"数据集已保存至: {output_dir}")
        logger.info(f"  训练集: {train_path}")
        logger.info(f"  验证集: {val_path}")
        logger.info(f"  测试集: {test_path}")

    def get_split_statistics(self, train: pd.DataFrame,
                            val: pd.DataFrame,
                            test: pd.DataFrame) -> Dict:
        """
        获取数据集划分的统计信息

        Args:
            train: 训练集
            val: 验证集
            test: 测试集

        Returns:
            Dict: 统计信息
        """
        stats = {
            'dataset_sizes': {
                'train': len(train),
                'val': len(val),
                'test': len(test),
                'total': len(train) + len(val) + len(test)
            },
            'dataset_ratios': {
                'train': len(train) / (len(train) + len(val) + len(test)),
                'val': len(val) / (len(train) + len(val) + len(test)),
                'test': len(test) / (len(train) + len(val) + len(test))
            },
            'time_ranges': {
                'train': {
                    'start': str(train.index[0]),
                    'end': str(train.index[-1]),
                    'duration': str(train.index[-1] - train.index[0])
                },
                'val': {
                    'start': str(val.index[0]),
                    'end': str(val.index[-1]),
                    'duration': str(val.index[-1] - val.index[0])
                },
                'test': {
                    'start': str(test.index[0]),
                    'end': str(test.index[-1]),
                    'duration': str(test.index[-1] - test.index[0])
                }
            },
            'feature_statistics': {}
        }

        # 计算每个特征的统计信息
        for col in train.columns:
            if col in val.columns and col in test.columns:
                stats['feature_statistics'][col] = {
                    'train': {
                        'mean': float(train[col].mean()),
                        'std': float(train[col].std()),
                        'min': float(train[col].min()),
                        'max': float(train[col].max())
                    },
                    'val': {
                        'mean': float(val[col].mean()),
                        'std': float(val[col].std()),
                        'min': float(val[col].min()),
                        'max': float(val[col].max())
                    },
                    'test': {
                        'mean': float(test[col].mean()),
                        'std': float(test[col].std()),
                        'min': float(test[col].min()),
                        'max': float(test[col].max())
                    }
                }

        return stats