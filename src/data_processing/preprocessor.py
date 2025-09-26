"""
数据预处理模块
功能：数据标准化、异常值处理、平滑滤波等预处理操作
重要：参数学习仅从训练集，防止数据泄露
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import logging
import json
import pickle
from scipy import signal
from scipy.ndimage import gaussian_filter1d

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Preprocessor:
    """
    数据预处理器

    负责数据标准化、异常值处理、平滑滤波等预处理操作。
    严格遵循参数隔离原则：所有预处理参数仅从训练集学习，
    然后应用到验证集和测试集，防止数据泄露。

    Attributes:
        method (str): 标准化方法
        scaler_params (dict): 标准化参数（从训练集学习）
        is_fitted (bool): 是否已经拟合
    """

    def __init__(self, method: str = 'minmax', feature_range: Tuple[float, float] = (0, 1)):
        """
        初始化预处理器

        Args:
            method: 标准化方法 ('minmax', 'standard', 'robust')
            feature_range: MinMax标准化的目标范围
        """
        self.method = method
        self.feature_range = feature_range
        self.scaler_params = {}
        self.is_fitted = False

        # 支持的标准化方法
        self.supported_methods = ['minmax', 'standard', 'robust', 'maxabs']

        if method not in self.supported_methods:
            raise ValueError(f"不支持的标准化方法: {method}. "
                           f"支持的方法: {self.supported_methods}")

        logger.info(f"预处理器初始化，标准化方法: {method}")

    def fit(self, train_data: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> 'Preprocessor':
        """
        从训练集学习预处理参数

        这是最关键的方法，确保所有预处理参数仅从训练集学习。

        Args:
            train_data: 训练集数据
            exclude_columns: 不需要标准化的列（如时间戳、站点ID等）

        Returns:
            self: 返回自身以支持链式调用
        """
        logger.info("开始从训练集学习预处理参数...")

        # 确定需要处理的列
        if exclude_columns is None:
            exclude_columns = []

        # 获取数值列
        numeric_columns = train_data.select_dtypes(include=[np.number]).columns.tolist()
        process_columns = [col for col in numeric_columns if col not in exclude_columns]

        logger.info(f"需要处理的列: {process_columns}")

        # 根据不同方法学习参数
        if self.method == 'minmax':
            self._fit_minmax(train_data[process_columns])
        elif self.method == 'standard':
            self._fit_standard(train_data[process_columns])
        elif self.method == 'robust':
            self._fit_robust(train_data[process_columns])
        elif self.method == 'maxabs':
            self._fit_maxabs(train_data[process_columns])

        self.scaler_params['columns'] = process_columns
        self.scaler_params['method'] = self.method
        self.is_fitted = True

        logger.info("预处理参数学习完成")
        return self

    def _fit_minmax(self, data: pd.DataFrame) -> None:
        """
        学习MinMax标准化参数

        Args:
            data: 训练数据
        """
        min_vals = data.min()
        max_vals = data.max()

        # 处理常数列（min == max的情况）
        constant_columns = (min_vals == max_vals)
        if constant_columns.any():
            logger.warning(f"发现常数列: {constant_columns[constant_columns].index.tolist()}")

        # 存储参数
        self.scaler_params['min'] = min_vals.to_dict()
        self.scaler_params['max'] = max_vals.to_dict()
        self.scaler_params['feature_range'] = self.feature_range

        logger.info(f"MinMax参数学习完成，范围: {self.feature_range}")

    def _fit_standard(self, data: pd.DataFrame) -> None:
        """
        学习标准化（Z-score）参数

        Args:
            data: 训练数据
        """
        mean_vals = data.mean()
        std_vals = data.std()

        # 处理零标准差的情况
        zero_std = (std_vals == 0) | std_vals.isna()
        if zero_std.any():
            logger.warning(f"发现零标准差列: {zero_std[zero_std].index.tolist()}")
            std_vals[zero_std] = 1.0  # 避免除零

        # 存储参数
        self.scaler_params['mean'] = mean_vals.to_dict()
        self.scaler_params['std'] = std_vals.to_dict()

        logger.info("Standard标准化参数学习完成")

    def _fit_robust(self, data: pd.DataFrame) -> None:
        """
        学习鲁棒标准化参数（基于中位数和四分位距）

        Args:
            data: 训练数据
        """
        median_vals = data.median()
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1

        # 处理零IQR的情况
        zero_iqr = (iqr == 0) | iqr.isna()
        if zero_iqr.any():
            logger.warning(f"发现零IQR列: {zero_iqr[zero_iqr].index.tolist()}")
            iqr[zero_iqr] = 1.0

        # 存储参数
        self.scaler_params['median'] = median_vals.to_dict()
        self.scaler_params['iqr'] = iqr.to_dict()

        logger.info("Robust标准化参数学习完成")

    def _fit_maxabs(self, data: pd.DataFrame) -> None:
        """
        学习MaxAbs标准化参数（按最大绝对值缩放）

        Args:
            data: 训练数据
        """
        max_abs = data.abs().max()

        # 处理零最大值的情况
        zero_max = (max_abs == 0) | max_abs.isna()
        if zero_max.any():
            logger.warning(f"发现零最大值列: {zero_max[zero_max].index.tolist()}")
            max_abs[zero_max] = 1.0

        # 存储参数
        self.scaler_params['max_abs'] = max_abs.to_dict()

        logger.info("MaxAbs标准化参数学习完成")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        使用已学习的参数转换数据

        Args:
            data: 待转换的数据

        Returns:
            pd.DataFrame: 转换后的数据

        Raises:
            RuntimeError: 如果还未调用fit方法
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit方法学习预处理参数")

        data_transformed = data.copy()
        columns = self.scaler_params['columns']

        # 检查列是否存在
        missing_cols = [col for col in columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"数据中缺少列: {missing_cols}")
            columns = [col for col in columns if col in data.columns]

        # 根据不同方法进行转换
        if self.method == 'minmax':
            data_transformed[columns] = self._transform_minmax(data[columns])
        elif self.method == 'standard':
            data_transformed[columns] = self._transform_standard(data[columns])
        elif self.method == 'robust':
            data_transformed[columns] = self._transform_robust(data[columns])
        elif self.method == 'maxabs':
            data_transformed[columns] = self._transform_maxabs(data[columns])

        return data_transformed

    def _transform_minmax(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MinMax标准化转换

        Args:
            data: 待转换数据

        Returns:
            pd.DataFrame: 转换后的数据
        """
        transformed = pd.DataFrame(index=data.index)
        min_val, max_val = self.feature_range

        for col in data.columns:
            if col in self.scaler_params['min']:
                col_min = self.scaler_params['min'][col]
                col_max = self.scaler_params['max'][col]

                if col_max > col_min:
                    # 正常缩放
                    scaled = (data[col] - col_min) / (col_max - col_min)
                    transformed[col] = scaled * (max_val - min_val) + min_val
                else:
                    # 常数列，设为范围中点
                    transformed[col] = (min_val + max_val) / 2
            else:
                transformed[col] = data[col]

        return transformed

    def _transform_standard(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        标准化转换

        Args:
            data: 待转换数据

        Returns:
            pd.DataFrame: 转换后的数据
        """
        transformed = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['mean']:
                mean = self.scaler_params['mean'][col]
                std = self.scaler_params['std'][col]
                transformed[col] = (data[col] - mean) / std
            else:
                transformed[col] = data[col]

        return transformed

    def _transform_robust(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        鲁棒标准化转换

        Args:
            data: 待转换数据

        Returns:
            pd.DataFrame: 转换后的数据
        """
        transformed = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['median']:
                median = self.scaler_params['median'][col]
                iqr = self.scaler_params['iqr'][col]
                transformed[col] = (data[col] - median) / iqr
            else:
                transformed[col] = data[col]

        return transformed

    def _transform_maxabs(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        MaxAbs标准化转换

        Args:
            data: 待转换数据

        Returns:
            pd.DataFrame: 转换后的数据
        """
        transformed = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['max_abs']:
                max_abs = self.scaler_params['max_abs'][col]
                transformed[col] = data[col] / max_abs
            else:
                transformed[col] = data[col]

        return transformed

    def fit_transform(self, data: pd.DataFrame, exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        组合fit和transform操作

        Args:
            data: 训练数据
            exclude_columns: 不需要标准化的列

        Returns:
            pd.DataFrame: 转换后的数据
        """
        self.fit(data, exclude_columns)
        return self.transform(data)

    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        逆转换，将标准化的数据还原到原始尺度

        Args:
            data: 标准化后的数据

        Returns:
            pd.DataFrame: 还原后的数据
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit方法学习预处理参数")

        data_inverse = data.copy()
        columns = self.scaler_params['columns']

        # 根据不同方法进行逆转换
        if self.method == 'minmax':
            data_inverse[columns] = self._inverse_minmax(data[columns])
        elif self.method == 'standard':
            data_inverse[columns] = self._inverse_standard(data[columns])
        elif self.method == 'robust':
            data_inverse[columns] = self._inverse_robust(data[columns])
        elif self.method == 'maxabs':
            data_inverse[columns] = self._inverse_maxabs(data[columns])

        return data_inverse

    def _inverse_minmax(self, data: pd.DataFrame) -> pd.DataFrame:
        """MinMax逆转换"""
        inverse = pd.DataFrame(index=data.index)
        min_val, max_val = self.feature_range

        for col in data.columns:
            if col in self.scaler_params['min']:
                col_min = self.scaler_params['min'][col]
                col_max = self.scaler_params['max'][col]

                if col_max > col_min:
                    scaled_back = (data[col] - min_val) / (max_val - min_val)
                    inverse[col] = scaled_back * (col_max - col_min) + col_min
                else:
                    inverse[col] = col_min
            else:
                inverse[col] = data[col]

        return inverse

    def _inverse_standard(self, data: pd.DataFrame) -> pd.DataFrame:
        """标准化逆转换"""
        inverse = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['mean']:
                mean = self.scaler_params['mean'][col]
                std = self.scaler_params['std'][col]
                inverse[col] = data[col] * std + mean
            else:
                inverse[col] = data[col]

        return inverse

    def _inverse_robust(self, data: pd.DataFrame) -> pd.DataFrame:
        """鲁棒标准化逆转换"""
        inverse = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['median']:
                median = self.scaler_params['median'][col]
                iqr = self.scaler_params['iqr'][col]
                inverse[col] = data[col] * iqr + median
            else:
                inverse[col] = data[col]

        return inverse

    def _inverse_maxabs(self, data: pd.DataFrame) -> pd.DataFrame:
        """MaxAbs逆转换"""
        inverse = pd.DataFrame(index=data.index)

        for col in data.columns:
            if col in self.scaler_params['max_abs']:
                max_abs = self.scaler_params['max_abs'][col]
                inverse[col] = data[col] * max_abs
            else:
                inverse[col] = data[col]

        return inverse

    def detect_outliers(self, data: pd.DataFrame, method: str = 'iqr',
                       threshold: float = 1.5) -> pd.DataFrame:
        """
        检测异常值

        Args:
            data: 输入数据
            method: 检测方法 ('iqr', 'zscore', 'isolation_forest')
            threshold: 阈值

        Returns:
            pd.DataFrame: 布尔数据框，True表示异常值
        """
        outliers = pd.DataFrame(False, index=data.index, columns=data.columns)

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (data[col] < (Q1 - threshold * IQR)) | \
                               (data[col] > (Q3 + threshold * IQR))

            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers[col] = z_scores > threshold

            else:
                raise ValueError(f"不支持的异常值检测方法: {method}")

        num_outliers = outliers.sum().sum()
        if num_outliers > 0:
            logger.info(f"检测到 {num_outliers} 个异常值")

        return outliers

    def handle_outliers(self, data: pd.DataFrame, method: str = 'clip',
                       outliers: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        处理异常值

        Args:
            data: 输入数据
            method: 处理方法 ('clip', 'remove', 'replace_mean', 'replace_median')
            outliers: 异常值标记，如果为None则自动检测

        Returns:
            pd.DataFrame: 处理后的数据
        """
        data_processed = data.copy()

        if outliers is None:
            outliers = self.detect_outliers(data)

        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in outliers.columns:
                continue

            outlier_mask = outliers[col]

            if method == 'clip':
                # 将异常值裁剪到合理范围
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                data_processed.loc[outlier_mask, col] = data_processed.loc[outlier_mask, col].clip(lower, upper)

            elif method == 'remove':
                # 删除异常值（设为NaN）
                data_processed.loc[outlier_mask, col] = np.nan

            elif method == 'replace_mean':
                # 用均值替换
                mean_val = data.loc[~outlier_mask, col].mean()
                data_processed.loc[outlier_mask, col] = mean_val

            elif method == 'replace_median':
                # 用中位数替换
                median_val = data.loc[~outlier_mask, col].median()
                data_processed.loc[outlier_mask, col] = median_val

            else:
                raise ValueError(f"不支持的异常值处理方法: {method}")

        logger.info(f"异常值处理完成，方法: {method}")
        return data_processed

    def smooth_data(self, data: pd.DataFrame, method: str = 'gaussian',
                   window_size: int = 5, **kwargs) -> pd.DataFrame:
        """
        数据平滑处理

        Args:
            data: 输入数据
            method: 平滑方法 ('gaussian', 'moving_average', 'savgol')
            window_size: 窗口大小
            **kwargs: 其他方法特定参数

        Returns:
            pd.DataFrame: 平滑后的数据
        """
        data_smoothed = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'gaussian':
                sigma = kwargs.get('sigma', 1)
                data_smoothed[col] = gaussian_filter1d(data[col].values, sigma=sigma)

            elif method == 'moving_average':
                data_smoothed[col] = data[col].rolling(window=window_size, center=True).mean()
                # 填充边缘的NaN值
                data_smoothed[col].fillna(data[col], inplace=True)

            elif method == 'savgol':
                polyorder = kwargs.get('polyorder', 2)
                if window_size > len(data):
                    window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
                if window_size > polyorder:
                    data_smoothed[col] = signal.savgol_filter(data[col].values, window_size, polyorder)

            else:
                raise ValueError(f"不支持的平滑方法: {method}")

        logger.info(f"数据平滑完成，方法: {method}, 窗口大小: {window_size}")
        return data_smoothed

    def save_params(self, filepath: Union[str, Path]) -> None:
        """
        保存预处理参数

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'method': self.method,
            'feature_range': self.feature_range,
            'scaler_params': self.scaler_params,
            'is_fitted': self.is_fitted
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(params, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"预处理参数已保存: {filepath}")

    def load_params(self, filepath: Union[str, Path]) -> None:
        """
        加载预处理参数

        Args:
            filepath: 参数文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"参数文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                params = json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.method = params['method']
        self.feature_range = tuple(params['feature_range'])
        self.scaler_params = params['scaler_params']
        self.is_fitted = params['is_fitted']

        logger.info(f"预处理参数已加载: {filepath}")