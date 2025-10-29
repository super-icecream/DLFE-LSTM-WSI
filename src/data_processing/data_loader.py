"""
数据加载模块
功能：加载甘肃光伏功率预测数据集，处理时序数据，进行数据完整性检查
GPU优化：添加PyTorch DataLoader支持，实现GPU加速数据传输
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import os
import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from pandas.tseries.frequencies import to_offset

logger = logging.getLogger(__name__)


class DataLoader:
    """
    光伏功率预测数据加载器

    负责加载CSV格式的原始数据文件，处理时间戳，进行数据完整性检查，
    支持多站点数据加载和合并。

    Attributes:
        data_path (str): 数据文件路径
        required_columns (list): 必需的数据列
        freq (str): 数据采样频率
        config (dict): 配置参数
    """

    def __init__(self, data_path: str = "./data/raw", config_path: Optional[str] = None):
        """
        初始化数据加载器

        Args:
            data_path (str): 原始数据文件路径
            config_path (str, optional): 配置文件路径
        """
        self.data_path = Path(data_path)
        self.required_columns = ['power', 'irradiance', 'temperature', 'pressure', 'humidity']
        self.freq = '15T'  # 15分钟采样频率
        self.station_column = 'station'
        self.frequency_minutes = 15

        # 加载配置文件
        if config_path:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                # 提取预处理配置
                preprocessing_config = full_config.get('preprocessing', {})
                self.config = self._merge_configs(
                    self._get_default_config(),
                    preprocessing_config
                )
        else:
            self.config = self._get_default_config()

        logger.info(f"数据加载器初始化完成，数据路径: {self.data_path}")

    def _get_default_config(self) -> Dict:
        """
        获取默认配置参数

        Returns:
            Dict: 默认配置字典
        """
        return {
            'missing_values': {
                'strategy': 'interpolation',
                'method': 'linear',
                'max_consecutive': 6,
            },
            'missing_threshold': 0.3,  # 缺失值比例阈值（向后兼容）
            'interpolation_method': 'linear',  # 插值方法（向后兼容）
            'max_consecutive_missing': 6,  # 最大连续缺失数（向后兼容）
            'outlier_detection': {
                'method': 'physical',  # 异常值检测方法: 'physical', 'iqr', 'zscore'
                'iqr_threshold': 1.5,  # IQR阈值（仅在method='iqr'时使用）
                'apply_to': ['power', 'irradiance', 'temperature', 'pressure', 'humidity'],
                # 物理约束范围（基于光伏系统和气象学的领域知识）
                'physical_ranges': {
                    'power': [0, 55000],  # kW，装机容量50MW + 10%过载保护
                    'irradiance': [0, 1200],  # W/m²，地面太阳辐照度物理上限
                    'temperature': [-40, 60],  # °C，极端气候范围
                    'pressure': [850, 1100],  # hPa，考虑海拔的气压范围
                    'humidity': [0, 100],  # %，相对湿度的物理范围
                },
                # 明确的错误标记值（将被视为缺失值）
                'error_markers': [-99, -999, -9999],
            },
            # 向后兼容的顶层字段
            'outlier_method': 'physical',
            'iqr_threshold': 1.5,
            'physical_ranges': {
                'power': [0, 55000],
                'irradiance': [0, 1200],
                'temperature': [-40, 60],
                'pressure': [850, 1100],
                'humidity': [0, 100],
            },
            'error_markers': [-99, -999, -9999],
        }
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        合并默认配置和加载的配置
        
        Args:
            default: 默认配置
            loaded: 从文件加载的配置
            
        Returns:
            Dict: 合并后的配置
        """
        merged = default.copy()
        
        # 处理缺失值配置
        if 'missing_values' in loaded:
            missing_config = loaded['missing_values']
            merged['interpolation_method'] = missing_config.get('method', default['interpolation_method'])
            merged['max_consecutive_missing'] = missing_config.get('max_consecutive', default['max_consecutive_missing'])
        
        # 处理异常值检测配置
        if 'outlier_detection' in loaded:
            outlier_config = loaded['outlier_detection']
            merged['outlier_method'] = outlier_config.get('method', default['outlier_method'])
            merged['iqr_threshold'] = outlier_config.get('iqr_threshold', default['iqr_threshold'])
            
            # 合并物理范围
            if 'physical_ranges' in outlier_config:
                merged['physical_ranges'] = outlier_config['physical_ranges']
            
            # 合并错误标记
            if 'error_markers' in outlier_config:
                merged['error_markers'] = outlier_config['error_markers']
            
            # 更新嵌套配置
            merged['outlier_detection'] = outlier_config
        
        return merged

    def load_single_file(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        加载单个CSV文件

        Args:
            file_path: CSV文件路径

        Returns:
            pd.DataFrame: 加载的数据框

        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误或缺少必需列
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        logger.info(f"正在加载数据文件: {file_path}")

        try:
            suffix = file_path.suffix.lower()
            if suffix in {'.csv', '.txt'}:
                data = pd.read_csv(file_path)
            elif suffix in {'.xlsx', '.xls'}:
                data = pd.read_excel(file_path, sheet_name=0)
            else:
                raise ValueError(f"暂不支持的文件格式: {suffix}")

            # 统一列名空白
            data.columns = [re.sub(r"\s+", " ", str(col)).strip() for col in data.columns]

            # 统一索引与列名
            data = self._prepare_dataframe(data)
            data = self._apply_column_mapping(data)

            # 检查必需列
            missing_cols = [col for col in self.required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"数据缺少必需列: {missing_cols}")

            # 数据完整性检查
            self._check_data_integrity(data)

            logger.info(f"数据加载成功，形状: {data.shape}")
            return data

        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise

    # ------------------------------------------------------------------
    # 数据准备辅助函数
    # ------------------------------------------------------------------

    def _prepare_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """统一处理时间列并设置索引"""

        # 优先寻找标准列名
        time_candidates = [
            'timestamp',
            'time',
            'datetime',
            'Time(year-month-day h:m:s)',
            'Time (year-month-day h:m:s)',
        ]

        timestamp_col = None
        for col in time_candidates:
            if col in data.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            # 兜底：查找包含 time 的列
            for col in data.columns:
                if re.search(r'time', col, re.IGNORECASE):
                    timestamp_col = col
                    break

        if timestamp_col is None:
            raise ValueError("未找到时间列，请确保原始数据包含时间字段")

        data = data.copy()
        data.rename(columns={timestamp_col: 'timestamp'}, inplace=True)

        # 解析索引
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)

        return data

    def _apply_column_mapping(self, data: pd.DataFrame) -> pd.DataFrame:
        """将原始列映射为项目标准列名，并处理单位"""

        column_mapping = {
            'P': 'power',
            'Power (MW)': 'power_mw',
            'Power (kW)': 'power',
            'power': 'power',
            'I': 'irradiance',
            'Global horizontal irradiance (W/m2)': 'irradiance',
            'Total solar irradiance (W/m2)': 'irradiance_total',
            'Direct normal irradiance (W/m2)': 'dni',
            'Diffuse horizontal irradiance (W/m2)': 'dhi',
            'Air temperature (°C)': 'temperature',
            'Temperature (°C)': 'temperature',
            'T': 'temperature',
            'Atmosphere (hpa)': 'pressure',
            'Atmospheric pressure (hPa)': 'pressure',
            'Atmospheric pressure (kPa)': 'pressure_kpa',
            'Pre': 'pressure',
            'Relative humidity (%)': 'humidity',
            'Humidity (%)': 'humidity',
            'Hum': 'humidity',
        }

        renamed = {}
        for col in data.columns:
            key = column_mapping.get(col)
            if key is None and isinstance(col, str):
                key = column_mapping.get(col.strip())
                if key is None:
                    key = column_mapping.get(col.lower())
            renamed[col] = key if key else col

        data = data.rename(columns=renamed)

        numeric_candidates = [
            'power_mw',
            'power_kwh',
            'power',
            'irradiance_total',
            'dni',
            'dhi',
            'temperature',
            'pressure',
            'pressure_kpa',
            'humidity',
        ]
        for col in numeric_candidates:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        # 功率列统一为kW
        if 'power' not in data.columns and 'power_mw' in data.columns:
            data['power'] = data['power_mw'] * 1000.0
        elif 'power' in data.columns and data['power'].max() <= 1.5:
            data['power'] = data['power'] * 1000.0
        elif 'power_kwh' in data.columns:
            data['power'] = data['power_kwh']

        # 辐照度优先使用GHI
        if 'irradiance' not in data.columns:
            if 'irradiance_total' in data.columns:
                data['irradiance'] = data['irradiance_total']
            elif 'dni' in data.columns:
                data['irradiance'] = data['dni']
            elif 'dhi' in data.columns:
                data['irradiance'] = data['dhi']

        if 'pressure' not in data.columns and 'pressure_kpa' in data.columns:
            data['pressure'] = data['pressure_kpa'] * 10.0

        # 统一湿度范围到0-100
        if 'humidity' in data.columns:
            data['humidity'] = data['humidity'].clip(lower=0, upper=100)

        # 确保温度/气压/湿度存在
        for required in ['temperature', 'pressure', 'humidity']:
            if required not in data.columns:
                raise ValueError(f"数据缺少必需列: {required}")

        for col in ['power', 'irradiance', 'temperature', 'pressure', 'humidity']:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')

        return data

    def load_multi_station(
        self,
        station_files: Optional[List[str]] = None,
        merge_method: Optional[str] = None,
        selected_station: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        加载多个站点的数据

        Args:
            station_files: 站点文件列表，如果为None则加载所有CSV文件

        Returns:
            Dict[str, pd.DataFrame]: 站点名称到数据框的映射
        """
        station_data = {}

        # 如果未指定文件列表，扫描数据目录支持 CSV/Excel
        if station_files is None:
            csv_files = list(self.data_path.glob("*.csv"))
            excel_files = list(self.data_path.glob("*.xlsx")) + list(self.data_path.glob("*.xls"))
            station_files = csv_files + excel_files

            target_method = merge_method or self.config.get('merge', {}).get('method')
            target_station = selected_station or self.config.get('merge', {}).get('selected_station')

            if target_method == 'single' and station_files:
                if target_station:
                    matches = [fp for fp in station_files if fp.stem == target_station or fp.name == target_station]
                    if matches:
                        station_files = matches[:1]
                    else:
                        logger.warning(f"配置的站点 {target_station} 不存在，默认使用第一个站点")
                        station_files = station_files[:1]
                else:
                    station_files = station_files[:1]
        else:
            station_files = [self.data_path / f for f in station_files]

        logger.info(f"准备加载 {len(station_files)} 个站点数据")

        for file_path in station_files:
            station_name = file_path.stem  # 使用文件名作为站点名
            try:
                station_data[station_name] = self.load_single_file(file_path)
                logger.info(f"站点 {station_name} 数据加载成功")
            except Exception as e:
                logger.error(f"站点 {station_name} 数据加载失败: {e}")
                continue

        logger.info(f"成功加载 {len(station_data)} 个站点数据")
        return station_data

    def merge_stations(self, station_data: Dict[str, pd.DataFrame], method: str = 'concat') -> pd.DataFrame:
        """
        合并多个站点的数据

        Args:
            station_data: 站点数据字典
            method: 合并方法 ('concat': 垂直拼接, 'average': 平均值)

        Returns:
            pd.DataFrame: 合并后的数据
        """
        if not station_data:
            raise ValueError("没有可合并的站点数据")

        if method == 'concat':
            # 添加站点标识列并垂直拼接
            dfs = []
            for station_name, df in station_data.items():
                df_copy = df.copy()
                df_copy['station'] = station_name
                dfs.append(df_copy)
            merged = pd.concat(dfs, axis=0, sort=True)
            merged.sort_index(inplace=True)

        elif method == 'average':
            # 对相同时间点的数据取平均
            merged = pd.concat(station_data.values(), axis=1, keys=station_data.keys())
            merged = merged.groupby(level=1, axis=1).mean()

        else:
            raise ValueError(f"不支持的合并方法: {method}")

        logger.info(f"数据合并完成，方法: {method}, 最终形状: {merged.shape}")
        return merged

    def save_params(self, filepath: Union[str, Path]) -> None:
        """保存DataLoader配置参数"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'data_path': str(self.data_path),
            'station_column': getattr(self, 'station_column', None),
            'required_columns': self.required_columns,
            'frequency_minutes': getattr(self, 'frequency_minutes', self.frequency_minutes),
            'freq': self.freq,
            'config': self.config,
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(params, f, ensure_ascii=False, indent=2)

        logger.info(f"DataLoader 参数已保存: {filepath}")

    def load_params(self, filepath: Union[str, Path]) -> None:
        """加载DataLoader配置参数"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"DataLoader 参数文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            params = json.load(f)

        self.data_path = Path(params['data_path'])
        self.station_column = params.get('station_column', getattr(self, 'station_column', None))
        self.required_columns = params['required_columns']
        self.optional_columns = params.get('optional_columns', getattr(self, 'optional_columns', []))
        self.frequency_minutes = params.get('frequency_minutes', getattr(self, 'frequency_minutes', 15))
        self.freq = params['freq']
        self.config = params['config']

        logger.info(f"DataLoader 参数已加载: {filepath}")

    def load_processed_dataset(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """加载缓存的合并原始数据"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"缓存合并数据不存在: {filepath}")

        logger.info(f"从缓存载入合并数据: {filepath}")
        return pd.read_parquet(filepath)

    def _check_data_integrity(self, data: pd.DataFrame) -> Dict[str, any]:
        """
        检查数据完整性

        Args:
            data: 输入数据框

        Returns:
            Dict: 数据质量报告
        """
        report = {
            'total_rows': len(data),
            'time_range': (data.index.min(), data.index.max()),
            'missing_values': {},
            'outliers': {},
            'statistics': {}
        }

        # 缺失值统计
        for col in self.required_columns:
            if col in data.columns:
                missing_count = data[col].isna().sum()
                missing_ratio = missing_count / len(data)
                report['missing_values'][col] = {
                    'count': missing_count,
                    'ratio': missing_ratio
                }

                if missing_ratio > self.config['missing_threshold']:
                    logger.warning(f"列 {col} 缺失值比例过高: {missing_ratio:.2%}")

        # 异常值检测
        if self.config['outlier_detection']:
            for col in self.required_columns:
                if col in data.columns and data[col].notna().any():
                    outlier_info = self._detect_outliers(data[col].dropna())
                    report['outliers'][col] = outlier_info
                    if outlier_info['count'] > 0:
                        # 终端：简要信息
                        print(f"⚠ {col}: 检测到 {outlier_info['count']} 个异常值 (方法: {outlier_info['method']})")
                        # 日志文件：详细信息
                        logger.debug(
                            f"列 {col} 异常值详情 | "
                            f"方法={outlier_info['method']} | 阈值={outlier_info['thresholds']} | 示例={outlier_info['samples'][:5]}"
                        )

        # 基本统计信息
        report['statistics'] = data[self.required_columns].describe().to_dict()

        # 时间连续性检查
        inferred_freq = pd.infer_freq(data.index)
        if inferred_freq is None:
            logger.warning("无法推断时间频率，数据可能存在缺口")
        else:
            try:
                inferred_offset = to_offset(inferred_freq)
                target_offset = to_offset(self.freq)
            except ValueError:
                inferred_offset = None
                target_offset = None

            if inferred_offset is not None and target_offset is not None:
                if inferred_offset.nanos == target_offset.nanos:
                    logger.info(f"检测到数据频率: {inferred_offset.freqstr}")
                else:
                    logger.warning(f"时间频率不一致，期望: {target_offset.freqstr}, 实际: {inferred_offset.freqstr}")
            else:
                norm_inferred = inferred_freq.replace('minute', 'min').replace('Minute', 'min')
                norm_target = self.freq.replace('minute', 'min').replace('Minute', 'min')
                if norm_inferred.lower() == norm_target.lower():
                    logger.info(f"检测到数据频率: {inferred_freq}")
                else:
                    logger.warning(f"时间频率不一致，期望: {self.freq}, 实际: {inferred_freq}")

        # 检查时间间隔
        time_diffs = data.index.to_series().diff()
        try:
            expected_delta = to_offset(self.freq).delta
        except ValueError:
            expected_delta = pd.Timedelta(minutes=self.frequency_minutes)
        irregular_intervals = time_diffs[time_diffs != expected_delta]
        if len(irregular_intervals) > 1:  # 第一个差值为NaT，忽略
            logger.warning(f"发现 {len(irregular_intervals)-1} 个不规则时间间隔")

        return report

    def _detect_outliers(self, series: pd.Series, method: Optional[str] = None) -> Dict[str, Any]:
        """
        检测异常值
        
        支持三种检测方法：
        1. 'physical': 基于物理约束范围（推荐用于光伏数据）
        2. 'iqr': 基于四分位距的统计方法
        3. 'zscore': 基于Z-score的统计方法

        Args:
            series: 数据序列
            method: 检测方法 ('physical', 'iqr', 'zscore')

        Returns:
            Dict: 异常值检测结果，包含索引、数量、阈值等信息
        """
        if method is None:
            method = self.config['outlier_method']

        samples_preview: List[Dict[str, Any]] = []
        thresholds: Dict[str, Any] = {}
        column_name = series.name if hasattr(series, 'name') else 'unknown'

        # 方法1：基于物理约束的检测（领域知识驱动）
        if method == 'physical':
            # 先检测错误标记值
            error_markers = self.config.get('error_markers', [-99, -999, -9999])
            error_mask = series.isin(error_markers)
            
            # 检测物理范围外的值
            physical_ranges = self.config.get('physical_ranges', {})
            
            if column_name in physical_ranges:
                lower_bound, upper_bound = physical_ranges[column_name]
                range_mask = (series < lower_bound) | (series > upper_bound)
                outlier_mask = error_mask | range_mask
            else:
                # 如果没有定义物理范围，只检测错误标记
                outlier_mask = error_mask
                lower_bound, upper_bound = None, None
            
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index
            
            # 分类异常值
            error_marker_count = error_mask.sum()
            range_violation_count = range_mask.sum() if column_name in physical_ranges else 0
            
            if not outlier_values.empty:
                # 优先显示错误标记
                error_samples = series[error_mask].head(3) if error_marker_count > 0 else pd.Series()
                # 然后显示范围违规（取极值）
                range_samples = series[range_mask if column_name in physical_ranges else pd.Series(dtype=float).index]
                if not range_samples.empty:
                    low_samples = range_samples.nsmallest(min(2, len(range_samples)))
                    high_samples = range_samples.nlargest(min(2, len(range_samples)))
                    range_samples = pd.concat([low_samples, high_samples])
                
                all_samples = pd.concat([error_samples, range_samples]).drop_duplicates()
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(series.loc[idx]),
                        'reason': 'error_marker' if series.loc[idx] in error_markers else 'out_of_range'
                    }
                    for idx in all_samples.index[:5]
                ]
            
            thresholds = {
                'method': 'physical_constraint',
                'lower_bound': float(lower_bound) if lower_bound is not None else None,
                'upper_bound': float(upper_bound) if upper_bound is not None else None,
                'error_markers': error_markers,
                'error_marker_count': int(error_marker_count),
                'range_violation_count': int(range_violation_count),
            }
        
        # 方法2：IQR统计方法（保留用于非物理数据）
        elif method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            threshold = self.config['iqr_threshold']
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (series < lower_bound) | (series > upper_bound)
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index

            if not outlier_values.empty:
                low_indices = list(outlier_values.nsmallest(min(3, len(outlier_values))).index)
                high_indices = list(outlier_values.nlargest(min(3, len(outlier_values))).index)
                preview_indices = list(dict.fromkeys(low_indices + high_indices))
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(outlier_values.loc[idx]),
                    }
                    for idx in preview_indices
                ]

            thresholds = {
                'lower_bound': float(lower_bound) if np.isfinite(lower_bound) else None,
                'upper_bound': float(upper_bound) if np.isfinite(upper_bound) else None,
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'threshold': float(threshold),
            }

        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            outlier_mask = z_scores > 3
            outlier_values = series[outlier_mask]
            outlier_indices = outlier_values.index

            if not outlier_values.empty:
                abs_order = (outlier_values - series.mean()).abs().sort_values(ascending=False)
                preview_indices = list(abs_order.head(min(5, len(abs_order))).index)
                samples_preview = [
                    {
                        'timestamp': str(idx),
                        'value': float(outlier_values.loc[idx]),
                        'zscore': float(z_scores.loc[idx]),
                    }
                    for idx in preview_indices
                ]

            thresholds = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'zscore_threshold': 3.0,
            }

        else:
            raise ValueError(f"不支持的异常值检测方法: {method}")

        outlier_indices = outlier_indices if 'outlier_indices' in locals() else series.index[:0]
        indices_list = outlier_indices.tolist()
        extrema = {
            'min': float(outlier_values.min()) if 'outlier_values' in locals() and not outlier_values.empty else None,
            'max': float(outlier_values.max()) if 'outlier_values' in locals() and not outlier_values.empty else None,
        }

        return {
            'method': method,
            'indices': indices_list,
            'count': len(indices_list),
            'thresholds': thresholds,
            'samples': samples_preview,
            'extrema': extrema,
        }

    def handle_missing_values(self, data: pd.DataFrame, method: Optional[str] = None) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            data: 输入数据
            method: 处理方法 ('linear', 'forward', 'backward', 'drop')

        Returns:
            pd.DataFrame: 处理后的数据
        """
        if method is None:
            method = self.config['interpolation_method']

        data_processed = data.copy()

        if method == 'linear':
            # 线性插值
            data_processed = data_processed.interpolate(method='linear', limit=self.config['max_consecutive_missing'])
        elif method == 'forward':
            # 前向填充
            data_processed = data_processed.fillna(method='ffill', limit=self.config['max_consecutive_missing'])
        elif method == 'backward':
            # 后向填充
            data_processed = data_processed.fillna(method='bfill', limit=self.config['max_consecutive_missing'])
        elif method == 'drop':
            # 删除缺失值
            data_processed = data_processed.dropna()
        else:
            raise ValueError(f"不支持的缺失值处理方法: {method}")

        # 检查是否还有缺失值
        remaining_missing = data_processed.isna().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"处理后仍有 {remaining_missing} 个缺失值")
            # 使用前向填充处理剩余缺失值
            data_processed = data_processed.fillna(method='ffill').fillna(method='bfill')

        logger.info(f"缺失值处理完成，方法: {method}")
        return data_processed

    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        验证数据质量是否满足要求

        Args:
            data: 待验证的数据

        Returns:
            Tuple[bool, Dict]: (是否通过验证, 质量报告)
        """
        quality_report = {
            'pass': True,
            'issues': [],
            'details': {},
        }

        # 检查功率范围
        if 'power' in data.columns:
            invalid_mask = (data['power'] < 0) | (data['power'] > 1000)
            if invalid_mask.any():
                quality_report['pass'] = False
                count = int(invalid_mask.sum())
                samples = data.loc[invalid_mask, 'power'].sort_values().head(5).tolist()
                quality_report['issues'].append(f"功率值超出合理范围: {count} 个")
                quality_report['details']['power'] = {
                    'count': count,
                    'samples': samples,
                    'min': float(data.loc[invalid_mask, 'power'].min()),
                    'max': float(data.loc[invalid_mask, 'power'].max()),
                }

        # 检查辐照度范围
        if 'irradiance' in data.columns:
            invalid_mask = (data['irradiance'] < 0) | (data['irradiance'] > 1500)
            if invalid_mask.any():
                quality_report['pass'] = False
                count = int(invalid_mask.sum())
                samples = data.loc[invalid_mask, 'irradiance'].sort_values().head(5).tolist()
                quality_report['issues'].append(f"辐照度超出合理范围: {count} 个")
                quality_report['details']['irradiance'] = {
                    'count': count,
                    'samples': samples,
                    'min': float(data.loc[invalid_mask, 'irradiance'].min()),
                    'max': float(data.loc[invalid_mask, 'irradiance'].max()),
                }

        # 检查温度范围
        if 'temperature' in data.columns:
            invalid_mask = (data['temperature'] < -30) | (data['temperature'] > 60)
            if invalid_mask.any():
                quality_report['pass'] = False
                count = int(invalid_mask.sum())
                samples = data.loc[invalid_mask, 'temperature'].sort_values().head(5).tolist()
                quality_report['issues'].append(f"温度超出合理范围: {count} 个")
                quality_report['details']['temperature'] = {
                    'count': count,
                    'samples': samples,
                    'min': float(data.loc[invalid_mask, 'temperature'].min()),
                    'max': float(data.loc[invalid_mask, 'temperature'].max()),
                }

        # 检查湿度范围
        if 'humidity' in data.columns:
            invalid_mask = (data['humidity'] < 0) | (data['humidity'] > 100)
            if invalid_mask.any():
                quality_report['pass'] = False
                count = int(invalid_mask.sum())
                samples = data.loc[invalid_mask, 'humidity'].sort_values().head(5).tolist()
                quality_report['issues'].append(f"湿度超出合理范围: {count} 个")
                quality_report['details']['humidity'] = {
                    'count': count,
                    'samples': samples,
                    'min': float(data.loc[invalid_mask, 'humidity'].min()),
                    'max': float(data.loc[invalid_mask, 'humidity'].max()),
                }

        # 检查气压范围
        if 'pressure' in data.columns:
            invalid_mask = (data['pressure'] < 800) | (data['pressure'] > 1100)
            if invalid_mask.any():
                quality_report['pass'] = False
                count = int(invalid_mask.sum())
                samples = data.loc[invalid_mask, 'pressure'].sort_values().head(5).tolist()
                quality_report['issues'].append(f"气压超出合理范围: {count} 个")
                quality_report['details']['pressure'] = {
                    'count': count,
                    'samples': samples,
                    'min': float(data.loc[invalid_mask, 'pressure'].min()),
                    'max': float(data.loc[invalid_mask, 'pressure'].max()),
                }

        if quality_report['pass']:
            logger.info("数据质量验证通过")
        else:
            logger.warning(f"数据质量问题: {quality_report['issues']}")

        return quality_report['pass'], quality_report

    # ============ GPU优化功能扩展 ============

    def create_sequence_data(self,
                            features: np.ndarray,
                            targets: np.ndarray,
                            sequence_length: int = 24,
                            weather_array: Optional[np.ndarray] = None
                            ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        创建序列数据用于LSTM训练

        Args:
            features: 特征数组 (n_timesteps, n_features)
            targets: 目标数组 (n_timesteps,)
            sequence_length: 序列长度
            weather_array: 天气标签数组（可选，与features长度一致）

        Returns:
            Tuple: (序列特征, 序列目标[, 序列天气标签])
        """
        if weather_array is not None and len(weather_array) != len(features):
            raise ValueError("weather_array 的长度必须与 features 一致")

        n_samples = len(features) - sequence_length + 1

        if n_samples <= 0:
            if weather_array is not None:
                return np.empty((0, sequence_length, features.shape[1])), np.empty((0, 1)), np.empty((0,), dtype=int)
            return np.empty((0, sequence_length, features.shape[1])), np.empty((0, 1))

        # 创建序列特征
        X = np.zeros((n_samples, sequence_length, features.shape[1]))
        y = np.zeros((n_samples, 1))
        weather_seq = None
        if weather_array is not None:
            weather_seq = np.zeros((n_samples,), dtype=int)

        for i in range(n_samples):
            X[i] = features[i:i+sequence_length]
            y[i] = targets[i+sequence_length-1]  # 预测最后一个时间步的目标
            if weather_seq is not None:
                weather_seq[i] = int(weather_array[i+sequence_length-1])

        if weather_seq is not None:
            return X, y, weather_seq
        return X, y

    def create_gpu_optimized_dataloader(self,
                                       data: pd.DataFrame,
                                       features_array: np.ndarray,
                                       targets_array: np.ndarray,
                                       batch_size: int = 64,
                                       sequence_length: int = 24,
                                       shuffle: bool = True,
                                       is_training: bool = True,
                                       weather_array: Optional[np.ndarray] = None) -> TorchDataLoader:
        """
        创建GPU优化的PyTorch DataLoader
        严格按照指导文件第735-746行配置

        Args:
            data: 原始数据框（用于验证）
            features_array: 特征数组
            targets_array: 目标数组
            batch_size: 批大小
            sequence_length: 序列长度
            shuffle: 是否打乱数据
            is_training: 是否为训练模式
            weather_array: 天气标签数组（可选）

        Returns:
            TorchDataLoader: GPU优化的数据加载器
        """
        # 创建序列数据
        seq_result = self.create_sequence_data(
            features_array,
            targets_array,
            sequence_length,
            weather_array=weather_array
        )

        if weather_array is not None:
            X, y, weather_seq = seq_result
        else:
            X, y = seq_result
            weather_seq = None

        # 创建PyTorch数据集
        dataset = DLFELSTMDataset(X, y, weather_seq)

        # 检测GPU可用性
        use_gpu = torch.cuda.is_available()

        # GPU优化配置（严格按照指导文件第735-746行）
        if use_gpu:
            if is_training:
                # 训练数据加载器配置
                dataloader = TorchDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=4,  # 多进程数据加载
                    pin_memory=True,  # 固定内存，加速GPU传输
                    persistent_workers=True,  # 保持worker进程
                    prefetch_factor=2,  # 预取批次数
                    drop_last=True  # 丢弃不完整批次（保持批大小一致）
                )
            else:
                # 验证/测试数据加载器配置
                dataloader = TorchDataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    persistent_workers=True,
                    prefetch_factor=2,
                    drop_last=False  # 验证时保留所有数据
                )
        else:
            # CPU配置
            dataloader = TorchDataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=2,
                pin_memory=False,
                drop_last=(True if is_training else False)
            )

        logger.info(f"创建{'GPU' if use_gpu else 'CPU'}优化DataLoader，批大小: {batch_size}")
        return dataloader

    def get_optimal_batch_size(self, gpu_id: int = 0) -> int:
        """
        根据GPU内存自动推荐最优批大小

        Args:
            gpu_id: GPU设备ID

        Returns:
            int: 推荐的批大小
        """
        if not torch.cuda.is_available():
            return 32

        # 获取GPU内存（GB）
        gpu_memory = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)

        # 根据内存大小推荐批大小
        if gpu_memory >= 16:  # 16GB+
            optimal_batch_size = 256
        elif gpu_memory >= 8:  # 8GB+
            optimal_batch_size = 128
        elif gpu_memory >= 4:  # 4GB+
            optimal_batch_size = 64
        else:
            optimal_batch_size = 32

        logger.info(f"GPU {gpu_id} 内存: {gpu_memory:.1f}GB, 推荐批大小: {optimal_batch_size}")
        return optimal_batch_size


class DLFELSTMDataset(Dataset):
    """
    DLFE-LSTM-WSI PyTorch数据集类
    用于GPU加速的数据加载
    """

    def __init__(self,
                 features: np.ndarray,
                 targets: np.ndarray,
                 weather: Optional[np.ndarray] = None,
                 transform=None):
        """
        初始化数据集

        Args:
            features: 特征数组 (n_samples, seq_len, n_features)
            targets: 目标数组 (n_samples, 1)
            weather: 天气标签 (n_samples,) 可选
            transform: 数据变换函数
        """
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
        if weather is not None:
            self.weather = torch.as_tensor(weather, dtype=torch.long)
        else:
            self.weather = None
        self.transform = transform

        # 验证数据维度
        assert len(self.features) == len(self.targets), "特征和目标数量不匹配"
        if self.weather is not None:
            assert len(self.weather) == len(self.targets), "天气标签数量与样本数量不匹配"

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """
        获取单个样本

        Returns:
            tuple: (features, targets)
        """
        features = self.features[idx]
        targets = self.targets[idx]

        if self.transform:
            features = self.transform(features)

        if self.weather is not None:
            weather = self.weather[idx]
            return features, targets, weather
        return features, targets