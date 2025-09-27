# -*- coding: utf-8 -*-
"""
测试数据夹具
提供标准化的测试数据生成器
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional


class TestDataGenerator:
    """测试数据生成器"""
    
    def __init__(self, seed: int = 42):
        """
        初始化测试数据生成器
        
        Args:
            seed: 随机种子
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    def generate_pv_data(self, 
                        n_days: int = 7,
                        sampling_rate: str = '5T',
                        noise_level: float = 0.1) -> pd.DataFrame:
        """
        生成模拟光伏数据
        
        Args:
            n_days: 数据天数
            sampling_rate: 采样率
            noise_level: 噪声水平
            
        Returns:
            包含光伏数据的DataFrame
        """
        # 生成时间戳
        start_date = datetime(2024, 6, 1)
        timestamps = pd.date_range(start=start_date, 
                                  end=start_date + timedelta(days=n_days),
                                  freq=sampling_rate)
        
        n_samples = len(timestamps)
        
        # 初始化数据
        data = pd.DataFrame({
            'timestamp': timestamps,
            'power': np.zeros(n_samples),
            'irradiance': np.zeros(n_samples),
            'temperature': np.zeros(n_samples),
            'pressure': np.zeros(n_samples),
            'humidity': np.zeros(n_samples)
        })
        
        # 生成数据
        for i, ts in enumerate(timestamps):
            hour = ts.hour + ts.minute / 60
            day_of_year = ts.timetuple().tm_yday
            
            # 日照模式
            if 6 <= hour <= 18:
                solar_elevation = np.sin((hour - 6) * np.pi / 12)
                
                # 季节调整
                seasonal_factor = 0.8 + 0.4 * np.sin((day_of_year - 80) * 2 * np.pi / 365)
                
                # 基础辐照度
                base_irradiance = 1000 * solar_elevation * seasonal_factor
                
                # 添加云层影响（随机）
                cloud_factor = 0.7 + 0.3 * np.random.rand()
                
                # 最终辐照度
                data.loc[i, 'irradiance'] = base_irradiance * cloud_factor * (1 + np.random.randn() * noise_level)
                
                # 功率（考虑温度影响）
                temp = 25 + 10 * solar_elevation + np.random.randn() * 5
                temp_efficiency = 1 - 0.004 * (temp - 25)  # 温度系数
                
                data.loc[i, 'power'] = data.loc[i, 'irradiance'] * 0.8 * temp_efficiency * (1 + np.random.randn() * noise_level)
                data.loc[i, 'temperature'] = temp
            else:
                # 夜间
                data.loc[i, 'power'] = 0
                data.loc[i, 'irradiance'] = 0
                data.loc[i, 'temperature'] = 15 + np.random.randn() * 3
            
            # 气压（缓慢变化）
            data.loc[i, 'pressure'] = 1013 + 20 * np.sin(i * 0.001) + np.random.randn() * 5
            
            # 湿度（反相关于温度）
            data.loc[i, 'humidity'] = 60 - data.loc[i, 'temperature'] * 0.5 + np.random.randn() * 10
            data.loc[i, 'humidity'] = np.clip(data.loc[i, 'humidity'], 10, 95)
        
        # 确保非负
        data['power'] = np.maximum(data['power'], 0)
        data['irradiance'] = np.maximum(data['irradiance'], 0)
        
        return data
    
    def generate_weather_labels(self, 
                               data: pd.DataFrame,
                               ci_thresholds: list = [0.2, 0.6],
                               wsi_thresholds: list = [0.3, 0.7]) -> pd.DataFrame:
        """
        为数据生成天气标签
        
        Args:
            data: 输入数据
            ci_thresholds: CI阈值
            wsi_thresholds: WSI阈值
            
        Returns:
            添加天气标签的DataFrame
        """
        data = data.copy()
        
        # 计算CI（简化版）
        max_irradiance = 1000
        data['ci'] = data['irradiance'] / max_irradiance
        
        # 计算WSI（简化版）
        pressure_norm = (1040 - data['pressure']) / 60
        humidity_norm = data['humidity'] / 100
        data['wsi'] = 0.4 * pressure_norm + 0.5 * humidity_norm + 0.1 * 0
        
        # 融合分类
        fusion_score = 0.7 * data['ci'] + 0.3 * data['wsi']
        
        # 天气标签
        data['weather'] = 'cloudy'  # 默认
        data.loc[fusion_score < 0.3, 'weather'] = 'overcast'
        data.loc[fusion_score > 0.6, 'weather'] = 'sunny'
        
        return data
    
    def generate_lstm_batch(self,
                           batch_size: int = 32,
                           seq_length: int = 10,
                           input_dim: int = 30,
                           output_dim: int = 1,
                           device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor]:
        """
        生成LSTM训练批次
        
        Args:
            batch_size: 批大小
            seq_length: 序列长度
            input_dim: 输入维度
            output_dim: 输出维度
            device: 设备
            
        Returns:
            (输入张量, 目标张量)
        """
        # 生成输入
        inputs = torch.randn(batch_size, seq_length, input_dim)
        
        # 生成目标（基于输入的简单函数）
        targets = inputs[:, -1, :output_dim] * 2 + torch.randn(batch_size, output_dim) * 0.1
        
        if device == 'cuda' and torch.cuda.is_available():
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        return inputs, targets
    
    def generate_vmd_test_signal(self,
                                n_samples: int = 1000,
                                n_components: int = 3) -> np.ndarray:
        """
        生成VMD测试信号
        
        Args:
            n_samples: 样本数
            n_components: 信号成分数
            
        Returns:
            合成信号
        """
        t = np.linspace(0, 1, n_samples)
        signal = np.zeros(n_samples)
        
        # 添加不同频率成分
        frequencies = [5, 10, 20, 40, 80]
        amplitudes = [1.0, 0.5, 0.3, 0.2, 0.1]
        
        for i in range(min(n_components, len(frequencies))):
            signal += amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t)
        
        # 添加噪声
        signal += np.random.randn(n_samples) * 0.05
        
        return signal
    
    def generate_train_val_test_split(self,
                                     data: pd.DataFrame,
                                     train_ratio: float = 0.7,
                                     val_ratio: float = 0.2,
                                     test_ratio: float = 0.1) -> Dict[str, pd.DataFrame]:
        """
        生成训练/验证/测试划分
        
        Args:
            data: 输入数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            划分后的数据字典
        """
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        splits = {
            'train': data.iloc[:train_end],
            'val': data.iloc[train_end:val_end],
            'test': data.iloc[val_end:]
        }
        
        return splits


# 预定义的测试数据集
def get_small_test_dataset():
    """获取小型测试数据集（1天）"""
    generator = TestDataGenerator(seed=42)
    data = generator.generate_pv_data(n_days=1)
    data = generator.generate_weather_labels(data)
    return generator.generate_train_val_test_split(data)


def get_medium_test_dataset():
    """获取中型测试数据集（7天）"""
    generator = TestDataGenerator(seed=42)
    data = generator.generate_pv_data(n_days=7)
    data = generator.generate_weather_labels(data)
    return generator.generate_train_val_test_split(data)


def get_large_test_dataset():
    """获取大型测试数据集（30天）"""
    generator = TestDataGenerator(seed=42)
    data = generator.generate_pv_data(n_days=30)
    data = generator.generate_weather_labels(data)
    return generator.generate_train_val_test_split(data)


# 测试夹具验证
if __name__ == '__main__':
    # 测试数据生成
    generator = TestDataGenerator()
    
    # 生成光伏数据
    pv_data = generator.generate_pv_data(n_days=3)
    print(f"生成光伏数据: {pv_data.shape}")
    print(pv_data.head())
    
    # 添加天气标签
    labeled_data = generator.generate_weather_labels(pv_data)
    print(f"\n天气分布:")
    print(labeled_data['weather'].value_counts())
    
    # 生成LSTM批次
    inputs, targets = generator.generate_lstm_batch()
    print(f"\nLSTM批次形状: 输入{inputs.shape}, 目标{targets.shape}")
    
    # 测试数据集划分
    splits = generator.generate_train_val_test_split(labeled_data)
    for split_name, split_data in splits.items():
        print(f"{split_name}: {len(split_data)} 样本")