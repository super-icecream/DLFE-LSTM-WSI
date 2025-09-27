# -*- coding: utf-8 -*-
"""
数据处理模块测试
测试数据加载、划分、预处理和VMD分解功能
"""

import unittest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock

# 导入待测试模块（假设已实现）
# from src.data_processing import DataLoader, DataSplitter, Preprocessor, VMDDecomposer


class TestDataLoader(unittest.TestCase):
    """数据加载器测试类"""
    
    def setUp(self):
        """设置测试环境"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "test_data.csv"
        
        # 创建测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=288*7, freq='5T')  # 一周数据
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'power': np.random.rand(len(dates)) * 1000,
            'irradiance': np.random.rand(len(dates)) * 800,
            'temperature': np.random.rand(len(dates)) * 40,
            'pressure': np.random.rand(len(dates)) * 50 + 1000,
            'humidity': np.random.rand(len(dates)) * 100
        })
        self.test_data.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def test_load_csv_data(self):
        """测试CSV数据加载"""
        # 模拟DataLoader（实际实现后替换）
        class MockDataLoader:
            def load(self, path):
                return pd.read_csv(path)
        
        loader = MockDataLoader()
        data = loader.load(self.data_path)
        
        self.assertIsNotNone(data)
        self.assertEqual(len(data), 288*7)
        self.assertIn('power', data.columns)
        self.assertIn('timestamp', data.columns)
    
    def test_data_validation(self):
        """测试数据验证功能"""
        # 添加异常值
        self.test_data.loc[10, 'power'] = -100  # 负功率
        self.test_data.loc[20, 'humidity'] = 150  # 超范围湿度
        
        class MockDataLoader:
            def validate(self, data):
                errors = []
                if (data['power'] < 0).any():
                    errors.append("负功率值")
                if (data['humidity'] > 100).any():
                    errors.append("湿度超范围")
                return len(errors) == 0, errors
        
        loader = MockDataLoader()
        is_valid, errors = loader.validate(self.test_data)
        
        self.assertFalse(is_valid)
        self.assertEqual(len(errors), 2)
    
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        # 添加缺失值
        self.test_data.loc[5:10, 'temperature'] = np.nan
        
        class MockDataLoader:
            def handle_missing(self, data, method='interpolate'):
                if method == 'interpolate':
                    return data.interpolate()
                elif method == 'drop':
                    return data.dropna()
                else:
                    return data.fillna(method)
        
        loader = MockDataLoader()
        
        # 测试插值
        cleaned_data = loader.handle_missing(self.test_data, 'interpolate')
        self.assertFalse(cleaned_data['temperature'].isna().any())
        
        # 测试删除
        cleaned_data = loader.handle_missing(self.test_data, 'drop')
        self.assertEqual(len(cleaned_data), len(self.test_data) - 6)


class TestDataSplitter(unittest.TestCase):
    """数据划分器测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.n_samples = 1000
        self.data = pd.DataFrame({
            'feature1': np.random.randn(self.n_samples),
            'feature2': np.random.randn(self.n_samples),
            'target': np.random.randn(self.n_samples)
        })
    
    def test_temporal_split_ratio(self):
        """测试时序划分比例"""
        class MockDataSplitter:
            def split(self, data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
                n = len(data)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                
                train = data.iloc[:train_end]
                val = data.iloc[train_end:val_end]
                test = data.iloc[val_end:]
                
                return train, val, test
        
        splitter = MockDataSplitter()
        train, val, test = splitter.split(self.data)
        
        # 验证划分比例
        self.assertAlmostEqual(len(train) / self.n_samples, 0.7, places=2)
        self.assertAlmostEqual(len(val) / self.n_samples, 0.2, places=2)
        self.assertAlmostEqual(len(test) / self.n_samples, 0.1, places=2)
        
        # 验证无重叠
        total_length = len(train) + len(val) + len(test)
        self.assertEqual(total_length, self.n_samples)
    
    def test_no_data_leakage(self):
        """测试数据泄露防范"""
        class MockDataSplitter:
            def split(self, data):
                # 时间戳应该严格递增
                train_end = int(len(data) * 0.7)
                val_end = int(len(data) * 0.9)
                
                train = data.iloc[:train_end]
                val = data.iloc[train_end:val_end]
                test = data.iloc[val_end:]
                
                return train, val, test
        
        # 添加时间索引
        self.data['timestamp'] = pd.date_range('2024-01-01', periods=self.n_samples, freq='5T')
        
        splitter = MockDataSplitter()
        train, val, test = splitter.split(self.data)
        
        # 验证时间顺序
        self.assertTrue(train['timestamp'].max() < val['timestamp'].min())
        self.assertTrue(val['timestamp'].max() < test['timestamp'].min())


class TestPreprocessor(unittest.TestCase):
    """预处理器测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        self.data = np.random.randn(100, 5) * 100 + 500
        self.tensor_data = torch.tensor(self.data, dtype=torch.float32)
    
    def test_minmax_normalization(self):
        """测试MinMax归一化"""
        class MockPreprocessor:
            def __init__(self):
                self.min_val = None
                self.max_val = None
            
            def fit_transform(self, data):
                self.min_val = data.min(axis=0)
                self.max_val = data.max(axis=0)
                return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
            
            def transform(self, data):
                return (data - self.min_val) / (self.max_val - self.min_val + 1e-8)
            
            def inverse_transform(self, data):
                return data * (self.max_val - self.min_val + 1e-8) + self.min_val
        
        preprocessor = MockPreprocessor()
        
        # 测试fit_transform
        normalized = preprocessor.fit_transform(self.data)
        self.assertTrue((normalized >= 0).all())
        self.assertTrue((normalized <= 1).all())
        
        # 测试inverse_transform
        recovered = preprocessor.inverse_transform(normalized)
        np.testing.assert_array_almost_equal(recovered, self.data, decimal=5)
    
    def test_standardization(self):
        """测试标准化"""
        class MockPreprocessor:
            def __init__(self):
                self.mean = None
                self.std = None
            
            def fit_transform(self, data):
                self.mean = data.mean(axis=0)
                self.std = data.std(axis=0)
                return (data - self.mean) / (self.std + 1e-8)
        
        preprocessor = MockPreprocessor()
        standardized = preprocessor.fit_transform(self.data)
        
        # 验证标准化后的均值和标准差
        np.testing.assert_array_almost_equal(standardized.mean(axis=0), 0, decimal=5)
        np.testing.assert_array_almost_equal(standardized.std(axis=0), 1, decimal=5)
    
    def test_gpu_tensor_support(self):
        """测试GPU张量支持"""
        if torch.cuda.is_available():
            gpu_data = self.tensor_data.cuda()
            
            # 验证数据在GPU上
            self.assertTrue(gpu_data.is_cuda)
            
            # 模拟GPU处理
            normalized = (gpu_data - gpu_data.min()) / (gpu_data.max() - gpu_data.min())
            self.assertTrue(normalized.is_cuda)


class TestVMDDecomposer(unittest.TestCase):
    """VMD分解器测试类"""
    
    def setUp(self):
        """创建测试信号"""
        np.random.seed(42)
        t = np.linspace(0, 1, 500)
        
        # 创建合成信号（多个频率成分）
        self.signal = (
            np.sin(2 * np.pi * 5 * t) +    # 5Hz
            np.sin(2 * np.pi * 10 * t) +   # 10Hz
            np.sin(2 * np.pi * 20 * t) +   # 20Hz
            np.random.randn(len(t)) * 0.1  # 噪声
        )
    
    def test_vmd_decomposition(self):
        """测试VMD分解"""
        class MockVMDDecomposer:
            def __init__(self, n_modes=5, alpha=2000):
                self.n_modes = n_modes
                self.alpha = alpha
            
            def decompose(self, signal):
                # 简化的VMD模拟
                n = len(signal)
                modes = []
                for i in range(self.n_modes):
                    # 创建不同频率的模态
                    freq = (i + 1) * 5
                    mode = np.sin(2 * np.pi * freq * np.linspace(0, 1, n))
                    modes.append(mode * np.random.rand())
                
                return np.array(modes)
            
            def reconstruct(self, modes):
                return np.sum(modes, axis=0)
        
        decomposer = MockVMDDecomposer(n_modes=5)
        modes = decomposer.decompose(self.signal)
        
        # 验证分解结果
        self.assertEqual(modes.shape[0], 5)  # 5个模态
        self.assertEqual(modes.shape[1], len(self.signal))
        
        # 验证重构
        reconstructed = decomposer.reconstruct(modes)
        self.assertEqual(len(reconstructed), len(self.signal))
    
    def test_parameter_validation(self):
        """测试参数验证"""
        class MockVMDDecomposer:
            def __init__(self, n_modes=5, alpha=2000):
                if n_modes < 2 or n_modes > 10:
                    raise ValueError("n_modes必须在2-10之间")
                if alpha <= 0:
                    raise ValueError("alpha必须为正数")
                
                self.n_modes = n_modes
                self.alpha = alpha
        
        # 测试有效参数
        decomposer = MockVMDDecomposer(n_modes=5, alpha=2000)
        self.assertIsNotNone(decomposer)
        
        # 测试无效参数
        with self.assertRaises(ValueError):
            MockVMDDecomposer(n_modes=1)
        
        with self.assertRaises(ValueError):
            MockVMDDecomposer(alpha=-100)


# 测试套件
def suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加数据加载测试
    suite.addTest(unittest.makeSuite(TestDataLoader))
    
    # 添加数据划分测试
    suite.addTest(unittest.makeSuite(TestDataSplitter))
    
    # 添加预处理测试
    suite.addTest(unittest.makeSuite(TestPreprocessor))
    
    # 添加VMD分解测试
    suite.addTest(unittest.makeSuite(TestVMDDecomposer))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())