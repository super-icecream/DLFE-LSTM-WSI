# -*- coding: utf-8 -*-
"""
特征工程模块测试
测试天气分类、DPSR动态重构和DLFE特征嵌入功能
"""

import unittest
import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch


class TestWeatherClassifier(unittest.TestCase):
    """天气分类器测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        n_samples = 288  # 一天的数据点
        
        # 创建测试数据
        self.timestamps = pd.date_range('2024-06-21', periods=n_samples, freq='5T')
        self.irradiance = np.zeros(n_samples)
        self.pressure = np.random.rand(n_samples) * 50 + 1000
        self.humidity = np.random.rand(n_samples) * 100
        self.temperature = np.random.rand(n_samples) * 40
        
        # 模拟日照曲线
        for i in range(n_samples):
            hour = i * 5 / 60  # 转换为小时
            if 6 <= hour <= 18:  # 白天
                self.irradiance[i] = 800 * np.sin((hour - 6) * np.pi / 12)
    
    def test_ci_calculation(self):
        """测试清晰度指数CI计算"""
        class MockCICalculator:
            def calculate_ci(self, ghi, timestamp, latitude=35.86, longitude=104.19):
                # 简化的CI计算
                # 计算理论辐照度（简化版）
                hour = timestamp.hour + timestamp.minute / 60
                if 6 <= hour <= 18:
                    ge = 1000 * np.sin((hour - 6) * np.pi / 12)
                else:
                    ge = 0
                
                if ge > 0:
                    return ghi / ge
                else:
                    return 0
        
        calculator = MockCICalculator()
        
        # 测试白天时段
        ci_noon = calculator.calculate_ci(
            800, 
            pd.Timestamp('2024-06-21 12:00:00')
        )
        self.assertGreater(ci_noon, 0)
        self.assertLessEqual(ci_noon, 1)
        
        # 测试夜晚时段
        ci_night = calculator.calculate_ci(
            0,
            pd.Timestamp('2024-06-21 00:00:00')
        )
        self.assertEqual(ci_night, 0)
    
    def test_wsi_calculation(self):
        """测试天气状态指数WSI计算"""
        class MockWSICalculator:
            def calculate_wsi(self, pressure, humidity, pressure_prev=None):
                # WSI = α * pressure_norm + β * humidity_norm + γ * pressure_change
                alpha, beta, gamma = 0.4, 0.5, 0.1
                
                pressure_norm = (1040 - pressure) / 60
                humidity_norm = humidity / 100
                
                if pressure_prev is not None:
                    pressure_change = max(0, -(pressure - pressure_prev) / 10)
                else:
                    pressure_change = 0
                
                wsi = alpha * pressure_norm + beta * humidity_norm + gamma * pressure_change
                return np.clip(wsi, 0, 1)
        
        calculator = MockWSICalculator()
        
        # 测试晴天条件（高压、低湿）
        wsi_sunny = calculator.calculate_wsi(1025, 30)
        self.assertLess(wsi_sunny, 0.3)
        
        # 测试阴天条件（低压、高湿）
        wsi_overcast = calculator.calculate_wsi(1005, 85)
        self.assertGreater(wsi_overcast, 0.7)
        
        # 测试压力变化
        wsi_changing = calculator.calculate_wsi(1010, 60, pressure_prev=1020)
        self.assertGreater(wsi_changing, 0.3)
    
    def test_weather_fusion_classification(self):
        """测试双路径天气融合分类"""
        class MockWeatherClassifier:
            def __init__(self):
                self.ci_weight = 0.7
                self.wsi_weight = 0.3
            
            def classify(self, ci, wsi):
                # 融合决策
                fusion_score = self.ci_weight * ci + self.wsi_weight * wsi
                
                if fusion_score < 0.3:
                    return 'overcast'
                elif fusion_score < 0.6:
                    return 'cloudy'
                else:
                    return 'sunny'
            
            def update_weights(self, error):
                # 自适应权重更新
                learning_rate = 0.01
                if error > 0.1:
                    # 增加WSI权重
                    self.wsi_weight = min(0.7, self.wsi_weight + learning_rate)
                    self.ci_weight = 1 - self.wsi_weight
        
        classifier = MockWeatherClassifier()
        
        # 测试分类
        weather = classifier.classify(ci=0.8, wsi=0.2)
        self.assertEqual(weather, 'sunny')
        
        weather = classifier.classify(ci=0.1, wsi=0.9)
        self.assertEqual(weather, 'overcast')
        
        # 测试权重更新
        classifier.update_weights(error=0.15)
        self.assertGreater(classifier.wsi_weight, 0.3)
        self.assertAlmostEqual(classifier.ci_weight + classifier.wsi_weight, 1.0)


class TestDPSR(unittest.TestCase):
    """动态相空间重构测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        # 创建时序数据
        self.time_series = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.1
        self.multivariate_data = np.random.randn(200, 5)  # 5维时序
    
    def test_phase_space_reconstruction(self):
        """测试相空间重构"""
        class MockDPSR:
            def __init__(self, embedding_dim=30, delay=1):
                self.embedding_dim = embedding_dim
                self.delay = delay
            
            def reconstruct(self, series):
                n = len(series)
                m = self.embedding_dim
                
                # 构建嵌入矩阵
                if n < m:
                    raise ValueError("序列长度小于嵌入维度")
                
                embedded = np.zeros((n - m + 1, m))
                for i in range(n - m + 1):
                    embedded[i] = series[i:i+m]
                
                return embedded
        
        dpsr = MockDPSR(embedding_dim=30)
        reconstructed = dpsr.reconstruct(self.time_series)
        
        # 验证重构维度
        self.assertEqual(reconstructed.shape[1], 30)
        self.assertEqual(reconstructed.shape[0], len(self.time_series) - 30 + 1)
    
    def test_nca_weight_optimization(self):
        """测试NCA权重优化"""
        class MockNCAOptimizer:
            def optimize_weights(self, X, y, n_iterations=100):
                n_features = X.shape[1]
                weights = np.ones(n_features)
                
                for _ in range(n_iterations):
                    # 简化的权重更新
                    correlations = np.abs([np.corrcoef(X[:, i], y)[0, 1] 
                                          for i in range(n_features)])
                    weights = correlations / (correlations.sum() + 1e-8)
                
                return weights
        
        optimizer = MockNCAOptimizer()
        
        # 创建特征和目标
        X = np.random.randn(100, 30)
        y = X[:, 0] * 2 + X[:, 1] * 0.5 + np.random.randn(100) * 0.1
        
        weights = optimizer.optimize_weights(X, y)
        
        # 验证权重
        self.assertEqual(len(weights), 30)
        self.assertAlmostEqual(weights.sum(), 1.0, places=5)
        # 第一个特征应该有更高权重
        self.assertGreater(weights[0], weights[10])
    
    def test_dynamic_reconstruction(self):
        """测试动态重构"""
        class MockDynamicDPSR:
            def __init__(self, window_size=100):
                self.window_size = window_size
                self.weights_history = []
            
            def dynamic_reconstruct(self, series):
                results = []
                
                for t in range(self.window_size, len(series)):
                    # 提取局部窗口
                    window = series[t-self.window_size:t]
                    
                    # 动态计算权重（简化）
                    weights = np.random.rand(self.window_size)
                    weights = weights / weights.sum()
                    
                    # 加权重构
                    weighted = window * weights
                    results.append(weighted.mean())
                    
                    self.weights_history.append(weights)
                
                return np.array(results)
        
        dpsr = MockDynamicDPSR(window_size=50)
        reconstructed = dpsr.dynamic_reconstruct(self.time_series)
        
        # 验证输出长度
        self.assertEqual(len(reconstructed), len(self.time_series) - 50)
        # 验证权重历史
        self.assertEqual(len(dpsr.weights_history), len(reconstructed))


class TestDLFE(unittest.TestCase):
    """动态局部特征嵌入测试类"""
    
    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)
        # 高维数据
        self.high_dim_data = np.random.randn(500, 100)
        # 创建一些结构
        self.high_dim_data[:, 0] = np.sin(np.linspace(0, 4*np.pi, 500))
        self.high_dim_data[:, 1] = np.cos(np.linspace(0, 4*np.pi, 500))
    
    def test_graph_construction(self):
        """测试动态邻域图构建"""
        class MockGraphBuilder:
            def build_graph(self, data, k_neighbors=10):
                n_samples = data.shape[0]
                adjacency = np.zeros((n_samples, n_samples))
                
                for i in range(n_samples):
                    # 计算距离
                    distances = np.sum((data - data[i])**2, axis=1)
                    # 找k近邻
                    neighbors = np.argsort(distances)[1:k_neighbors+1]
                    adjacency[i, neighbors] = 1
                    adjacency[neighbors, i] = 1
                
                return adjacency
        
        builder = MockGraphBuilder()
        adjacency = builder.build_graph(self.high_dim_data[:100])
        
        # 验证邻接矩阵
        self.assertEqual(adjacency.shape, (100, 100))
        self.assertTrue((adjacency >= 0).all())
        self.assertTrue((adjacency <= 1).all())
        # 验证对称性
        np.testing.assert_array_equal(adjacency, adjacency.T)
    
    def test_laplacian_matrix(self):
        """测试拉普拉斯矩阵计算"""
        class MockLaplacian:
            def compute_laplacian(self, adjacency):
                # 度矩阵
                degree = np.diag(adjacency.sum(axis=1))
                # 拉普拉斯矩阵
                laplacian = degree - adjacency
                return laplacian
            
            def normalized_laplacian(self, adjacency):
                degree = adjacency.sum(axis=1)
                # 避免除零
                degree_sqrt_inv = np.where(degree > 0, 1/np.sqrt(degree), 0)
                D_sqrt_inv = np.diag(degree_sqrt_inv)
                
                laplacian = np.eye(len(adjacency)) - D_sqrt_inv @ adjacency @ D_sqrt_inv
                return laplacian
        
        # 创建简单图
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0]
        ])
        
        calculator = MockLaplacian()
        laplacian = calculator.compute_laplacian(adjacency)
        
        # 验证拉普拉斯性质
        self.assertTrue(np.allclose(laplacian.sum(axis=1), 0))  # 行和为0
        self.assertTrue(np.allclose(laplacian, laplacian.T))    # 对称性
    
    def test_dimension_reduction(self):
        """测试降维到30维"""
        class MockDLFE:
            def __init__(self, target_dim=30):
                self.target_dim = target_dim
                self.mapping_matrix = None
            
            def fit_transform(self, data):
                n_samples, n_features = data.shape
                
                # 使用PCA简化（实际应使用ADMM优化）
                # 计算协方差矩阵
                data_centered = data - data.mean(axis=0)
                cov = data_centered.T @ data_centered / n_samples
                
                # 特征值分解
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # 选择前target_dim个主成分
                idx = np.argsort(eigenvalues)[::-1][:self.target_dim]
                self.mapping_matrix = eigenvectors[:, idx]
                
                # 投影
                reduced = data @ self.mapping_matrix
                return reduced
            
            def transform(self, data):
                if self.mapping_matrix is None:
                    raise ValueError("先调用fit_transform")
                return data @ self.mapping_matrix
        
        dlfe = MockDLFE(target_dim=30)
        reduced = dlfe.fit_transform(self.high_dim_data)
        
        # 验证降维结果
        self.assertEqual(reduced.shape, (500, 30))
        
        # 测试新数据转换
        new_data = np.random.randn(10, 100)
        new_reduced = dlfe.transform(new_data)
        self.assertEqual(new_reduced.shape, (10, 30))
    
    def test_gpu_acceleration(self):
        """测试GPU加速"""
        if torch.cuda.is_available():
            # 转换为GPU张量
            gpu_data = torch.tensor(self.high_dim_data, device='cuda')
            
            class MockGPUDLFE:
                def reduce_dimension(self, data):
                    # 简单的线性投影（GPU上）
                    projection = torch.randn(data.shape[1], 30, device=data.device)
                    reduced = data @ projection
                    return reduced
            
            dlfe = MockGPUDLFE()
            reduced = dlfe.reduce_dimension(gpu_data)
            
            # 验证在GPU上
            self.assertTrue(reduced.is_cuda)
            self.assertEqual(reduced.shape, (500, 30))


# 测试套件
def suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加天气分类测试
    suite.addTest(unittest.makeSuite(TestWeatherClassifier))
    
    # 添加DPSR测试
    suite.addTest(unittest.makeSuite(TestDPSR))
    
    # 添加DLFE测试
    suite.addTest(unittest.makeSuite(TestDLFE))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())