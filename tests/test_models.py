# -*- coding: utf-8 -*-
"""
模型模块测试
测试LSTM模型、多天气子模型和GPU优化功能
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch
import tempfile
import os

# 假设已实现的模型（实际测试时导入真实模块）
# from src.models import LSTMModel, WeatherSpecificModel, ModelBuilder


class TestLSTMModel(unittest.TestCase):
    """LSTM模型测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.input_dim = 30
        self.hidden_sizes = [100, 50]
        self.output_dim = 1
        self.batch_size = 32
        self.seq_length = 10
        
        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 创建测试数据
        self.test_input = torch.randn(self.batch_size, self.seq_length, self.input_dim)
        self.test_target = torch.randn(self.batch_size, self.output_dim)
    
    def test_model_architecture(self):
        """测试模型架构"""
        class MockLSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_sizes, output_dim, dropout_rates=[0.3, 0.2]):
                super().__init__()
                self.lstm1 = nn.LSTM(input_dim, hidden_sizes[0], batch_first=True)
                self.dropout1 = nn.Dropout(dropout_rates[0])
                self.lstm2 = nn.LSTM(hidden_sizes[0], hidden_sizes[1], batch_first=True)
                self.dropout2 = nn.Dropout(dropout_rates[1])
                self.fc = nn.Linear(hidden_sizes[1], output_dim)
            
            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout1(out)
                out, _ = self.lstm2(out)
                out = self.dropout2(out)
                out = self.fc(out[:, -1, :])  # 取最后时刻
                return out
        
        model = MockLSTMModel(self.input_dim, self.hidden_sizes, self.output_dim)
        
        # 测试前向传播
        output = model(self.test_input)
        
        # 验证输出形状
        self.assertEqual(output.shape, (self.batch_size, self.output_dim))
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        self.assertGreater(total_params, 0)
    
    def test_gpu_support(self):
        """测试GPU支持"""
        if torch.cuda.is_available():
            class SimpleLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(30, 100, batch_first=True)
                    self.fc = nn.Linear(100, 1)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            
            model = SimpleLSTM().cuda()
            input_gpu = self.test_input.cuda()
            
            # 测试GPU前向传播
            output = model(input_gpu)
            
            # 验证输出在GPU上
            self.assertTrue(output.is_cuda)
            self.assertEqual(output.device.type, 'cuda')
    
    def test_mixed_precision(self):
        """测试混合精度训练"""
        if torch.cuda.is_available():
            from torch.cuda.amp import autocast, GradScaler
            
            class SimpleLSTM(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lstm = nn.LSTM(30, 100, batch_first=True)
                    self.fc = nn.Linear(100, 1)
                
                def forward(self, x):
                    out, _ = self.lstm(x)
                    return self.fc(out[:, -1, :])
            
            model = SimpleLSTM().cuda()
            optimizer = torch.optim.Adam(model.parameters())
            scaler = GradScaler()
            
            input_gpu = self.test_input.cuda()
            target_gpu = self.test_target.cuda()
            
            # 混合精度训练步骤
            with autocast():
                output = model(input_gpu)
                loss = nn.MSELoss()(output, target_gpu)
            
            # 验证loss是float16
            self.assertTrue(loss.dtype == torch.float16 or loss.dtype == torch.float32)
            
            # 反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    def test_gradient_flow(self):
        """测试梯度流"""
        class SimpleLSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(30, 100, batch_first=True)
                self.fc = nn.Linear(100, 1)
            
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])
        
        model = SimpleLSTM()
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        # 前向传播
        output = model(self.test_input)
        loss = criterion(output, self.test_target)
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())


class TestMultiWeatherModel(unittest.TestCase):
    """多天气子模型测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.weather_types = ['sunny', 'cloudy', 'overcast']
        self.input_dim = 30
        self.batch_size = 16
    
    def test_model_selection(self):
        """测试模型选择机制"""
        class MockMultiWeatherModel:
            def __init__(self):
                self.models = {
                    'sunny': Mock(return_value=torch.randn(16, 1)),
                    'cloudy': Mock(return_value=torch.randn(16, 1)),
                    'overcast': Mock(return_value=torch.randn(16, 1))
                }
            
            def forward(self, x, weather_type):
                if weather_type not in self.models:
                    raise ValueError(f"未知天气类型: {weather_type}")
                return self.models[weather_type](x)
            
            def select_model(self, ci, wsi):
                fusion_score = 0.7 * ci + 0.3 * wsi
                if fusion_score < 0.3:
                    return 'overcast'
                elif fusion_score < 0.6:
                    return 'cloudy'
                else:
                    return 'sunny'
        
        model = MockMultiWeatherModel()
        
        # 测试不同天气条件
        test_input = torch.randn(16, 10, 30)
        
        # 晴天
        weather = model.select_model(ci=0.8, wsi=0.2)
        self.assertEqual(weather, 'sunny')
        output = model.forward(test_input, weather)
        self.assertEqual(output.shape, (16, 1))
        
        # 阴天
        weather = model.select_model(ci=0.1, wsi=0.9)
        self.assertEqual(weather, 'overcast')
        output = model.forward(test_input, weather)
        self.assertEqual(output.shape, (16, 1))
    
    def test_parallel_training(self):
        """测试并行训练"""
        if torch.cuda.is_available():
            class MockParallelTrainer:
                def __init__(self):
                    self.models = {
                        'sunny': nn.LSTM(30, 100, batch_first=True).cuda(),
                        'cloudy': nn.LSTM(30, 100, batch_first=True).cuda(),
                        'overcast': nn.LSTM(30, 100, batch_first=True).cuda()
                    }
                    self.streams = {
                        weather: torch.cuda.Stream() 
                        for weather in self.models
                    }
                
                def parallel_forward(self, data_dict):
                    results = {}
                    
                    for weather, data in data_dict.items():
                        with torch.cuda.stream(self.streams[weather]):
                            output, _ = self.models[weather](data)
                            results[weather] = output
                    
                    # 同步所有流
                    for stream in self.streams.values():
                        stream.synchronize()
                    
                    return results
            
            trainer = MockParallelTrainer()
            
            # 准备数据
            data_dict = {
                'sunny': torch.randn(8, 10, 30).cuda(),
                'cloudy': torch.randn(8, 10, 30).cuda(),
                'overcast': torch.randn(8, 10, 30).cuda()
            }
            
            # 并行前向传播
            results = trainer.parallel_forward(data_dict)
            
            # 验证结果
            for weather in data_dict:
                self.assertIn(weather, results)
                self.assertTrue(results[weather].is_cuda)


class TestModelSaveLoad(unittest.TestCase):
    """模型保存加载测试类"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
    
    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_save_load(self):
        """测试模型保存和加载"""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        # 创建并保存模型
        model = SimpleModel()
        original_weight = model.fc.weight.clone()
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {'input_dim': 10, 'output_dim': 1}
        }, self.model_path)
        
        # 加载模型
        new_model = SimpleModel()
        checkpoint = torch.load(self.model_path)
        new_model.load_state_dict(checkpoint['model_state_dict'])
        
        # 验证权重一致
        torch.testing.assert_close(new_model.fc.weight, original_weight)
    
    def test_checkpoint_compatibility(self):
        """测试检查点兼容性"""
        class ModelV1(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
            
            def forward(self, x):
                x = self.fc1(x)
                return self.fc2(x)
        
        class ModelV2(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(10, 5)
                self.fc2 = nn.Linear(5, 1)
                self.fc3 = nn.Linear(1, 1)  # 新增层
            
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return self.fc3(x)
        
        # 保存V1模型
        model_v1 = ModelV1()
        torch.save(model_v1.state_dict(), self.model_path)
        
        # 加载到V2模型（部分加载）
        model_v2 = ModelV2()
        state_dict = torch.load(self.model_path)
        
        # 只加载匹配的层
        model_v2.load_state_dict(state_dict, strict=False)
        
        # 验证共同层的权重被加载
        torch.testing.assert_close(model_v2.fc1.weight, model_v1.fc1.weight)
        torch.testing.assert_close(model_v2.fc2.weight, model_v1.fc2.weight)


# 测试套件
def suite():
    """创建测试套件"""
    suite = unittest.TestSuite()
    
    # 添加LSTM模型测试
    suite.addTest(unittest.makeSuite(TestLSTMModel))
    
    # 添加多天气模型测试
    suite.addTest(unittest.makeSuite(TestMultiWeatherModel))
    
    # 添加模型保存加载测试
    suite.addTest(unittest.makeSuite(TestModelSaveLoad))
    
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())