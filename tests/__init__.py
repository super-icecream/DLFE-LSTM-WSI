# -*- coding: utf-8 -*-
"""
DLFE-LSTM-WSI 测试模块
提供完整的单元测试框架和测试工具
"""

import unittest
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .test_data_processing import TestDataLoader, TestDataSplitter, TestPreprocessor, TestVMDDecomposer
from .test_feature_engineering import TestWeatherClassifier, TestDPSR, TestDLFE
from .test_models import TestLSTMModel, TestMultiWeatherModel, TestModelSaveLoad
from .fixtures.test_data import TestDataGenerator, get_small_test_dataset

__all__ = [
    # 数据处理测试
    'TestDataLoader',
    'TestDataSplitter', 
    'TestPreprocessor',
    'TestVMDDecomposer',
    
    # 特征工程测试
    'TestWeatherClassifier',
    'TestDPSR',
    'TestDLFE',
    
    # 模型测试
    'TestLSTMModel',
    'TestMultiWeatherModel',
    'TestModelSaveLoad',
    
    # 测试工具
    'TestDataGenerator',
    'get_small_test_dataset',
    
    # 测试套件
    'run_all_tests',
    'run_module_tests'
]


def run_all_tests(verbosity=2):
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试
    suite.addTests(loader.loadTestsFromModule(sys.modules[__name__]))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def run_module_tests(module_name: str, verbosity=2):
    """
    运行特定模块的测试
    
    Args:
        module_name: 模块名称 ('data_processing', 'feature_engineering', 'models')
        verbosity: 输出详细程度
    """
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    if module_name == 'data_processing':
        from . import test_data_processing
        suite.addTests(loader.loadTestsFromModule(test_data_processing))
    elif module_name == 'feature_engineering':
        from . import test_feature_engineering
        suite.addTests(loader.loadTestsFromModule(test_feature_engineering))
    elif module_name == 'models':
        from . import test_models
        suite.addTests(loader.loadTestsFromModule(test_models))
    else:
        raise ValueError(f"未知模块: {module_name}")
    
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result.wasSuccessful()


# 运行快速测试
def quick_test():
    """运行快速冒烟测试"""
    print("="*60)
    print("DLFE-LSTM-WSI 快速测试")
    print("="*60)
    
    # 测试数据生成
    print("\n1. 测试数据生成...")
    generator = TestDataGenerator()
    data = generator.generate_pv_data(n_days=1)
    assert len(data) > 0, "数据生成失败"
    print("✓ 通过")
    
    # 测试模型创建
    print("\n2. 测试模型架构...")
    import torch
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(30, 100, batch_first=True)
        
        def forward(self, x):
            out, _ = self.lstm(x)
            return out
    
    model = SimpleModel()
    test_input = torch.randn(16, 10, 30)
    output = model(test_input)
    assert output.shape == (16, 10, 100), "模型输出形状错误"
    print("✓ 通过")
    
    # 测试GPU可用性
    print("\n3. 测试GPU支持...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
        model = model.to(device)
        test_input = test_input.to(device)
        output = model(test_input)
        assert output.is_cuda, "GPU计算失败"
        print(f"✓ 通过 (GPU: {torch.cuda.get_device_name()})")
    else:
        print("⚠ 跳过 (GPU不可用)")
    
    print("\n" + "="*60)
    print("快速测试完成!")
    print("="*60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DLFE-LSTM-WSI 测试运行器')
    parser.add_argument('--module', type=str, help='指定测试模块')
    parser.add_argument('--quick', action='store_true', help='运行快速测试')
    parser.add_argument('-v', '--verbose', type=int, default=2, help='输出详细程度')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_test()
    elif args.module:
        success = run_module_tests(args.module, args.verbose)
        sys.exit(0 if success else 1)
    else:
        success = run_all_tests(args.verbose)
        sys.exit(0 if success else 1)