"""
DLFE-LSTM-WSI GPU优化模型使用示例
演示如何使用GPU优化的模型模块进行训练和预测
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src"))

# 导入GPU优化模块
from models import LSTMPredictor, ModelBuilder, MultiWeatherModel
from utils.gpu_dataloader import create_gpu_optimized_loaders

# 创建模拟数据集类
class MockDLFEDataset(torch.utils.data.Dataset):
    """模拟DLFE特征数据集，用于演示"""

    def __init__(self, num_samples=1000, seq_len=24, feature_dim=30):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        # 生成模拟的DLFE特征数据
        np.random.seed(42)
        self.features = np.random.randn(num_samples, seq_len, feature_dim).astype(np.float32)

        # 生成模拟的功率目标值（归一化到[0,1]）
        self.targets = np.random.beta(2, 2, size=(num_samples, 1)).astype(np.float32)

        # 生成模拟的天气类型 (0:晴天, 1:多云, 2:阴天)
        self.weather_types = np.random.randint(0, 3, size=num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.features[idx]),
            torch.from_numpy(self.targets[idx]),
            self.weather_types[idx]
        )


def demo_single_model():
    """演示单个LSTM模型的使用"""
    print("=== 单个LSTM模型演示 ===")

    # 创建GPU优化的LSTM模型
    model = LSTMPredictor(
        input_dim=30,
        hidden_dims=[100, 50],
        dropout_rates=[0.3, 0.2],
        use_cuda=True,
        use_mixed_precision=True
    )

    print(f"模型设备: {model.device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入数据
    batch_size = 16
    seq_len = 24
    input_data = torch.randn(batch_size, seq_len, 30)

    if torch.cuda.is_available():
        input_data = input_data.cuda()

    # 前向传播
    model.eval()
    with torch.no_grad():
        predictions, hidden_states = model(input_data)
        print(f"预测结果形状: {predictions.shape}")
        print(f"预测值范围: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")

    print("单个LSTM模型演示完成!\n")


def demo_model_builder():
    """演示模型构建器的使用"""
    print("=== 模型构建器演示 ===")

    # 创建模型构建器
    builder = ModelBuilder()

    # 显示GPU配置
    print("GPU配置信息:")
    for key, value in builder.gpu_config.items():
        print(f"  {key}: {value}")

    # 构建模型
    model = builder.build_model(use_data_parallel=False)

    # 创建优化器
    optimizer = builder.create_optimizer_gpu(model, lr=0.001)
    print(f"优化器类型: {type(optimizer).__name__}")

    # 估算内存使用
    memory_info = builder.estimate_memory_usage(model, batch_size=64)
    print("\n内存使用估算:")
    for key, value in memory_info.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    print("模型构建器演示完成!\n")


def demo_multi_weather_model():
    """演示多天气模型管理器的使用"""
    print("=== 多天气模型管理器演示 ===")

    # 创建模型构建器和多天气管理器
    builder = ModelBuilder()
    multi_model = MultiWeatherModel(builder, use_model_parallel=False)

    print(f"创建的天气模型: {list(multi_model.models.keys())}")

    # 准备测试数据
    batch_size = 8
    seq_len = 24
    features = torch.randn(batch_size, seq_len, 30)
    weather_prob = torch.softmax(torch.randn(batch_size, 3), dim=1)

    if torch.cuda.is_available():
        features = features.cuda()
        weather_prob = weather_prob.cuda()

    # 单模型预测
    with torch.no_grad():
        sunny_pred = multi_model.predict_gpu_optimized(
            features, weather_type=0, use_ensemble=False
        )
        print(f"晴天模型预测: {sunny_pred.shape}")

    # 集成预测
    with torch.no_grad():
        ensemble_pred = multi_model.predict_gpu_optimized(
            features, weather_prob=weather_prob, use_ensemble=True
        )
        print(f"集成预测: {ensemble_pred.shape}")

    print("多天气模型管理器演示完成!\n")


def demo_gpu_dataloader():
    """演示GPU优化的数据加载器"""
    print("=== GPU优化数据加载器演示 ===")

    # 创建模拟数据集
    train_dataset = MockDLFEDataset(num_samples=800)
    val_dataset = MockDLFEDataset(num_samples=200)

    # 创建GPU优化的数据加载器
    train_loader, val_loader, _ = create_gpu_optimized_loaders(
        train_dataset, val_dataset, batch_size=32
    )

    print(f"训练集批次数: {len(train_loader)}")
    print(f"验证集批次数: {len(val_loader)}")

    # 测试数据加载
    for batch_idx, (features, targets, weather_types) in enumerate(train_loader):
        print(f"批次 {batch_idx+1}:")
        print(f"  特征形状: {features.shape}")
        print(f"  目标形状: {targets.shape}")
        print(f"  天气类型: {weather_types[:5].tolist()}")  # 显示前5个
        if batch_idx >= 2:  # 只显示前3个批次
            break

    print("GPU优化数据加载器演示完成!\n")


def demo_training_loop():
    """演示简单的训练循环"""
    print("=== 训练循环演示 ===")

    # 创建数据
    train_dataset = MockDLFEDataset(num_samples=200)
    val_dataset = MockDLFEDataset(num_samples=50)

    train_loader, val_loader, _ = create_gpu_optimized_loaders(
        train_dataset, val_dataset, batch_size=16
    )

    # 创建模型和优化器
    builder = ModelBuilder()
    multi_model = MultiWeatherModel(builder)

    # 选择晴天模型进行演示训练
    model = multi_model.models['sunny']
    optimizer = builder.create_optimizer_gpu(model, lr=0.01)
    criterion = nn.MSELoss()

    if torch.cuda.is_available():
        criterion = criterion.cuda()

    # 简单训练循环（仅演示）
    model.train()
    for epoch in range(3):  # 只训练3个epoch用于演示
        train_loss = 0.0

        for batch_idx, (features, targets, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                features = features.cuda()
                targets = targets.cuda()

            # 前向传播
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions, _ = model(features)
                loss = criterion(predictions, targets)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx >= 5:  # 只训练几个批次用于演示
                break

        print(f"Epoch {epoch+1}, 平均损失: {train_loss/(batch_idx+1):.6f}")

    print("训练循环演示完成!\n")


def main():
    """主演示函数"""
    print("🚀 DLFE-LSTM-WSI GPU优化模型使用演示\n")

    try:
        demo_single_model()
        demo_model_builder()
        demo_multi_weather_model()
        demo_gpu_dataloader()
        demo_training_loop()

        print("🎉 所有演示完成！GPU优化模型模块工作正常。")

    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()