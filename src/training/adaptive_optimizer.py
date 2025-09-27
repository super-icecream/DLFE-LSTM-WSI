"""
自适应优化器模块
功能：基于误差反馈的在线参数调整，CI/WSI融合权重动态优化，天气分类边界自适应调整
GPU优化：使用GPU加速误差统计和参数更新，L-BFGS二阶优化
作者：DLFE-LSTM-WSI Team
日期：2025-09-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple, List
import numpy as np
import logging
from collections import deque
from pathlib import Path
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveOptimizer:
    """
    自适应优化器：基于误差反馈的在线参数调整

    核心功能：
    - 多级误差触发机制
    - 融合权重动态优化
    - 分类边界自适应调整
    - 模型参数微调
    """

    def __init__(self,
                 error_window: int = 100,
                 trigger_thresholds: Dict[str, float] = None,
                 device: str = 'cuda'):
        """
        参数：
        - error_window: 误差监控窗口大小
        - trigger_thresholds: 多级触发阈值
        - device: 计算设备
        """
        self.error_window = error_window
        self.trigger_thresholds = trigger_thresholds or {
            'level1': 0.1,   # 基础阈值
            'level2': 0.15,  # 波动异常
            'level3': 0.05,  # 趋势变化
            'level4': 5      # 持续性触发
        }
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # 误差历史记录（使用deque实现滑动窗口）
        self.error_history = deque(maxlen=error_window)
        self.error_stats_history = []

        # 融合权重（GPU张量）
        self.fusion_weights = torch.tensor([0.7, 0.3], device=self.device, requires_grad=True)

        # 天气分类阈值（初始值）
        self.weather_thresholds = {
            'ci_threshold_1': 0.2,    # 晴天-多云边界
            'ci_threshold_2': 0.6,    # 多云-阴天边界
            'wsi_threshold_1': 0.35,  # WSI晴天-多云边界
            'wsi_threshold_2': 0.75   # WSI多云-阴天边界
        }

        # 触发计数器
        self.trigger_counters = {
            'level1': 0,
            'level2': 0,
            'level3': 0,
            'level4': 0
        }

        # 参数更新历史
        self.update_history = {
            'fusion_weights': [],
            'weather_thresholds': [],
            'timestamps': []
        }

        # L-BFGS优化器（用于融合权重优化）
        self.weight_optimizer = optim.LBFGS(
            [self.fusion_weights],
            lr=0.01,
            max_iter=20,
            history_size=10
        )

        logger.info(f"自适应优化器初始化完成，设备: {self.device}")

    def monitor_errors(self, predictions: torch.Tensor,
                      targets: torch.Tensor) -> Dict:
        """
        误差监控与统计（GPU计算）

        返回误差统计指标：
        - mean_error: 平均误差
        - std_error: 误差标准差
        - error_trend: 误差变化趋势
        """
        # 确保在正确的设备上
        if not predictions.is_cuda and self.device.type == 'cuda':
            predictions = predictions.to(self.device)
        if not targets.is_cuda and self.device.type == 'cuda':
            targets = targets.to(self.device)

        # 计算误差
        errors = torch.abs(predictions - targets)

        # 添加到历史记录
        self.error_history.extend(errors.cpu().numpy().flatten().tolist())

        # GPU上计算统计量
        mean_error = errors.mean().item()
        std_error = errors.std().item()

        # 计算误差趋势（最近误差vs历史平均）
        if len(self.error_history) >= 20:
            recent_errors = list(self.error_history)[-20:]
            historical_errors = list(self.error_history)[:-20]

            recent_mean = np.mean(recent_errors)
            historical_mean = np.mean(historical_errors) if historical_errors else recent_mean

            error_trend = (recent_mean - historical_mean) / (historical_mean + 1e-10)
        else:
            error_trend = 0.0

        # 计算其他统计指标
        error_array = np.array(list(self.error_history))
        percentile_95 = np.percentile(error_array, 95) if len(error_array) > 0 else 0
        percentile_5 = np.percentile(error_array, 5) if len(error_array) > 0 else 0

        error_stats = {
            'mean_error': mean_error,
            'std_error': std_error,
            'error_trend': error_trend,
            'percentile_95': percentile_95,
            'percentile_5': percentile_5,
            'window_size': len(self.error_history)
        }

        # 保存统计历史
        self.error_stats_history.append(error_stats)

        logger.debug(f"误差统计: 均值={mean_error:.4f}, 标准差={std_error:.4f}, 趋势={error_trend:.4f}")

        return error_stats

    def check_adaptation_trigger(self, error_stats: Dict) -> int:
        """
        检查自适应触发条件

        返回触发级别：
        - 0: 不触发
        - 1-4: 对应不同触发级别
        """
        trigger_level = 0

        # Level 1: 平均误差超过基础阈值
        if error_stats['mean_error'] > self.trigger_thresholds['level1']:
            self.trigger_counters['level1'] += 1
            if self.trigger_counters['level1'] >= 3:  # 连续3次触发
                trigger_level = max(trigger_level, 1)
                logger.info(f"触发Level 1: 平均误差 {error_stats['mean_error']:.4f} > {self.trigger_thresholds['level1']}")
        else:
            self.trigger_counters['level1'] = 0

        # Level 2: 误差波动异常（标准差过大）
        if error_stats['std_error'] > self.trigger_thresholds['level2']:
            self.trigger_counters['level2'] += 1
            if self.trigger_counters['level2'] >= 2:  # 连续2次触发
                trigger_level = max(trigger_level, 2)
                logger.info(f"触发Level 2: 误差波动 {error_stats['std_error']:.4f} > {self.trigger_thresholds['level2']}")
        else:
            self.trigger_counters['level2'] = 0

        # Level 3: 误差趋势恶化
        if abs(error_stats['error_trend']) > self.trigger_thresholds['level3']:
            self.trigger_counters['level3'] += 1
            if self.trigger_counters['level3'] >= 2:
                trigger_level = max(trigger_level, 3)
                logger.info(f"触发Level 3: 误差趋势 {error_stats['error_trend']:.4f}")
        else:
            self.trigger_counters['level3'] = 0

        # Level 4: 持续性高误差
        if len(self.error_stats_history) >= self.trigger_thresholds['level4']:
            recent_stats = self.error_stats_history[-int(self.trigger_thresholds['level4']):]
            high_error_count = sum(1 for s in recent_stats if s['mean_error'] > self.trigger_thresholds['level1'] * 0.8)
            if high_error_count >= self.trigger_thresholds['level4'] * 0.8:
                trigger_level = max(trigger_level, 4)
                logger.info(f"触发Level 4: 持续性高误差")

        return trigger_level

    def update_fusion_weights(self,
                            error_gradient: torch.Tensor,
                            learning_rate: float = 0.01):
        """
        更新CI/WSI融合权重（GPU加速）

        使用梯度下降更新：
        W_CI(t+1) = W_CI(t) + η * ∇_CI E(t)
        W_WSI(t+1) = 1 - W_CI(t+1)
        """
        # 确保梯度在GPU上
        if not error_gradient.is_cuda and self.device.type == 'cuda':
            error_gradient = error_gradient.to(self.device)

        def closure():
            """L-BFGS闭包函数"""
            self.weight_optimizer.zero_grad()

            # 计算加权误差
            weighted_error = self.fusion_weights[0] * error_gradient[0] + \
                           self.fusion_weights[1] * error_gradient[1]

            # 添加正则化项（保持权重和为1）
            regularization = 100 * (self.fusion_weights.sum() - 1) ** 2

            loss = weighted_error + regularization
            loss.backward()

            return loss

        # 使用L-BFGS优化
        old_weights = self.fusion_weights.clone()
        self.weight_optimizer.step(closure)

        # 归一化权重（确保和为1）
        with torch.no_grad():
            self.fusion_weights.data = torch.softmax(self.fusion_weights, dim=0)

        # 限制权重范围
        with torch.no_grad():
            self.fusion_weights.data = torch.clamp(self.fusion_weights.data, min=0.2, max=0.8)
            # 重新归一化
            self.fusion_weights.data = self.fusion_weights.data / self.fusion_weights.data.sum()

        # 记录更新
        weight_change = torch.norm(self.fusion_weights - old_weights).item()
        if weight_change > 0.01:  # 只记录显著更新
            self.update_history['fusion_weights'].append(self.fusion_weights.cpu().tolist())
            self.update_history['timestamps'].append(datetime.now().isoformat())

            logger.info(f"融合权重更新: CI={self.fusion_weights[0]:.3f}, WSI={self.fusion_weights[1]:.3f}")

    def optimize_weather_thresholds(self,
                                   classification_errors: Dict) -> Dict:
        """
        优化天气分类阈值

        动态调整CI和WSI的分类边界
        """
        # 分析分类错误模式
        sunny_error = classification_errors.get('sunny', 0)
        cloudy_error = classification_errors.get('cloudy', 0)
        overcast_error = classification_errors.get('overcast', 0)

        # 自适应调整步长
        kappa = 0.005  # 小步长调整

        # 调整CI阈值
        if sunny_error > cloudy_error:
            # 晴天误差大，调整晴天-多云边界
            self.weather_thresholds['ci_threshold_1'] += kappa
        elif cloudy_error > sunny_error:
            self.weather_thresholds['ci_threshold_1'] -= kappa

        if cloudy_error > overcast_error:
            # 多云误差大，调整多云-阴天边界
            self.weather_thresholds['ci_threshold_2'] += kappa
        elif overcast_error > cloudy_error:
            self.weather_thresholds['ci_threshold_2'] -= kappa

        # 限制阈值范围
        self.weather_thresholds['ci_threshold_1'] = np.clip(
            self.weather_thresholds['ci_threshold_1'], 0.15, 0.25
        )
        self.weather_thresholds['ci_threshold_2'] = np.clip(
            self.weather_thresholds['ci_threshold_2'], 0.55, 0.65
        )

        # WSI阈值与CI阈值联动
        self.weather_thresholds['wsi_threshold_1'] = \
            self.weather_thresholds['ci_threshold_1'] * 1.75  # 映射函数f1
        self.weather_thresholds['wsi_threshold_2'] = \
            self.weather_thresholds['ci_threshold_2'] * 1.25  # 映射函数f2

        # 记录更新
        self.update_history['weather_thresholds'].append(self.weather_thresholds.copy())

        logger.info(f"天气阈值更新: CI=[{self.weather_thresholds['ci_threshold_1']:.3f}, "
                   f"{self.weather_thresholds['ci_threshold_2']:.3f}], "
                   f"WSI=[{self.weather_thresholds['wsi_threshold_1']:.3f}, "
                   f"{self.weather_thresholds['wsi_threshold_2']:.3f}]")

        return self.weather_thresholds

    def finetune_model(self,
                      model: nn.Module,
                      recent_data: Tuple[torch.Tensor, torch.Tensor],
                      learning_rate: float = 0.001,
                      epochs: int = 5):
        """
        在线模型微调（GPU加速）

        使用最近数据微调模型参数：
        - 冻结底层，只更新顶层
        - 使用L-BFGS二阶优化
        - 防止灾难性遗忘
        """
        features, targets = recent_data

        # 确保数据在正确的设备上
        if not features.is_cuda and self.device.type == 'cuda':
            features = features.to(self.device)
        if not targets.is_cuda and self.device.type == 'cuda':
            targets = targets.to(self.device)

        # 保存原始参数（用于回退）
        original_state = {k: v.clone() for k, v in model.state_dict().items()}

        # 冻结底层参数，只微调最后两层
        trainable_params = []
        for name, param in model.named_parameters():
            if 'fc' in name or 'output' in name or name.startswith('lstm.weight_hh_l1'):
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False

        # 创建L-BFGS优化器（二阶优化，收敛更快）
        finetune_optimizer = optim.LBFGS(
            trainable_params,
            lr=learning_rate,
            max_iter=20,
            history_size=10
        )

        # 损失函数
        criterion = nn.MSELoss()

        # 微调前的性能
        model.eval()
        with torch.no_grad():
            predictions, _ = model(features)
            rmse_before = torch.sqrt(criterion(predictions, targets)).item()

        # 微调
        model.train()
        for epoch in range(epochs):
            def closure():
                finetune_optimizer.zero_grad()
                predictions, _ = model(features)
                loss = criterion(predictions, targets)

                # 添加L2正则化防止过拟合
                l2_reg = sum(torch.norm(p) for p in trainable_params)
                loss = loss + 0.001 * l2_reg

                loss.backward()
                return loss

            loss = finetune_optimizer.step(closure)

            if epoch % 2 == 0:
                logger.debug(f"微调Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

        # 微调后的性能
        model.eval()
        with torch.no_grad():
            predictions, _ = model(features)
            rmse_after = torch.sqrt(criterion(predictions, targets)).item()

        # 性能提升判断
        improvement = (rmse_before - rmse_after) / rmse_before * 100

        if improvement < 5:  # 提升小于5%，回退到原参数
            model.load_state_dict(original_state)
            logger.info(f"微调提升不足({improvement:.1f}%)，回退到原参数")
        else:
            logger.info(f"微调成功！RMSE: {rmse_before:.4f} -> {rmse_after:.4f} (提升{improvement:.1f}%)")

        # 恢复所有参数的梯度计算
        for param in model.parameters():
            param.requires_grad = True

    def get_fusion_weights(self) -> np.ndarray:
        """获取当前融合权重"""
        return self.fusion_weights.cpu().detach().numpy()

    def get_weather_thresholds(self) -> Dict:
        """获取当前天气分类阈值"""
        return self.weather_thresholds.copy()

    def save_state(self, save_path: str):
        """保存优化器状态"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        state = {
            'fusion_weights': self.fusion_weights.cpu().detach().numpy().tolist(),
            'weather_thresholds': self.weather_thresholds,
            'error_history': list(self.error_history),
            'error_stats_history': self.error_stats_history[-100:],  # 只保存最近100条
            'update_history': self.update_history,
            'trigger_counters': self.trigger_counters,
            'timestamp': datetime.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"优化器状态已保存到: {save_path}")

    def load_state(self, load_path: str):
        """加载优化器状态"""
        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"状态文件不存在: {load_path}")
            return False

        with open(load_path, 'r') as f:
            state = json.load(f)

        self.fusion_weights = torch.tensor(
            state['fusion_weights'],
            device=self.device,
            requires_grad=True
        )
        self.weather_thresholds = state['weather_thresholds']
        self.error_history = deque(state['error_history'], maxlen=self.error_window)
        self.error_stats_history = state['error_stats_history']
        self.update_history = state['update_history']
        self.trigger_counters = state['trigger_counters']

        logger.info(f"优化器状态已从 {load_path} 加载")
        return True

    def reset(self):
        """重置优化器状态"""
        self.error_history.clear()
        self.error_stats_history.clear()
        self.fusion_weights = torch.tensor([0.7, 0.3], device=self.device, requires_grad=True)
        self.weather_thresholds = {
            'ci_threshold_1': 0.2,
            'ci_threshold_2': 0.6,
            'wsi_threshold_1': 0.35,
            'wsi_threshold_2': 0.75
        }
        self.trigger_counters = {k: 0 for k in self.trigger_counters}
        self.update_history = {
            'fusion_weights': [],
            'weather_thresholds': [],
            'timestamps': []
        }

        logger.info("优化器状态已重置")


if __name__ == "__main__":
    # 测试代码
    print("自适应优化器测试")

    # 创建优化器
    optimizer = AdaptiveOptimizer(error_window=50)

    print(f"设备: {optimizer.device}")
    print(f"初始融合权重: {optimizer.get_fusion_weights()}")
    print(f"初始天气阈值: {optimizer.get_weather_thresholds()}")

    # 模拟误差监控
    print("\n测试误差监控:")
    for i in range(10):
        predictions = torch.randn(16, 1) * 0.5 + 0.5
        targets = torch.randn(16, 1) * 0.5 + 0.5

        if optimizer.device.type == 'cuda':
            predictions = predictions.cuda()
            targets = targets.cuda()

        error_stats = optimizer.monitor_errors(predictions, targets)
        trigger_level = optimizer.check_adaptation_trigger(error_stats)

        if i % 3 == 0:
            print(f"步骤 {i+1}: 平均误差={error_stats['mean_error']:.4f}, 触发级别={trigger_level}")

    # 测试融合权重更新
    print("\n测试融合权重更新:")
    error_gradient = torch.tensor([0.1, 0.05], device=optimizer.device)
    optimizer.update_fusion_weights(error_gradient)
    print(f"更新后融合权重: {optimizer.get_fusion_weights()}")

    # 测试天气阈值优化
    print("\n测试天气阈值优化:")
    classification_errors = {'sunny': 0.12, 'cloudy': 0.08, 'overcast': 0.06}
    new_thresholds = optimizer.optimize_weather_thresholds(classification_errors)
    print(f"更新后天气阈值: {new_thresholds}")

    print("\n自适应优化器测试完成！")