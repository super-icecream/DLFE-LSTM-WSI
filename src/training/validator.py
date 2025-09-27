"""
模型验证器模块
功能：验证集性能评估，早停机制实现，最优模型选择
GPU优化：使用torch.no_grad()包装验证代码，异步数据传输
作者：DLFE-LSTM-WSI Team
日期：2025-09-27
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, List
import logging
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Validator:
    """
    模型验证器
    - 验证集评估
    - 早停策略
    - 性能跟踪
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min'):
        """
        参数：
        - patience: 早停耐心值
        - min_delta: 最小改善阈值
        - mode: 'min'或'max'
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

        # 验证历史记录
        self.val_history = {
            'loss': [],
            'metrics': [],
            'best_epoch': 0
        }

        # 最佳模型状态
        self.best_model_state = None
        self.best_metrics = None

        logger.info(f"验证器初始化完成，patience: {patience}, mode: {mode}")

    @torch.no_grad()
    def validate(self,
                model: nn.Module,
                val_loader: DataLoader,
                criterion: nn.Module,
                device: torch.device) -> Tuple[float, Dict]:
        """
        验证集评估（GPU加速）

        返回：
        - val_loss: 验证损失
        - metrics: 评估指标字典
        """
        model.eval()  # 设置为评估模式

        total_loss = 0
        predictions_list = []
        targets_list = []
        num_batches = 0

        # 遍历验证集
        for batch_idx, (features, targets) in enumerate(val_loader):
            # 异步数据传输到GPU
            if device.type == 'cuda':
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                features = features.to(device)
                targets = targets.to(device)

            # 前向传播（不需要梯度）
            predictions, _ = model(features)

            # 计算损失
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            # 收集预测和目标值用于计算指标
            predictions_list.append(predictions.cpu().numpy())
            targets_list.append(targets.cpu().numpy())
            num_batches += 1

        # 计算平均损失
        val_loss = total_loss / num_batches

        # 合并所有预测和目标值
        all_predictions = np.concatenate(predictions_list, axis=0)
        all_targets = np.concatenate(targets_list, axis=0)

        # 计算评估指标
        metrics = self._calculate_metrics(all_predictions, all_targets)
        metrics['loss'] = val_loss

        # 记录历史
        self.val_history['loss'].append(val_loss)
        self.val_history['metrics'].append(metrics)

        logger.info(f"验证完成 - Loss: {val_loss:.6f}, RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}")

        return val_loss, metrics

    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict:
        """
        计算评估指标

        参数：
        - predictions: 预测值
        - targets: 目标值

        返回：
        - metrics: 包含RMSE, MAE, MAPE等指标的字典
        """
        # 确保形状一致
        predictions = predictions.flatten()
        targets = targets.flatten()

        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((predictions - targets) ** 2))

        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(predictions - targets))

        # MAPE (Mean Absolute Percentage Error)
        # 避免除零错误
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((targets[non_zero_mask] - predictions[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = 0

        # R² (Coefficient of Determination)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # 添加小值避免除零

        # 最大误差
        max_error = np.max(np.abs(predictions - targets))

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'r2': float(r2),
            'max_error': float(max_error)
        }

        return metrics

    def check_early_stopping(self, val_loss: float) -> bool:
        """
        检查是否触发早停

        返回：
        - True: 应该停止训练
        - False: 继续训练
        """
        score = -val_loss if self.mode == 'min' else val_loss

        if self.best_score is None:
            # 第一次验证
            self.best_score = score
            self.counter = 0
            logger.info(f"初始验证分数: {val_loss:.6f}")
        elif score < self.best_score + self.min_delta:
            # 没有改善
            self.counter += 1
            logger.info(f"早停计数器: {self.counter}/{self.patience} (最佳: {-self.best_score if self.mode == 'min' else self.best_score:.6f})")

            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("触发早停！训练将停止。")
                return True
        else:
            # 有改善
            logger.info(f"验证分数改善: {-self.best_score if self.mode == 'min' else self.best_score:.6f} -> {val_loss:.6f}")
            self.best_score = score
            self.counter = 0

        return False

    def update_best_model(self, model: nn.Module, metrics: Dict):
        """更新最佳模型记录"""
        # 深拷贝模型状态
        self.best_model_state = {
            k: v.cpu().clone() for k, v in model.state_dict().items()
        }
        self.best_metrics = metrics.copy()
        self.val_history['best_epoch'] = len(self.val_history['loss']) - 1

        logger.info(f"更新最佳模型，指标: {metrics}")

    def get_best_model_state(self) -> Dict:
        """获取最佳模型状态"""
        return self.best_model_state

    def get_best_metrics(self) -> Dict:
        """获取最佳模型的评估指标"""
        return self.best_metrics

    def reset(self):
        """重置验证器状态"""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_history = {
            'loss': [],
            'metrics': [],
            'best_epoch': 0
        }
        self.best_model_state = None
        self.best_metrics = None

        logger.info("验证器状态已重置")

    def save_validation_history(self, save_path: str):
        """保存验证历史"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        history_data = {
            'val_history': self.val_history,
            'best_metrics': self.best_metrics,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat()
        }

        with open(save_path, 'w') as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"验证历史已保存到: {save_path}")

    def load_validation_history(self, load_path: str):
        """加载验证历史"""
        load_path = Path(load_path)

        if not load_path.exists():
            logger.warning(f"验证历史文件不存在: {load_path}")
            return False

        with open(load_path, 'r') as f:
            history_data = json.load(f)

        self.val_history = history_data['val_history']
        self.best_metrics = history_data.get('best_metrics')
        self.patience = history_data.get('patience', self.patience)
        self.min_delta = history_data.get('min_delta', self.min_delta)
        self.mode = history_data.get('mode', self.mode)

        logger.info(f"验证历史已从 {load_path} 加载")
        return True

    def plot_validation_curves(self) -> Dict:
        """生成验证曲线数据（供可视化使用）"""
        if not self.val_history['loss']:
            logger.warning("没有验证历史数据")
            return {}

        curves_data = {
            'epochs': list(range(1, len(self.val_history['loss']) + 1)),
            'loss': self.val_history['loss'],
            'metrics': {}
        }

        # 提取各个指标的历史
        if self.val_history['metrics']:
            metric_names = self.val_history['metrics'][0].keys()
            for metric_name in metric_names:
                if metric_name != 'loss':  # loss已经单独处理
                    curves_data['metrics'][metric_name] = [
                        m.get(metric_name, 0) for m in self.val_history['metrics']
                    ]

        curves_data['best_epoch'] = self.val_history['best_epoch'] + 1

        return curves_data


class MultiModelValidator:
    """多模型验证器（用于三个天气子模型）"""

    def __init__(self,
                 weather_types: List[str] = ['sunny', 'cloudy', 'overcast'],
                 patience: int = 10,
                 min_delta: float = 1e-4):
        """
        初始化多模型验证器

        参数：
        - weather_types: 天气类型列表
        - patience: 早停耐心值
        - min_delta: 最小改善阈值
        """
        self.weather_types = weather_types
        self.validators = {
            weather_type: Validator(patience, min_delta)
            for weather_type in weather_types
        }

        logger.info(f"多模型验证器初始化完成，模型数量: {len(weather_types)}")

    def validate_all(self,
                     models: Dict[str, nn.Module],
                     val_loaders: Dict[str, DataLoader],
                     criterion: nn.Module,
                     device: torch.device) -> Dict:
        """
        验证所有模型

        返回：
        - results: 包含所有模型验证结果的字典
        """
        results = {}

        for weather_type in self.weather_types:
            if weather_type in models and weather_type in val_loaders:
                val_loss, metrics = self.validators[weather_type].validate(
                    models[weather_type],
                    val_loaders[weather_type],
                    criterion,
                    device
                )

                results[weather_type] = {
                    'loss': val_loss,
                    'metrics': metrics,
                    'early_stop': self.validators[weather_type].check_early_stopping(val_loss)
                }

                # 如果是最佳结果，更新最佳模型
                if self.validators[weather_type].counter == 0:
                    self.validators[weather_type].update_best_model(
                        models[weather_type],
                        metrics
                    )

        return results

    def should_stop_training(self) -> bool:
        """检查是否所有模型都触发了早停"""
        return all(
            validator.early_stop
            for validator in self.validators.values()
        )

    def get_best_models(self) -> Dict:
        """获取所有最佳模型状态"""
        return {
            weather_type: validator.get_best_model_state()
            for weather_type, validator in self.validators.items()
        }

    def get_all_best_metrics(self) -> Dict:
        """获取所有模型的最佳指标"""
        return {
            weather_type: validator.get_best_metrics()
            for weather_type, validator in self.validators.items()
        }


if __name__ == "__main__":
    # 测试代码
    print("模型验证器测试")

    # 创建验证器
    validator = Validator(patience=5, min_delta=0.0001)

    # 模拟验证损失序列
    val_losses = [0.1, 0.09, 0.085, 0.084, 0.0839, 0.0838, 0.0838, 0.0838]

    print("\n测试早停机制:")
    for epoch, loss in enumerate(val_losses):
        should_stop = validator.check_early_stopping(loss)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, 早停 = {should_stop}")
        if should_stop:
            break

    # 测试多模型验证器
    print("\n测试多模型验证器:")
    multi_validator = MultiModelValidator()
    print(f"创建了 {len(multi_validator.validators)} 个子验证器")

    print("\n验证器模块测试完成！")