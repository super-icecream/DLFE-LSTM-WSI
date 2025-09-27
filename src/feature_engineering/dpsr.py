"""
动态相空间重构模块 (DPSR)
功能：实现基于NCA的动态权重学习和相空间重构
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import json
import pickle
from scipy.special import softmax
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DPSR:
    """
    动态相空间重构 (Dynamic Phase Space Reconstruction)

    基于NCA（邻域成分分析）的动态权重学习，实现时变特征的相空间重构。
    将9维输入（5个IMF分量 + 4个气象特征）重构为30维动态特征向量。

    Attributes:
        embedding_dim (int): 嵌入维度（默认30）
        neighborhood_size (int): 局部邻域大小
        regularization (float): 正则化参数λ
        time_delay (int): 时间延迟τ
        max_iter (int): 最大迭代次数
    """

    def __init__(self,
                 embedding_dim: int = 30,
                 neighborhood_size: int = 50,
                 regularization: float = 0.01,
                 time_delay: int = 1,
                 max_iter: int = 100,
                 learning_rate: float = 0.01):
        """
        初始化DPSR

        Args:
            embedding_dim: 目标嵌入维度M=30
            neighborhood_size: 局部邻域大小L
            regularization: L2正则化参数λ
            time_delay: 时间延迟τ（默认1）
            max_iter: NCA优化最大迭代次数
            learning_rate: 初始学习率
        """
        self.embedding_dim = embedding_dim
        self.neighborhood_size = neighborhood_size
        self.regularization = regularization
        self.time_delay = time_delay
        self.max_iter = max_iter
        self.learning_rate = learning_rate

        # 动态权重存储
        self.weights = {}  # 每个时间点的权重
        self.global_weights = None  # 全局权重（训练后平均）
        self.is_fitted = False

        # 优化历史
        self.optimization_history = {
            'loss': [],
            'accuracy': []
        }

        logger.info(f"DPSR初始化: 嵌入维度={embedding_dim}, 邻域大小={neighborhood_size}")

    def construct_neighborhood(self,
                             data: np.ndarray,
                             time_idx: int,
                             window_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        构建局部时序邻域

        对于时刻i，构建局部邻域Si = {zi-L+1, ..., zi}，
        保持时序连续性，不跨数据集边界。

        Args:
            data: 输入数据 (n_samples x n_features)
            time_idx: 当前时间索引
            window_size: 邻域窗口大小，如果None则使用默认值

        Returns:
            Tuple[np.ndarray, np.ndarray]: (邻域数据, 对应索引)
        """
        if window_size is None:
            window_size = self.neighborhood_size

        n_samples = data.shape[0]

        # 确定邻域边界
        start_idx = max(0, time_idx - window_size + 1)
        end_idx = min(n_samples, time_idx + 1)

        # 提取邻域数据
        neighborhood_data = data[start_idx:end_idx]
        neighborhood_indices = np.arange(start_idx, end_idx)

        # 如果邻域太小，进行填充
        if len(neighborhood_data) < 3:
            # 至少需要3个样本
            if time_idx == 0:
                # 使用后续数据填充
                end_idx = min(3, n_samples)
                neighborhood_data = data[0:end_idx]
                neighborhood_indices = np.arange(0, end_idx)
            else:
                # 使用前面的数据填充
                start_idx = max(0, time_idx - 2)
                neighborhood_data = data[start_idx:time_idx + 1]
                neighborhood_indices = np.arange(start_idx, time_idx + 1)

        return neighborhood_data, neighborhood_indices

    def phase_space_embedding(self,
                             signal: np.ndarray,
                             embedding_dim: int,
                             time_delay: int = 1) -> np.ndarray:
        """
        经典相空间重构

        将一维时序重构为高维相空间：
        zi = [xi, xi-τ, xi-2τ, ..., xi-(m-1)τ]

        Args:
            signal: 一维时序信号
            embedding_dim: 嵌入维度m
            time_delay: 时间延迟τ

        Returns:
            重构的相空间矩阵
        """
        n = len(signal)
        m = embedding_dim

        # 计算重构后的样本数
        n_embedded = n - (m - 1) * time_delay

        if n_embedded <= 0:
            raise ValueError(f"信号太短，无法进行{m}维嵌入")

        # 构建相空间矩阵
        embedded = np.zeros((n_embedded, m))

        for i in range(n_embedded):
            for j in range(m):
                embedded[i, j] = signal[i + j * time_delay]

        return embedded

    def nca_optimization(self,
                        X: np.ndarray,
                        y: np.ndarray,
                        init_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        NCA（邻域成分分析）权重优化

        优化目标函数：
        f(w) = (1/n)Σl(yi, ŷi) + λΣwr²

        其中预测值：
        ŷi = Σpij*yj, pij = exp(-dw(si,sj))/Σexp(-dw(si,sk))

        加权距离：
        dw(si,sj) = Σwr²|sir - sjr|

        Args:
            X: 特征矩阵 (n_samples x n_features)
            y: 目标值 (n_samples,)
            init_weights: 初始权重

        Returns:
            优化后的权重向量
        """
        n_samples, n_features = X.shape

        # 初始化权重
        if init_weights is None:
            weights = np.ones(n_features) / n_features
        else:
            weights = init_weights.copy()

        # 标准化目标值（回归问题）
        y_mean = np.mean(y)
        y_std = np.std(y) + 1e-8
        y_normalized = (y - y_mean) / y_std

        # 定义目标函数
        def objective(w):
            """NCA目标函数"""
            # 重塑权重
            w = w.reshape(-1)
            w_squared = w ** 2

            # 计算加权距离矩阵
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j:
                        diff = np.abs(X[i] - X[j])
                        distances[i, j] = np.sum(w_squared * diff)

            # 计算概率矩阵（softmax）
            # 避免数值溢出
            max_dist = np.max(distances, axis=1, keepdims=True)
            exp_neg_dist = np.exp(-(distances - max_dist))

            # 对角线设为0（不包括自己）
            np.fill_diagonal(exp_neg_dist, 0)

            # 归一化得到概率
            row_sums = np.sum(exp_neg_dist, axis=1, keepdims=True) + 1e-10
            probabilities = exp_neg_dist / row_sums

            # Leave-one-out预测
            y_pred = np.dot(probabilities, y_normalized)

            # 计算损失
            prediction_error = np.mean((y_normalized - y_pred) ** 2)

            # 正则化项
            regularization_term = self.regularization * np.sum(w ** 2)

            total_loss = prediction_error + regularization_term

            return total_loss

        def gradient(w):
            """NCA梯度"""
            w = w.reshape(-1)
            w_squared = w ** 2
            grad = np.zeros_like(w)

            # 计算加权距离矩阵
            distances = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(n_samples):
                    if i != j:
                        diff = np.abs(X[i] - X[j])
                        distances[i, j] = np.sum(w_squared * diff)

            # 计算概率矩阵
            max_dist = np.max(distances, axis=1, keepdims=True)
            exp_neg_dist = np.exp(-(distances - max_dist))
            np.fill_diagonal(exp_neg_dist, 0)
            row_sums = np.sum(exp_neg_dist, axis=1, keepdims=True) + 1e-10
            probabilities = exp_neg_dist / row_sums

            # Leave-one-out预测
            y_pred = np.dot(probabilities, y_normalized)

            # 计算梯度
            for k in range(n_features):
                grad_k = 0
                for i in range(n_samples):
                    error_i = y_normalized[i] - y_pred[i]

                    for j in range(n_samples):
                        if i != j:
                            diff_ij_k = np.abs(X[i, k] - X[j, k])
                            grad_contrib = probabilities[i, j] * diff_ij_k * \
                                         (y_normalized[j] - y_pred[i])
                            grad_k += 2 * w[k] * error_i * grad_contrib

                grad[k] = -2 * grad_k / n_samples + 2 * self.regularization * w[k]

            return grad

        # 优化选项
        options = {
            'maxiter': self.max_iter,
            'gtol': 1e-5,
            'ftol': 1e-5
        }

        # 执行优化
        result = minimize(
            objective,
            weights,
            method='L-BFGS-B',
            jac=gradient,
            bounds=[(0, 1)] * n_features,  # 权重约束在[0,1]
            options=options
        )

        # 归一化权重
        optimized_weights = result.x
        optimized_weights = optimized_weights / (np.sum(optimized_weights) + 1e-10)

        # 记录优化历史
        self.optimization_history['loss'].append(result.fun)

        if result.success:
            logger.debug(f"NCA优化成功: 损失={result.fun:.4f}, 迭代={result.nit}")
        else:
            logger.warning(f"NCA优化未完全收敛: {result.message}")

        return optimized_weights

    def dynamic_reconstruction(self,
                             data: np.ndarray,
                             weights: np.ndarray) -> np.ndarray:
        """
        动态相空间重构

        使用NCA学习的权重进行动态重构：
        Vi = [wi1*xi1, wi2*xi-1,1, ..., wiM*xi-M+1,1,
              ..., wid*xid, ..., wiMd*xi-M+1,d]

        Args:
            data: 输入数据 (n_samples x n_features)
            weights: 特征权重

        Returns:
            重构的特征向量 (n_samples x embedding_dim)
        """
        n_samples, n_features = data.shape

        # 确定每个特征的嵌入维度
        dim_per_feature = self.embedding_dim // n_features
        extra_dims = self.embedding_dim % n_features

        # 构建重构矩阵
        reconstructed = []

        for feat_idx in range(n_features):
            # 当前特征的嵌入维度
            current_dim = dim_per_feature
            if feat_idx < extra_dims:
                current_dim += 1

            # 提取当前特征的时序
            feature_series = data[:, feat_idx]

            # 相空间嵌入
            if len(feature_series) >= current_dim:
                embedded = self.phase_space_embedding(
                    feature_series,
                    embedding_dim=current_dim,
                    time_delay=self.time_delay
                )
            else:
                # 信号太短，使用重复填充
                embedded = np.tile(feature_series.reshape(-1, 1),
                                  (1, current_dim))[:n_samples]

            # 应用动态权重
            weighted_embedded = embedded * weights[feat_idx]

            reconstructed.append(weighted_embedded)

        # 合并所有特征的嵌入
        if reconstructed:
            # 确保所有嵌入具有相同的样本数
            min_samples = min(r.shape[0] for r in reconstructed)
            reconstructed_aligned = [r[:min_samples] for r in reconstructed]

            # 水平拼接
            final_reconstruction = np.hstack(reconstructed_aligned)

            # 调整到目标维度
            if final_reconstruction.shape[1] > self.embedding_dim:
                final_reconstruction = final_reconstruction[:, :self.embedding_dim]
            elif final_reconstruction.shape[1] < self.embedding_dim:
                # 填充到目标维度
                padding = np.zeros((final_reconstruction.shape[0],
                                   self.embedding_dim - final_reconstruction.shape[1]))
                final_reconstruction = np.hstack([final_reconstruction, padding])
        else:
            final_reconstruction = np.zeros((n_samples, self.embedding_dim))

        return final_reconstruction

    def fit_transform(self,
                     data: Union[np.ndarray, pd.DataFrame],
                     labels: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """
        主处理流程：学习权重并进行动态重构

        1. 对每个时间窗口构建邻域
        2. NCA优化学习权重
        3. 应用权重进行动态重构
        4. 返回重构后的特征矩阵

        Args:
            data: 输入数据 (n_samples x n_features)
            labels: 目标标签（可选，用于监督学习）

        Returns:
            Tuple[重构特征矩阵, 权重字典]
        """
        # 转换为numpy数组
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        n_samples, n_features = data_array.shape
        logger.info(f"开始DPSR处理: {n_samples}样本, {n_features}特征")

        # 如果没有标签，使用下一时刻的第一个特征作为目标
        if labels is None:
            # 自监督：预测下一时刻
            labels = np.roll(data_array[:, 0], -1)
            labels[-1] = labels[-2]  # 处理最后一个样本

        # 初始化输出
        reconstructed_features = np.zeros((n_samples, self.embedding_dim))
        time_weights = {}

        # 分批处理，避免内存溢出
        batch_size = min(100, n_samples)
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = np.arange(start_idx, end_idx)

            logger.debug(f"处理批次 {batch_idx + 1}/{n_batches}")

            # 对批次中的每个时间点
            for local_idx, global_idx in enumerate(batch_indices):
                # 构建邻域
                neighborhood_data, neighbor_indices = self.construct_neighborhood(
                    data_array, global_idx
                )

                # 获取邻域标签
                neighborhood_labels = labels[neighbor_indices]

                # NCA优化学习权重
                if len(neighborhood_data) >= 3:  # 至少需要3个样本
                    try:
                        weights = self.nca_optimization(
                            neighborhood_data,
                            neighborhood_labels
                        )
                    except Exception as e:
                        logger.warning(f"时刻{global_idx}的NCA优化失败: {e}")
                        weights = np.ones(n_features) / n_features
                else:
                    # 邻域太小，使用均匀权重
                    weights = np.ones(n_features) / n_features

                # 存储权重
                time_weights[global_idx] = weights

                # 动态重构
                # 使用当前时刻的数据进行重构
                current_data = data_array[global_idx:global_idx + 1]
                reconstructed = self.dynamic_reconstruction(current_data, weights)

                if reconstructed.shape[0] > 0:
                    reconstructed_features[global_idx] = reconstructed[0]

        # 计算全局平均权重
        self.global_weights = np.mean(list(time_weights.values()), axis=0)
        self.weights = time_weights
        self.is_fitted = True

        logger.info(f"DPSR完成: 输出形状={reconstructed_features.shape}")

        return reconstructed_features, time_weights

    def transform(self,
                 data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        使用已学习的权重进行转换

        Args:
            data: 输入数据

        Returns:
            重构的特征矩阵
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit_transform学习权重")

        # 转换为numpy数组
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = np.asarray(data)

        n_samples = data_array.shape[0]

        # 使用全局权重进行重构
        reconstructed_features = self.dynamic_reconstruction(data_array, self.global_weights)

        # 确保输出维度正确
        if reconstructed_features.shape[0] < n_samples:
            # 填充不足的样本
            padding = np.zeros((n_samples - reconstructed_features.shape[0],
                               self.embedding_dim))
            reconstructed_features = np.vstack([reconstructed_features, padding])
        elif reconstructed_features.shape[0] > n_samples:
            # 截断多余的样本
            reconstructed_features = reconstructed_features[:n_samples]

        return reconstructed_features

    def save_weights(self, filepath: Union[str, Path]) -> None:
        """
        保存学习的权重

        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无权重可保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        splits = getattr(self, 'last_split_sizes', {})
        weights_data = {
            'global_weights': self.global_weights.tolist(),
            'time_weights': {int(k): v.tolist() for k, v in self.weights.items()},
            'embedding_dim': self.embedding_dim,
            'neighborhood_size': self.neighborhood_size,
            'regularization': self.regularization,
            'split_sizes': splits
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(weights_data, f, indent=2)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'wb') as f:
                pickle.dump(weights_data, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"DPSR权重已保存: {filepath}")

    def load_weights(self, filepath: Union[str, Path]) -> None:
        """
        加载权重

        Args:
            filepath: 权重文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"权重文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                weights_data = json.load(f)
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                weights_data = pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.global_weights = np.array(weights_data['global_weights'])
        self.weights = {int(k): np.array(v) for k, v in weights_data['time_weights'].items()}
        self.embedding_dim = weights_data['embedding_dim']
        self.neighborhood_size = weights_data['neighborhood_size']
        self.regularization = weights_data['regularization']
        self.is_fitted = True
        self.last_split_sizes = weights_data.get('split_sizes', {})

        logger.info(f"DPSR权重已加载: {filepath}")


def test_dpsr():
    """测试DPSR模块"""
    # 创建模拟数据（9维：5个IMF + 4个气象特征）
    n_samples = 200
    n_features = 9

    np.random.seed(42)

    # 生成具有时序相关性的数据
    data = np.zeros((n_samples, n_features))
    for i in range(n_features):
        # 生成不同频率的正弦波 + 噪声
        freq = 0.1 * (i + 1)
        data[:, i] = np.sin(2 * np.pi * freq * np.arange(n_samples) / n_samples)
        data[:, i] += 0.1 * np.random.randn(n_samples)

    # 创建简单的标签（基于第一个特征）
    labels = np.roll(data[:, 0], -1) + 0.01 * np.random.randn(n_samples)

    # 初始化DPSR
    dpsr = DPSR(embedding_dim=30, neighborhood_size=20)

    # 执行动态重构
    print("开始DPSR处理...")
    reconstructed, weights = dpsr.fit_transform(data, labels)

    print(f"输入形状: {data.shape}")
    print(f"输出形状: {reconstructed.shape}")
    print(f"学习的权重数: {len(weights)}")
    print(f"平均权重: {dpsr.global_weights}")

    # 测试transform方法
    test_data = data[:50]  # 使用前50个样本测试
    transformed = dpsr.transform(test_data)
    print(f"Transform测试: 输入{test_data.shape} -> 输出{transformed.shape}")

    return dpsr, reconstructed


if __name__ == "__main__":
    # 运行测试
    test_dpsr()