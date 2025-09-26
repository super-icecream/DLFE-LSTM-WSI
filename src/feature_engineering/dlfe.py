"""
动态局部特征嵌入模块 (DLFE)
功能：基于ADMM算法的流形学习，实现动态特征降维
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
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DLFE:
    """
    动态局部特征嵌入 (Dynamic Local Feature Embedding)

    基于流形学习的特征降维方法，通过ADMM算法优化保持局部邻域结构。
    将DPSR的输出降维到30维，同时保持数据的局部几何结构。

    Attributes:
        target_dim (int): 目标维度（默认30）
        sigma (float): 高斯核参数
        alpha (float): ADMM平衡参数（数据保真度）
        beta (float): ADMM平衡参数（稀疏性）
        max_iter (int): ADMM最大迭代次数
        tol (float): 收敛容差
    """

    def __init__(self,
                 target_dim: int = 30,
                 sigma: float = 1.0,
                 alpha: float = 2**-10,
                 beta: float = 0.1,
                 max_iter: int = 100,
                 tol: float = 1e-6):
        """
        初始化DLFE

        Args:
            target_dim: 目标降维维度（30维）
            sigma: 高斯核宽度参数
            alpha: ADMM数据保真度参数
            beta: ADMM稀疏性正则化参数
            max_iter: ADMM最大迭代次数
            tol: 收敛容差
        """
        self.target_dim = target_dim
        self.sigma = sigma
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol

        # 映射矩阵
        self.mapping_matrix = None  # A矩阵
        self.is_fitted = False

        # 优化历史
        self.optimization_history = {
            'objective': [],
            'constraint_violation': [],
            'iterations': 0
        }

        logger.info(f"DLFE初始化: 目标维度={target_dim}, σ={sigma}, α={alpha}, β={beta}")

    def build_similarity_matrix(self,
                              X: np.ndarray,
                              weights: Optional[np.ndarray] = None,
                              k_neighbors: Optional[int] = None) -> np.ndarray:
        """
        构建相似度矩阵Q

        使用高斯核计算样本间的相似度：
        Qij = exp(-dw(si-sj)²/σ²) if sj ∈ Si(si)
              0                    otherwise

        Args:
            X: 输入数据 (n_samples x n_features)
            weights: DPSR提供的特征权重（可选）
            k_neighbors: k近邻数量（如果None则使用全连接）

        Returns:
            相似度矩阵Q (n_samples x n_samples)
        """
        n_samples, n_features = X.shape

        # 如果提供了权重，进行加权
        if weights is not None:
            if len(weights.shape) == 1:
                # 一维权重，广播到所有特征
                X_weighted = X * np.sqrt(weights.reshape(1, -1))
            else:
                # 多维权重（每个样本不同权重）
                X_weighted = X * np.sqrt(weights)
        else:
            X_weighted = X

        # 计算欧氏距离矩阵
        # 使用广播计算所有点对的距离
        # ||xi - xj||² = ||xi||² + ||xj||² - 2*xi·xj
        X_norm = np.sum(X_weighted ** 2, axis=1, keepdims=True)
        distances_squared = X_norm + X_norm.T - 2 * np.dot(X_weighted, X_weighted.T)

        # 数值稳定性：确保距离非负
        distances_squared = np.maximum(distances_squared, 0)

        # 计算高斯相似度
        Q = np.exp(-distances_squared / (2 * self.sigma ** 2))

        # 如果指定k近邻，只保留k个最近邻
        if k_neighbors is not None and k_neighbors < n_samples - 1:
            for i in range(n_samples):
                # 找到第k+1近的距离作为阈值
                sorted_indices = np.argsort(distances_squared[i])
                # 保留自己和k个最近邻
                threshold_idx = min(k_neighbors + 1, n_samples)
                keep_indices = sorted_indices[:threshold_idx]

                # 创建掩码
                mask = np.zeros(n_samples, dtype=bool)
                mask[keep_indices] = True

                # 应用掩码
                Q[i, ~mask] = 0
                Q[~mask, i] = 0

        # 对角线设为0（不包括自相似）
        np.fill_diagonal(Q, 0)

        # 对称化（确保数值对称性）
        Q = (Q + Q.T) / 2

        logger.debug(f"相似度矩阵构建完成: 形状={Q.shape}, 非零元素比例={np.count_nonzero(Q) / Q.size:.2%}")

        return Q

    def construct_laplacian(self, Q: np.ndarray) -> np.ndarray:
        """
        构建图拉普拉斯矩阵

        L = D - Q
        其中D为度矩阵，Dii = ΣQij

        Args:
            Q: 相似度矩阵

        Returns:
            拉普拉斯矩阵L
        """
        # 计算度矩阵
        degrees = np.sum(Q, axis=1)
        D = np.diag(degrees)

        # 计算拉普拉斯矩阵
        L = D - Q

        # 归一化拉普拉斯（可选，提高数值稳定性）
        # L_norm = D^(-1/2) * L * D^(-1/2)
        # 避免除零
        degrees_sqrt_inv = np.zeros_like(degrees)
        non_zero_mask = degrees > 1e-10
        degrees_sqrt_inv[non_zero_mask] = 1.0 / np.sqrt(degrees[non_zero_mask])

        D_sqrt_inv = np.diag(degrees_sqrt_inv)
        L_normalized = D_sqrt_inv @ L @ D_sqrt_inv

        # 确保对称性
        L_normalized = (L_normalized + L_normalized.T) / 2

        return L_normalized

    def admm_optimization(self,
                         X: np.ndarray,
                         L: np.ndarray) -> np.ndarray:
        """
        ADMM算法求解映射矩阵

        优化目标：
        min tr(F^T*L*F) + α||ZA-F||²_F + β||A||_2,1
        s.t. F^T*F = I

        迭代步骤：
        1. F子问题：固定A，更新F
        2. A子问题：固定F，更新A
        3. 收敛判断：||F(k+1)-F(k)||_F < ε

        Args:
            X: 输入数据矩阵 Z (n_samples x n_features)
            L: 拉普拉斯矩阵 (n_samples x n_samples)

        Returns:
            映射矩阵A (n_features x target_dim)
        """
        n_samples, n_features = X.shape
        d = self.target_dim

        # 初始化
        A = np.zeros((n_features, d))  # 映射矩阵
        F = np.random.randn(n_samples, d)  # 低维表示
        F = F / np.linalg.norm(F, axis=0, keepdims=True)  # 列归一化

        # ADMM迭代
        for iter_idx in range(self.max_iter):
            F_old = F.copy()

            # ========== Step 1: 更新F（固定A） ==========
            # 优化目标：min tr(F^T*L*F) + α||X*A - F||²_F
            # 等价于：min tr(F^T*(L + αI)*F) - 2α*tr(F^T*X*A)

            # 构造矩阵M = L + αI
            M = L + self.alpha * np.eye(n_samples)

            # 目标：F^T*M*F - 2α*F^T*X*A
            # 使用特征分解求解
            if n_features < 500:  # 小规模问题，直接求解
                # 计算M的特征分解
                eigenvalues, eigenvectors = eigh(M)

                # 选择最小的d个特征值对应的特征向量
                idx = np.argsort(eigenvalues)[:d]
                F = eigenvectors[:, idx]
            else:  # 大规模问题，使用稀疏求解
                # 使用Lanczos方法求解最小特征值
                eigenvalues, eigenvectors = eigsh(M, k=d, which='SM')
                F = eigenvectors

            # 如果A不为零，需要考虑X*A项
            if np.any(A != 0):
                XA = X @ A
                # 修正F使其更接近XA
                F = F + self.alpha * XA / (1 + self.alpha)
                # 重新正交化
                U, S, Vt = np.linalg.svd(F, full_matrices=False)
                F = U @ Vt

            # ========== Step 2: 更新A（固定F） ==========
            # 优化目标：min α||X*A - F||²_F + β||A||_2,1
            # 这是一个L2,1正则化的最小二乘问题

            # 计算X^T*X
            XTX = X.T @ X
            XTF = X.T @ F

            # 使用软阈值算子求解
            for j in range(d):
                # 对每一列单独求解
                # 梯度：2α*X^T*(X*aj - fj)
                # L2,1正则化的近端算子

                # 计算残差
                if j == 0:
                    residual = XTF[:, j]
                else:
                    residual = XTF[:, j] - XTX @ A[:, j]

                # 软阈值
                norm_residual = np.linalg.norm(residual)
                if norm_residual > self.beta / (2 * self.alpha):
                    A[:, j] = (1 - self.beta / (2 * self.alpha * norm_residual)) * \
                             residual / (np.diagonal(XTX) + 1e-10)
                else:
                    A[:, j] = 0

            # ========== Step 3: 收敛判断 ==========
            F_change = np.linalg.norm(F - F_old, 'fro')
            relative_change = F_change / (np.linalg.norm(F_old, 'fro') + 1e-10)

            # 计算目标函数值
            obj_value = np.trace(F.T @ L @ F) + \
                       self.alpha * np.linalg.norm(X @ A - F, 'fro') ** 2 + \
                       self.beta * np.sum(np.linalg.norm(A, axis=0))

            self.optimization_history['objective'].append(obj_value)
            self.optimization_history['constraint_violation'].append(F_change)

            # 日志输出（每10次迭代）
            if (iter_idx + 1) % 10 == 0:
                logger.debug(f"ADMM迭代{iter_idx + 1}: 目标值={obj_value:.4f}, "
                           f"相对变化={relative_change:.6f}")

            # 收敛判断
            if relative_change < self.tol:
                logger.info(f"ADMM收敛于第{iter_idx + 1}次迭代")
                self.optimization_history['iterations'] = iter_idx + 1
                break

        else:
            logger.warning(f"ADMM达到最大迭代次数{self.max_iter}，可能未完全收敛")
            self.optimization_history['iterations'] = self.max_iter

        return A

    def fit(self,
            X_train: Union[np.ndarray, pd.DataFrame],
            dpsr_weights: Optional[Union[np.ndarray, Dict]] = None) -> 'DLFE':
        """
        从训练集学习映射矩阵A

        1. 构建相似度矩阵
        2. 计算拉普拉斯矩阵
        3. ADMM优化求解A
        4. 存储映射矩阵供transform使用

        Args:
            X_train: 训练数据 (n_samples x n_features)
            dpsr_weights: DPSR提供的动态权重（可选）

        Returns:
            self: 返回自身以支持链式调用
        """
        # 转换为numpy数组
        if isinstance(X_train, pd.DataFrame):
            X = X_train.values
        else:
            X = np.asarray(X_train)

        n_samples, n_features = X.shape
        logger.info(f"开始DLFE训练: {n_samples}样本, {n_features}特征 -> {self.target_dim}维")

        # 处理DPSR权重
        if dpsr_weights is not None:
            if isinstance(dpsr_weights, dict):
                # 如果是字典，计算平均权重
                weights_list = list(dpsr_weights.values())
                avg_weights = np.mean(weights_list, axis=0)
            else:
                avg_weights = dpsr_weights
        else:
            avg_weights = None

        # Step 1: 构建相似度矩阵
        logger.info("构建相似度矩阵...")
        Q = self.build_similarity_matrix(X, weights=avg_weights, k_neighbors=min(50, n_samples - 1))

        # Step 2: 构建拉普拉斯矩阵
        logger.info("构建拉普拉斯矩阵...")
        L = self.construct_laplacian(Q)

        # Step 3: ADMM优化
        logger.info("开始ADMM优化...")
        self.mapping_matrix = self.admm_optimization(X, L)

        self.is_fitted = True

        logger.info(f"DLFE训练完成，映射矩阵形状: {self.mapping_matrix.shape}")

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        应用学习到的映射矩阵

        F = X * A
        将输入降维到30维

        Args:
            X: 输入数据 (n_samples x n_features)

        Returns:
            降维后的特征 (n_samples x target_dim)
        """
        if not self.is_fitted:
            raise RuntimeError("请先调用fit方法学习映射矩阵")

        # 转换为numpy数组
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = np.asarray(X)

        # 检查特征维度
        if X_array.shape[1] != self.mapping_matrix.shape[0]:
            raise ValueError(f"特征维度不匹配: 输入{X_array.shape[1]}维, "
                           f"期望{self.mapping_matrix.shape[0]}维")

        # 应用映射
        F = X_array @ self.mapping_matrix

        # 归一化（可选）
        # F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-10)

        return F

    def fit_transform(self,
                     X_train: Union[np.ndarray, pd.DataFrame],
                     dpsr_weights: Optional[Union[np.ndarray, Dict]] = None) -> np.ndarray:
        """
        组合fit和transform

        Args:
            X_train: 训练数据
            dpsr_weights: DPSR权重

        Returns:
            降维后的特征 (n_samples x target_dim)
        """
        self.fit(X_train, dpsr_weights)
        return self.transform(X_train)

    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """
        保存映射矩阵

        Args:
            filepath: 保存路径
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练，无映射矩阵可保存")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mapping_data = {
            'mapping_matrix': self.mapping_matrix.tolist(),
            'target_dim': self.target_dim,
            'sigma': self.sigma,
            'alpha': self.alpha,
            'beta': self.beta,
            'optimization_history': self.optimization_history
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(mapping_data, f, indent=2)
        elif filepath.suffix == '.pkl':
            # pickle可以直接保存numpy数组
            mapping_data['mapping_matrix'] = self.mapping_matrix
            with open(filepath, 'wb') as f:
                pickle.dump(mapping_data, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"DLFE映射矩阵已保存: {filepath}")

    def load_mapping(self, filepath: Union[str, Path]) -> None:
        """
        加载映射矩阵

        Args:
            filepath: 映射文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"映射文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
            self.mapping_matrix = np.array(mapping_data['mapping_matrix'])
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                mapping_data = pickle.load(f)
            self.mapping_matrix = mapping_data['mapping_matrix']
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.target_dim = mapping_data['target_dim']
        self.sigma = mapping_data['sigma']
        self.alpha = mapping_data['alpha']
        self.beta = mapping_data['beta']
        self.optimization_history = mapping_data.get('optimization_history', {})
        self.is_fitted = True

        logger.info(f"DLFE映射矩阵已加载: {filepath}")

    def reconstruction_error(self,
                            X_original: np.ndarray,
                            X_reduced: np.ndarray) -> float:
        """
        计算重构误差

        Args:
            X_original: 原始高维数据
            X_reduced: 降维后的数据

        Returns:
            重构误差
        """
        if not self.is_fitted:
            raise RuntimeError("模型未训练")

        # 尝试重构（使用伪逆）
        A_pinv = np.linalg.pinv(self.mapping_matrix)
        X_reconstructed = X_reduced @ A_pinv.T

        # 计算误差
        error = np.mean((X_original - X_reconstructed) ** 2)

        return error


def test_dlfe():
    """测试DLFE模块"""
    # 创建模拟数据
    n_samples = 150
    n_features = 50  # 高维输入

    np.random.seed(42)

    # 生成具有流形结构的数据
    # 使用瑞士卷数据集的思想
    t = np.linspace(0, 4 * np.pi, n_samples)
    # 3D流形
    X_3d = np.column_stack([
        t * np.cos(t),
        t * np.sin(t),
        t
    ])

    # 扩展到高维（添加噪声维度）
    X_high = np.hstack([
        X_3d,
        0.1 * np.random.randn(n_samples, n_features - 3)
    ])

    # 创建模拟的DPSR权重
    dpsr_weights = np.random.rand(n_features)
    dpsr_weights = dpsr_weights / np.sum(dpsr_weights)

    # 初始化DLFE
    dlfe = DLFE(target_dim=30)

    # 训练
    print("开始DLFE训练...")
    dlfe.fit(X_high, dpsr_weights)

    # 转换
    X_embedded = dlfe.transform(X_high)

    print(f"输入形状: {X_high.shape}")
    print(f"输出形状: {X_embedded.shape}")
    print(f"映射矩阵形状: {dlfe.mapping_matrix.shape}")
    print(f"优化迭代次数: {dlfe.optimization_history['iterations']}")

    # 计算重构误差
    error = dlfe.reconstruction_error(X_high, X_embedded)
    print(f"重构误差: {error:.6f}")

    return dlfe, X_embedded


if __name__ == "__main__":
    # 运行测试
    test_dlfe()