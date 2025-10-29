"""
VMD变分模态分解模块
功能：实现VMD算法，对光伏功率序列进行模态分解
重要：仅对功率P进行分解，其他特征直接保留
作者：DLFE-LSTM-WSI Team
日期：2025-09-26
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Union, Dict
from pathlib import Path
import logging
import json
import pickle
from scipy.signal import hilbert
from scipy.optimize import minimize

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class VMDDecomposer:
    """
    变分模态分解(VMD)处理器

    实现VMD算法，将光伏功率时序信号自适应地分解为若干个本征模态函数(IMF)。
    VMD通过求解约束优化问题，将信号分解为具有特定中心频率的模态分量。

    Attributes:
        n_modes (int): 模态数量（IMF个数）
        alpha (float): 惩罚参数（控制带宽）
        tau (float): 噪声容限（默认0.1，对偶软约束）
        DC (int): 是否包含直流分量（0或1）
        init (int): 初始化方式（0=全零，1=随机）
        tol (float): 收敛容差
    """

    def __init__(self, n_modes: int = 5, alpha: float = 2000, tau: float = 0.1,
                 DC: int = 0, init: int = 1, tol: float = 1e-6, max_iter: int = 500):
        """
        初始化VMD分解器

        Args:
            n_modes: 模态数量，默认5（根据光伏功率特性确定）
            alpha: 惩罚参数，控制数据保真度，默认2000
            tau: 拉格朗日乘子更新步长，默认0.1
            DC: 是否提取直流分量，默认0
            init: 初始化方式，0=全零，1=随机，默认1
            tol: 收敛容差，默认1e-6
            max_iter: 最大迭代次数，默认500
        """
        self.n_modes = n_modes
        self.alpha = alpha
        self.tau = tau
        self.DC = DC
        self.init = init
        self.tol = tol
        self.max_iter = max_iter

        # 存储分解参数
        self.decompose_params = {
            'n_modes': n_modes,
            'alpha': alpha,
            'tau': tau,
            'DC': DC,
            'init': init,
            'tol': tol,
            'max_iter': max_iter
        }

        # 存储分解结果
        self.u = None  # 模态分量
        self.omega = None  # 中心频率
        self.is_fitted = False

        logger.info(f"VMD分解器初始化: {n_modes}个模态, alpha={alpha}")

    def decompose(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        执行VMD分解

        对输入信号执行变分模态分解，将信号分解为指定数量的IMF分量。

        Args:
            signal: 一维输入信号（光伏功率序列）

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - u: 模态分量 (n_modes x signal_length)
                - omega: 各模态的中心频率 (n_modes,)

        Raises:
            ValueError: 如果输入信号不是一维数组
        """
        # 验证输入
        signal = np.asarray(signal).flatten()
        if signal.ndim != 1:
            raise ValueError(f"输入信号必须是一维数组，当前维度: {signal.ndim}")

        T = len(signal)
        logger.info(f"开始VMD分解，信号长度: {T}")

        # 时域转频域准备
        freqs = np.fft.fftfreq(T, d=1.0)
        half = T // 2
        pos_slice = slice(0, half + 1)
        neg_slice = slice(half + 1, None)
        freqs_pos = freqs[pos_slice]
        abs_freqs_pos = np.abs(freqs_pos)

        # FFT变换到频域
        f_hat = np.fft.fft(signal)
        f_hat_plus = f_hat.copy()
        f_hat_plus[neg_slice] = 0  # 仅保留非负频率
        if T % 2 == 0:
            f_hat_plus[half] = np.real(f_hat_plus[half])

        # 初始化
        u_hat_plus = np.zeros((self.n_modes, T), dtype=complex)
        omega = np.zeros((self.n_modes, self.max_iter))
        lambda_hat = np.zeros(T, dtype=complex)

        # 初始化中心频率
        if self.init == 0:
            # 均匀分布初始化
            for k in range(self.n_modes):
                omega[k, 0] = (0.5 / self.n_modes) * k
        elif self.init == 1:
            # 随机初始化
            omega[:, 0] = np.sort(np.random.rand(self.n_modes) * 0.5)
        else:
            # 均匀分布
            omega[:, 0] = np.linspace(0, 0.5, self.n_modes, endpoint=False)

        # 如果包含DC分量，将第一个模态设为DC
        if self.DC:
            omega[0, 0] = 0

        # 开始ADMM优化迭代
        logger.info("开始ADMM迭代优化...")
        n_iter = 0
        u_hat_plus_prev = u_hat_plus.copy()

        for n in range(self.max_iter - 1):
            # 更新模态u
            for k in range(self.n_modes):
                # 累加其他模态
                sum_uk = np.zeros(T, dtype=complex)
                if k == 0:
                    for i in range(1, self.n_modes):
                        sum_uk += u_hat_plus[i, :]
                elif k == self.n_modes - 1:
                    for i in range(0, self.n_modes - 1):
                        sum_uk += u_hat_plus[i, :]
                else:
                    for i in range(0, k):
                        sum_uk += u_hat_plus[i, :]
                    for i in range(k + 1, self.n_modes):
                        sum_uk += u_hat_plus[i, :]

                # Wiener滤波更新
                numerator = f_hat_plus - sum_uk - lambda_hat / 2
                denominator = 1 + self.alpha * (abs_freqs_pos - omega[k, n]) ** 2
                updated_spectrum = numerator[pos_slice] / denominator

                u_hat_plus[k, :] = 0
                u_hat_plus[k, pos_slice] = updated_spectrum
                if T % 2 == 0:
                    u_hat_plus[k, half] = np.real(u_hat_plus[k, half])

                # 处理DC分量
                if self.DC and k == 0:
                    dc_value = np.real(u_hat_plus[0, 0])
                    u_hat_plus[0, :] = 0
                    u_hat_plus[0, 0] = dc_value

            # 更新中心频率omega
            for k in range(self.n_modes):
                if self.DC and k == 0:
                    # DC模态频率固定为0
                    omega[k, n + 1] = 0
                else:
                    # 计算频谱重心作为新的中心频率
                    positive_spectrum = u_hat_plus[k, pos_slice]
                    energy_density = np.abs(positive_spectrum) ** 2
                    numerator_omega = np.dot(abs_freqs_pos, energy_density)
                    denominator_omega = np.sum(energy_density)

                    if denominator_omega > self.tol:
                        omega[k, n + 1] = numerator_omega / denominator_omega
                    else:
                        omega[k, n + 1] = omega[k, n]

            # 更新拉格朗日乘子
            sum_uk = np.sum(u_hat_plus, axis=0)
            lambda_hat += self.tau * (sum_uk - f_hat_plus)

            # 收敛性检查
            diff = np.sum(np.abs(u_hat_plus - u_hat_plus_prev) ** 2)
            norm = np.sum(np.abs(u_hat_plus_prev) ** 2)
            conv_check = diff / (norm + 1e-10)

            if conv_check < self.tol:
                logger.info(f"VMD收敛于第{n + 1}次迭代，误差: {conv_check:.2e}")
                n_iter = n + 1
                break

            u_hat_plus_prev = u_hat_plus.copy()
            n_iter = n + 1

        # 后处理
        N = n_iter
        omega = omega[:, :N]

        # 信号重构（逆FFT）
        u = np.zeros((self.n_modes, T))
        for k in range(self.n_modes):
            # 构造完整频谱（负频率部分）
            positive_freqs = u_hat_plus[k, pos_slice]
            u_hat_temp = np.zeros(T, dtype=complex)
            u_hat_temp[:pos_slice.stop] = positive_freqs
            u_hat_temp[0] = np.real(u_hat_temp[0])

            if T % 2 == 0:
                u_hat_temp[half] = np.real(u_hat_temp[half])
                if half > 1:
                    mirrored = np.conj(positive_freqs[1:half][::-1])
                    u_hat_temp[half + 1:] = mirrored
            else:
                if half >= 1:
                    mirrored = np.conj(positive_freqs[1:][::-1])
                    u_hat_temp[half + 1:] = mirrored

            # 逆FFT得到时域信号
            u[k, :] = np.real(np.fft.ifft(u_hat_temp))

        # 提取最终中心频率
        omega_final = omega[:, -1]

        # 存储结果
        self.u = u
        self.omega = omega_final
        self.is_fitted = True

        # ============================================================
        # VMD分解质量诊断
        # ============================================================
        reconstructed = np.sum(u, axis=0)
        if np.linalg.norm(signal) > 0:
            reconstruction_error = np.linalg.norm(signal - reconstructed) / np.linalg.norm(signal)
        else:
            reconstruction_error = 0.0

        correlation_matrix = np.corrcoef(u)
        off_diag_idx = np.triu_indices(self.n_modes, k=1)
        off_diag_values = correlation_matrix[off_diag_idx]
        avg_correlation = float(np.mean(np.abs(off_diag_values))) if off_diag_values.size else 0.0
        max_correlation = float(np.max(np.abs(off_diag_values))) if off_diag_values.size else 0.0

        sorted_omega = np.sort(omega_final)
        freq_gaps = np.diff(sorted_omega)
        min_freq_gap = float(np.min(freq_gaps)) if freq_gaps.size else 0.0
        avg_freq_gap = float(np.mean(freq_gaps)) if freq_gaps.size else 0.0

        energy = np.sum(u ** 2, axis=1)
        total_energy = float(np.sum(energy))
        if total_energy > 0:
            energy_ratio = energy / total_energy
        else:
            energy_ratio = np.zeros_like(energy)

        logger.info("=" * 60)
        logger.info("VMD分解质量诊断")
        logger.info("=" * 60)
        logger.info(f"重构误差: {reconstruction_error:.2e}")
        logger.info(f"模态正交性 - 平均相关系数: {avg_correlation:.3f}, 最大相关系数: {max_correlation:.3f}")
        logger.info(f"频率分离度 - 最小间隔: {min_freq_gap:.4f}, 平均间隔: {avg_freq_gap:.4f}")
        logger.info(f"频率分布: {sorted_omega}")
        logger.info("能量分布:")
        for idx, ratio in enumerate(energy_ratio):
            logger.info(f"  IMF{idx + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
        logger.info("=" * 60)
        # ============================================================

        logger.info(f"VMD分解完成，各模态中心频率: {omega_final}")

        return u, omega_final

    def reconstruct_features(self, imfs: np.ndarray,
                           other_features: pd.DataFrame) -> pd.DataFrame:
        """
        重组特征

        将VMD分解得到的IMF分量与其他原始特征(I, T, Pre, Hum)组合，
        形成新的特征矩阵用于后续处理。

        Args:
            imfs: IMF分量矩阵 (n_modes x time_steps)
            other_features: 其他特征数据框，包含I, T, Pre, Hum列

        Returns:
            pd.DataFrame: 重组后的特征数据框，
                         列为[IMF1, IMF2, ..., IMF5, I, T, Pre, Hum]
        """
        if imfs.shape[0] != self.n_modes:
            raise ValueError(f"IMF数量不匹配，期望{self.n_modes}，实际{imfs.shape[0]}")

        if imfs.shape[1] != len(other_features):
            raise ValueError(f"时间步长不匹配，IMF: {imfs.shape[1]}, "
                           f"其他特征: {len(other_features)}")

        # 创建IMF数据框
        imf_df = pd.DataFrame()
        for i in range(self.n_modes):
            imf_df[f'IMF{i+1}'] = imfs[i, :]

        # 设置与原始数据相同的索引
        imf_df.index = other_features.index

        # 合并IMF和其他特征
        feature_columns = ['irradiance', 'temperature', 'pressure', 'humidity']
        # 检查列名兼容性
        available_columns = []
        for col in feature_columns:
            if col in other_features.columns:
                available_columns.append(col)
            else:
                # 尝试缩写形式
                abbrev_map = {
                    'irradiance': ['I', 'irr'],
                    'temperature': ['T', 'temp'],
                    'pressure': ['Pre', 'P', 'pres'],
                    'humidity': ['Hum', 'H', 'hum']
                }
                for abbrev in abbrev_map.get(col, []):
                    if abbrev in other_features.columns:
                        available_columns.append(abbrev)
                        break

        # 组合特征
        combined_features = pd.concat([imf_df, other_features[available_columns]], axis=1)

        logger.info(f"特征重组完成，最终特征维度: {combined_features.shape}")
        logger.info(f"特征列: {list(combined_features.columns)}")

        return combined_features

    def process_dataset(self, data: pd.DataFrame,
                       power_column: str = 'power') -> pd.DataFrame:
        """
        处理整个数据集

        对数据集中的功率序列进行VMD分解，并与其他特征重组。

        Args:
            data: 输入数据集，必须包含功率列和其他特征列
            power_column: 功率列名称

        Returns:
            pd.DataFrame: 处理后的特征数据框
        """
        if power_column not in data.columns:
            # 尝试其他可能的列名
            possible_names = ['P', 'Power', 'power', 'pv_power']
            for name in possible_names:
                if name in data.columns:
                    power_column = name
                    break
            else:
                raise ValueError(f"找不到功率列，尝试过: {possible_names}")

        logger.info(f"处理数据集，功率列: {power_column}")

        # 提取功率序列
        power_signal = data[power_column].values

        # 执行VMD分解
        imfs, omega = self.decompose(power_signal)

        # 提取其他特征
        other_columns = [col for col in data.columns if col != power_column]
        other_features = data[other_columns]

        # 重组特征
        processed_features = self.reconstruct_features(imfs, other_features)

        return processed_features

    def analyze_imfs(self, imfs: np.ndarray) -> Dict:
        """
        分析IMF分量的特性

        计算每个IMF的能量、频率特性等统计信息。

        Args:
            imfs: IMF分量矩阵

        Returns:
            Dict: IMF分析结果
        """
        analysis = {
            'n_modes': imfs.shape[0],
            'signal_length': imfs.shape[1],
            'imf_stats': []
        }

        total_energy = 0
        for i in range(imfs.shape[0]):
            imf = imfs[i, :]

            # 计算能量
            energy = np.sum(imf ** 2)
            total_energy += energy

            # 计算统计量
            stats = {
                'mode': i + 1,
                'energy': float(energy),
                'mean': float(np.mean(imf)),
                'std': float(np.std(imf)),
                'min': float(np.min(imf)),
                'max': float(np.max(imf)),
                'center_frequency': float(self.omega[i]) if self.omega is not None else None
            }

            # 计算瞬时频率（使用Hilbert变换）
            analytic_signal = hilbert(imf)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)
            stats['mean_inst_freq'] = float(np.mean(instantaneous_frequency))
            stats['std_inst_freq'] = float(np.std(instantaneous_frequency))

            analysis['imf_stats'].append(stats)

        # 计算能量占比
        for stats in analysis['imf_stats']:
            stats['energy_ratio'] = stats['energy'] / total_energy if total_energy > 0 else 0

        analysis['total_energy'] = float(total_energy)

        logger.info("IMF分析完成")
        return analysis

    def plot_decomposition(self, signal: np.ndarray,
                          imfs: np.ndarray,
                          save_path: Optional[Union[str, Path]] = None) -> None:
        """
        可视化VMD分解结果

        绘制原始信号和各个IMF分量。

        Args:
            signal: 原始信号
            imfs: IMF分量
            save_path: 图像保存路径（可选）
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib未安装，跳过绘图")
            return

        n_plots = self.n_modes + 1
        fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2 * n_plots))

        # 绘制原始信号
        axes[0].plot(signal, 'b-', linewidth=1)
        axes[0].set_title('原始光伏功率信号', fontsize=12)
        axes[0].set_ylabel('功率 (kW)')
        axes[0].grid(True, alpha=0.3)

        # 绘制各IMF分量
        for i in range(self.n_modes):
            axes[i + 1].plot(imfs[i, :], 'g-', linewidth=1)
            if self.omega is not None:
                freq = self.omega[i]
                axes[i + 1].set_title(f'IMF{i+1} (中心频率: {freq:.4f})', fontsize=10)
            else:
                axes[i + 1].set_title(f'IMF{i+1}', fontsize=10)
            axes[i + 1].set_ylabel(f'IMF{i+1}')
            axes[i + 1].grid(True, alpha=0.3)

        axes[-1].set_xlabel('时间步')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            logger.info(f"分解图像已保存: {save_path}")

        plt.show()

    def validate_decomposition(self, signal: np.ndarray,
                              imfs: np.ndarray) -> Dict[str, float]:
        """
        验证VMD分解的质量

        检查重构误差、正交性等指标。

        Args:
            signal: 原始信号
            imfs: IMF分量

        Returns:
            Dict: 验证指标
        """
        # 重构信号
        reconstructed = np.sum(imfs, axis=0)

        # 计算重构误差
        reconstruction_error = np.mean((signal - reconstructed) ** 2)
        relative_error = reconstruction_error / np.mean(signal ** 2)

        # 计算正交性（IMF之间的相关性）
        orthogonality = []
        for i in range(self.n_modes):
            for j in range(i + 1, self.n_modes):
                corr = np.corrcoef(imfs[i, :], imfs[j, :])[0, 1]
                orthogonality.append(abs(corr))

        validation_metrics = {
            'reconstruction_error': float(reconstruction_error),
            'relative_error': float(relative_error),
            'mean_orthogonality': float(np.mean(orthogonality)) if orthogonality else 0,
            'max_orthogonality': float(np.max(orthogonality)) if orthogonality else 0,
            'signal_energy': float(np.sum(signal ** 2)),
            'reconstructed_energy': float(np.sum(reconstructed ** 2))
        }

        logger.info(f"分解验证: 重构误差={relative_error:.2e}, "
                   f"平均正交性={validation_metrics['mean_orthogonality']:.3f}")

        return validation_metrics

    def adaptive_mode_selection(self, signal: np.ndarray,
                              min_modes: int = 3,
                              max_modes: int = 10) -> int:
        """
        自适应选择最优模态数

        通过分析不同模态数下的分解质量，自动选择最优的模态数。

        Args:
            signal: 输入信号
            min_modes: 最小模态数
            max_modes: 最大模态数

        Returns:
            int: 最优模态数
        """
        best_modes = min_modes
        best_score = float('inf')

        logger.info(f"自适应模态选择: 测试{min_modes}到{max_modes}个模态")

        for n in range(min_modes, max_modes + 1):
            # 临时修改模态数
            original_modes = self.n_modes
            self.n_modes = n

            # 执行分解
            try:
                imfs, omega = self.decompose(signal)

                # 计算评价指标
                metrics = self.validate_decomposition(signal, imfs)

                # 综合评分（考虑重构误差和正交性）
                score = metrics['relative_error'] + 0.1 * metrics['mean_orthogonality']

                logger.info(f"  {n}个模态: 误差={metrics['relative_error']:.4f}, "
                          f"正交性={metrics['mean_orthogonality']:.4f}, 得分={score:.4f}")

                if score < best_score:
                    best_score = score
                    best_modes = n

            except Exception as e:
                logger.warning(f"  {n}个模态分解失败: {e}")

            # 恢复原始设置
            self.n_modes = original_modes

        logger.info(f"最优模态数: {best_modes}, 得分: {best_score:.4f}")
        return best_modes

    def save_params(self, filepath: Union[str, Path]) -> None:
        """
        保存VMD参数

        Args:
            filepath: 保存路径
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        params = {
            'decompose_params': self.decompose_params,
            'is_fitted': self.is_fitted,
            'omega': self.omega.tolist() if self.omega is not None else None
        }

        if filepath.suffix == '.json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(params, f, indent=2)
        elif filepath.suffix == '.pkl':
            params['u'] = self.u  # pickle可以保存numpy数组
            with open(filepath, 'wb') as f:
                pickle.dump(params, f)
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        logger.info(f"VMD参数已保存: {filepath}")

    def load_params(self, filepath: Union[str, Path]) -> None:
        """
        加载VMD参数

        Args:
            filepath: 参数文件路径
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"参数文件不存在: {filepath}")

        if filepath.suffix == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                params = json.load(f)
            self.omega = np.array(params['omega']) if params['omega'] else None
        elif filepath.suffix == '.pkl':
            with open(filepath, 'rb') as f:
                params = pickle.load(f)
            self.u = params.get('u')
            self.omega = params.get('omega')
        else:
            raise ValueError(f"不支持的文件格式: {filepath.suffix}")

        self.decompose_params = params['decompose_params']
        self.is_fitted = params['is_fitted']

        # 更新参数
        for key, value in self.decompose_params.items():
            setattr(self, key, value)

        logger.info(f"VMD参数已加载: {filepath}")
