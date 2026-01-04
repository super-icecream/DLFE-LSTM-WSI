# -*- coding: utf-8 -*-
"""
FCM 聚类参数网格搜索脚本

基于 B3 代码构建，仅在训练集上进行参数搜索，避免数据泄露。

搜索参数:
- K: 固定为 3
- m: [1.3, 1.5, 1.8, 2.0, 2.2, 2.5]
- PCA_n_components: [3, 5, 7, 10]

约束条件:
- 最小簇样本占比 >= 20%

评估指标:
- Silhouette Score (越大越好)
- Davies-Bouldin Index (越小越好)
- min_cluster_ratio (最小簇占比)

输出:
- 按 Silhouette 降序排列的满足约束的参数组合表格
- 最优参数保存到 JSON 文件
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_solar_data, print_load_summary
from src.data.splitter import split_by_day, print_split_summary
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows
from src.utils.seed import set_seed


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


# =============================================================================
# 状态向量 g_t 构建 (复用 B3)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """计算单个样本的状态向量 g_t"""
    X = sample.X
    
    g_values = []
    g_names = []
    
    def get_col_idx(partial_name):
        for i, name in enumerate(feature_names):
            if partial_name.lower() in name.lower():
                return i
        return None
    
    tsi_idx = get_col_idx('Total solar')
    dni_idx = get_col_idx('Direct normal')
    ghi_idx = get_col_idx('Global horizontal')
    temp_idx = get_col_idx('temperature')
    atm_idx = get_col_idx('Atmosphere')
    power_idx = get_col_idx('Power')
    
    # 辐照统计
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])
            g_names.extend([f'{name}_mean', f'{name}_std', f'{name}_min', f'{name}_max'])
    
    # 辐照变化强度
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])
                g_names.extend([f'{name}_delta_mean_abs', f'{name}_delta_std', f'{name}_delta_max_abs'])
    
    # 历史功率
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        g_names.extend(['Power_pu_mean', 'Power_pu_std'])
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))
            g_names.append('Power_pu_delta_mean_abs')
    
    # 气温、气压
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])
        g_names.extend(['Temp_mean', 'Temp_std'])
    
    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])
        g_names.extend(['Atm_mean', 'Atm_std'])
    
    return np.array(g_values, dtype=np.float32), g_names


def build_state_vectors(samples: List[WindowSample], feature_names: List[str], capacity: float) -> Tuple[np.ndarray, List[str]]:
    """为所有样本构建状态向量"""
    G_list = []
    g_names = None
    
    for sample in samples:
        g, names = compute_state_vector(sample, feature_names, capacity)
        G_list.append(g)
        if g_names is None:
            g_names = names
    
    G = np.array(G_list, dtype=np.float32)
    return G, g_names


# =============================================================================
# FCM 聚类 (复用 B3)
# =============================================================================

class FCM:
    """Fuzzy C-Means 聚类"""
    
    def __init__(self, n_clusters: int = 3, m: float = 2.0, max_iter: int = 300, 
                 tol: float = 1e-5, random_state: int = 42):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.n_iter_ = 0
        self.jm_ = None
    
    def _init_membership(self, n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        U = rng.rand(n_samples, self.n_clusters)
        U = U / U.sum(axis=1, keepdims=True)
        return U
    
    def _compute_centers(self, X: np.ndarray, U: np.ndarray) -> np.ndarray:
        Um = U ** self.m
        centers = (Um.T @ X) / Um.sum(axis=0, keepdims=True).T
        return centers
    
    def _compute_distances(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        for k in range(self.n_clusters):
            diff = X - centers[k]
            distances[:, k] = np.sum(diff ** 2, axis=1)
        return distances
    
    def _compute_membership(self, distances: np.ndarray) -> np.ndarray:
        distances = np.maximum(distances, 1e-10)
        power = 2.0 / (self.m - 1)
        
        U = np.zeros_like(distances)
        for k in range(self.n_clusters):
            denom = 0
            for j in range(self.n_clusters):
                denom += (distances[:, k] / distances[:, j]) ** power
            U[:, k] = 1.0 / denom
        
        return U
    
    def _compute_jm(self, X: np.ndarray, U: np.ndarray, centers: np.ndarray) -> float:
        distances = self._compute_distances(X, centers)
        Um = U ** self.m
        return np.sum(Um * distances)
    
    def fit(self, X: np.ndarray) -> 'FCM':
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        U = self._init_membership(n_samples, rng)
        
        for i in range(self.max_iter):
            centers = self._compute_centers(X, U)
            distances = self._compute_distances(X, centers)
            U_new = self._compute_membership(distances)
            
            diff = np.abs(U_new - U).max()
            U = U_new
            
            if diff < self.tol:
                self.n_iter_ = i + 1
                break
        else:
            self.n_iter_ = self.max_iter
        
        self.centers_ = centers
        self.jm_ = self._compute_jm(X, U, centers)
        
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        distances = self._compute_distances(X, self.centers_)
        U = self._compute_membership(distances)
        labels = np.argmax(U, axis=1)
        return U, labels


def run_fcm_multi_init(X: np.ndarray, n_clusters: int = 3, m: float = 2.0, 
                       max_iter: int = 300, tol: float = 1e-5, n_init: int = 10,
                       random_seed: int = 42) -> FCM:
    """多次初始化运行 FCM，选择 J_m 最小的"""
    best_fcm = None
    best_jm = float('inf')
    
    for i in range(n_init):
        fcm = FCM(n_clusters=n_clusters, m=m, max_iter=max_iter, tol=tol, 
                  random_state=random_seed + i)
        fcm.fit(X)
        
        if fcm.jm_ < best_jm:
            best_jm = fcm.jm_
            best_fcm = fcm
    
    return best_fcm


# =============================================================================
# 参数搜索结果
# =============================================================================

@dataclass
class SearchResult:
    """单次参数搜索结果"""
    m: float
    pca_n_components: int
    silhouette: float
    davies_bouldin: float
    min_cluster_ratio: float
    cluster_ratios: List[float]
    jm: float
    n_iter: int
    satisfies_constraint: bool


def evaluate_clustering(Z: np.ndarray, labels: np.ndarray, fcm: FCM, 
                        min_ratio_threshold: float = 0.20) -> SearchResult:
    """
    评估聚类质量
    
    Args:
        Z: PCA 降维后的数据
        labels: 硬标签
        fcm: 训练好的 FCM 模型
        min_ratio_threshold: 最小簇占比阈值
    
    Returns:
        SearchResult
    """
    n_samples = len(labels)
    n_clusters = fcm.n_clusters
    
    # 计算各簇样本占比
    cluster_counts = [np.sum(labels == k) for k in range(n_clusters)]
    cluster_ratios = [c / n_samples for c in cluster_counts]
    min_cluster_ratio = min(cluster_ratios)
    
    # 检查约束
    satisfies_constraint = min_cluster_ratio >= min_ratio_threshold
    
    # 计算评估指标
    if len(np.unique(labels)) > 1:
        silhouette = silhouette_score(Z, labels)
        davies_bouldin = davies_bouldin_score(Z, labels)
    else:
        silhouette = -1.0
        davies_bouldin = float('inf')
    
    return SearchResult(
        m=fcm.m,
        pca_n_components=Z.shape[1],
        silhouette=silhouette,
        davies_bouldin=davies_bouldin,
        min_cluster_ratio=min_cluster_ratio,
        cluster_ratios=cluster_ratios,
        jm=fcm.jm_,
        n_iter=fcm.n_iter_,
        satisfies_constraint=satisfies_constraint
    )


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("FCM 聚类参数网格搜索")
    print("=" * 80)
    
    # =========================================================================
    # 1. 加载配置
    # =========================================================================
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    print(f"\n[1/6] 配置加载完成")
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    daylight_config = config.get('daylight_filter', {'enabled': False})
    output_config = config['output']
    train_config = config.get('training', {})
    
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    
    # 随机种子
    seed = train_config.get('seed', 42)
    set_seed(seed, deterministic=True)
    
    # 搜索参数空间
    K = 3  # 固定
    m_list = [1.3, 1.5, 1.8, 2.0, 2.2, 2.5]
    pca_n_components_list = [3, 5, 7, 10]
    min_ratio_threshold = 0.20  # 最小簇占比约束
    
    print(f"  - K (固定): {K}")
    print(f"  - m: {m_list}")
    print(f"  - PCA n_components: {pca_n_components_list}")
    print(f"  - 最小簇占比约束: >= {min_ratio_threshold*100:.0f}%")
    print(f"  - 总参数组合: {len(m_list) * len(pca_n_components_list)}")
    
    # =========================================================================
    # 2. 数据读取
    # =========================================================================
    print("\n[2/6] 读取数据...")
    
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']
    
    print(f"  数据行数: {len(load_result.df)}")
    
    # =========================================================================
    # 3. 白天筛选 + 按天切分
    # =========================================================================
    print("\n[3/6] 白天筛选 + 按天切分...")
    
    df_for_split = load_result.df
    
    if daylight_enabled:
        daylight_result = add_daylight_flag(
            df=load_result.df,
            time_column=data_config['time_column'],
            dni_col=daylight_config['dni_col'],
            threshold=daylight_config['threshold']
        )
        
        if daylight_mode == 'drop':
            filtered_df, _, _ = filter_daylight_rows(daylight_result.df)
            df_for_split = filtered_df
        else:
            df_for_split = daylight_result.df
    
    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    print(f"  Train days: {split_result.train_days}, Val days: {split_result.val_days}, Test days: {split_result.test_days}")
    
    # =========================================================================
    # 4. 仅生成训练集滑窗样本 (避免数据泄露)
    # =========================================================================
    print("\n[4/6] 生成训练集滑窗样本...")
    
    use_mask_mode = daylight_enabled and daylight_mode == 'mask'
    
    train_window_result = generate_windows(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=use_mask_mode
    )
    
    train_samples = train_window_result.samples
    print(f"  训练集样本数: {len(train_samples)}")
    
    # =========================================================================
    # 5. 构建状态向量
    # =========================================================================
    print("\n[5/6] 构建状态向量 g_t...")
    
    G_train_raw, g_names = build_state_vectors(train_samples, feature_names, capacity)
    print(f"  状态向量维度: {G_train_raw.shape[1]}")
    
    # 移除 NaN/inf 样本
    valid_mask = np.all(np.isfinite(G_train_raw), axis=1)
    G_train_clean = G_train_raw[valid_mask]
    n_dropped = np.sum(~valid_mask)
    if n_dropped > 0:
        print(f"  移除无效样本: {n_dropped}")
    print(f"  有效样本数: {len(G_train_clean)}")
    
    # 标准化
    scaler_g = StandardScaler()
    G_train_scaled = scaler_g.fit_transform(G_train_clean)
    
    # =========================================================================
    # 6. 网格搜索
    # =========================================================================
    print("\n[6/6] 开始网格搜索...")
    print("-" * 80)
    print(f"{'m':>6} {'PCA_n':>8} {'Silhouette':>12} {'DB Index':>12} {'Min Ratio':>12} {'Constraint':>12} {'J_m':>14}")
    print("-" * 80)
    
    all_results: List[SearchResult] = []
    
    for pca_n in pca_n_components_list:
        # PCA 降维
        pca = PCA(n_components=pca_n, random_state=seed)
        Z_train = pca.fit_transform(G_train_scaled)
        
        for m in m_list:
            # FCM 聚类
            fcm = run_fcm_multi_init(
                X=Z_train,
                n_clusters=K,
                m=m,
                max_iter=300,
                tol=1e-5,
                n_init=10,
                random_seed=seed
            )
            
            # 预测标签
            _, labels = fcm.predict(Z_train)
            
            # 评估
            result = evaluate_clustering(Z_train, labels, fcm, min_ratio_threshold)
            result.m = m
            result.pca_n_components = pca_n
            all_results.append(result)
            
            # 打印当前结果
            constraint_str = "PASS" if result.satisfies_constraint else "FAIL"
            print(f"{m:>6.1f} {pca_n:>8} {result.silhouette:>12.4f} {result.davies_bouldin:>12.4f} "
                  f"{result.min_cluster_ratio*100:>11.1f}% {constraint_str:>12} {result.jm:>14.2f}")
    
    print("-" * 80)
    
    # =========================================================================
    # 筛选满足约束的结果，按 Silhouette 降序排列
    # =========================================================================
    valid_results = [r for r in all_results if r.satisfies_constraint]
    valid_results.sort(key=lambda x: x.silhouette, reverse=True)
    
    print(f"\n满足约束 (min_cluster_ratio >= {min_ratio_threshold*100:.0f}%) 的参数组合: {len(valid_results)}/{len(all_results)}")
    
    if valid_results:
        print("\n" + "=" * 100)
        print("满足约束的参数组合 (按 Silhouette 降序)")
        print("=" * 100)
        print(f"{'Rank':>6} {'m':>6} {'PCA_n':>8} {'Silhouette':>12} {'DB Index':>12} {'Min Ratio':>12} {'Cluster Ratios':>30}")
        print("-" * 100)
        
        for i, r in enumerate(valid_results):
            ratios_str = ", ".join([f"{x*100:.1f}%" for x in r.cluster_ratios])
            print(f"{i+1:>6} {r.m:>6.1f} {r.pca_n_components:>8} {r.silhouette:>12.4f} {r.davies_bouldin:>12.4f} "
                  f"{r.min_cluster_ratio*100:>11.1f}% [{ratios_str:>26}]")
        
        print("=" * 100)
        
        # 最优参数
        best = valid_results[0]
        print(f"\n最优参数组合:")
        print(f"  - m: {best.m}")
        print(f"  - PCA n_components: {best.pca_n_components}")
        print(f"  - Silhouette Score: {best.silhouette:.4f}")
        print(f"  - Davies-Bouldin Index: {best.davies_bouldin:.4f}")
        print(f"  - Min Cluster Ratio: {best.min_cluster_ratio*100:.1f}%")
        print(f"  - Cluster Ratios: {[f'{x*100:.1f}%' for x in best.cluster_ratios]}")
        
        # 保存最优参数到 JSON
        results_dir = PROJECT_ROOT / output_config['results_dir']
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        best_params = {
            'timestamp': timestamp,
            'search_space': {
                'K': K,
                'm_list': m_list,
                'pca_n_components_list': pca_n_components_list,
                'min_ratio_threshold': min_ratio_threshold
            },
            'best_params': {
                'K': K,
                'm': float(best.m),
                'pca_n_components': int(best.pca_n_components)
            },
            'best_metrics': {
                'silhouette': float(best.silhouette),
                'davies_bouldin': float(best.davies_bouldin),
                'min_cluster_ratio': float(best.min_cluster_ratio),
                'cluster_ratios': [float(x) for x in best.cluster_ratios],
                'jm': float(best.jm)
            },
            'all_valid_results': [
                {
                    'm': float(r.m),
                    'pca_n_components': int(r.pca_n_components),
                    'silhouette': float(r.silhouette),
                    'davies_bouldin': float(r.davies_bouldin),
                    'min_cluster_ratio': float(r.min_cluster_ratio),
                    'cluster_ratios': [float(x) for x in r.cluster_ratios]
                }
                for r in valid_results
            ],
            'train_samples': len(G_train_clean)
        }
        
        json_path = results_dir / "fcm_best_params.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)
        print(f"\n最优参数已保存: {json_path}")
        
        # 保存完整搜索结果 CSV
        csv_path = results_dir / f"fcm_sweep_results_{timestamp}.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("m,pca_n_components,silhouette,davies_bouldin,min_cluster_ratio,cluster_0_ratio,cluster_1_ratio,cluster_2_ratio,jm,satisfies_constraint\n")
            for r in all_results:
                f.write(f"{r.m},{r.pca_n_components},{r.silhouette:.6f},{r.davies_bouldin:.6f},"
                        f"{r.min_cluster_ratio:.4f},{r.cluster_ratios[0]:.4f},{r.cluster_ratios[1]:.4f},{r.cluster_ratios[2]:.4f},"
                        f"{r.jm:.4f},{r.satisfies_constraint}\n")
        print(f"完整搜索结果已保存: {csv_path}")
        
    else:
        print("\n[WARNING] 没有满足约束的参数组合!")
        print("建议: 降低 min_ratio_threshold 或调整搜索空间")
    
    print("\n" + "=" * 80)
    print("FCM 参数网格搜索完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
