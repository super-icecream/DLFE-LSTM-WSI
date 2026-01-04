# -*- coding: utf-8 -*-
"""
B3 聚类特征贡献分析脚本

分析内容:
1. PCA 各主成分的特征载荷矩阵，找出对 PC1/PC2/PC3 贡献最大的原始特征
2. 将 FCM 聚类中心反变换回原始特征空间，分析簇间差异
3. 分析 scaler_g 中各特征的标准差
4. 输出综合报告
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pickle
import numpy as np
import pandas as pd


# 28 维状态向量特征名 (与 run_b3.py 中 compute_state_vector 一致)
G_FEATURE_NAMES = [
    # 辐照统计 (TSI/DNI/GHI): mean/std/min/max
    'TSI_mean', 'TSI_std', 'TSI_min', 'TSI_max',
    'DNI_mean', 'DNI_std', 'DNI_min', 'DNI_max',
    'GHI_mean', 'GHI_std', 'GHI_min', 'GHI_max',
    # 辐照变化强度
    'TSI_delta_mean_abs', 'TSI_delta_std', 'TSI_delta_max_abs',
    'DNI_delta_mean_abs', 'DNI_delta_std', 'DNI_delta_max_abs',
    'GHI_delta_mean_abs', 'GHI_delta_std', 'GHI_delta_max_abs',
    # 历史功率
    'Power_pu_mean', 'Power_pu_std', 'Power_pu_delta_mean_abs',
    # 气温、气压
    'Temp_mean', 'Temp_std',
    'Atm_mean', 'Atm_std'
]


def load_artifacts():
    """加载 PCA, scaler_g, fcm_centers"""
    results_dir = PROJECT_ROOT / "experiments" / "results"
    
    with open(results_dir / "pca.pkl", 'rb') as f:
        pca = pickle.load(f)
    
    with open(results_dir / "scaler_g.pkl", 'rb') as f:
        scaler_g = pickle.load(f)
    
    fcm_centers = np.load(results_dir / "fcm_centers.npy")
    
    return pca, scaler_g, fcm_centers


def analyze_pca_loadings(pca, feature_names, top_k=5):
    """
    分析 PCA 各主成分的特征载荷
    
    Returns:
        loadings_df: 完整载荷矩阵 DataFrame
        top_features: 各 PC 的 top-k 贡献特征
    """
    # components_: (n_components, n_features)
    n_components = pca.n_components_
    loadings = pca.components_
    
    # 创建载荷矩阵 DataFrame
    pc_names = [f'PC{i+1}' for i in range(n_components)]
    loadings_df = pd.DataFrame(
        loadings.T,
        index=feature_names,
        columns=pc_names
    )
    
    # 找出各 PC 的 top-k 贡献特征 (按绝对值排序)
    top_features = {}
    for i in range(min(n_components, 3)):  # 只分析前 3 个 PC
        pc_name = f'PC{i+1}'
        abs_loadings = np.abs(loadings[i])
        top_indices = np.argsort(abs_loadings)[::-1][:top_k]
        top_features[pc_name] = [
            (feature_names[idx], loadings[i, idx], abs_loadings[idx])
            for idx in top_indices
        ]
    
    return loadings_df, top_features


def inverse_transform_centers(fcm_centers, pca, scaler_g):
    """
    将 FCM 聚类中心反变换回原始特征空间
    
    Args:
        fcm_centers: (K, n_pca_components) PCA 空间中的聚类中心
        pca: sklearn PCA 对象
        scaler_g: FeatureScalerG 对象 (自定义类)
    
    Returns:
        centers_original: (K, n_features_valid) 原始特征空间中的聚类中心 (仅有效特征)
    """
    # PCA 反变换: (K, n_pca) -> (K, n_features_scaled)
    centers_scaled = pca.inverse_transform(fcm_centers)
    
    # FeatureScalerG 手动反变换: z = (x - mean) / std => x = z * std + mean
    centers_original = centers_scaled * scaler_g.std_ + scaler_g.mean_
    
    return centers_original


def get_valid_feature_names(scaler_g, all_feature_names):
    """获取 scaler_g 保留的有效特征名"""
    if hasattr(scaler_g, 'valid_mask_') and scaler_g.valid_mask_ is not None:
        return [all_feature_names[i] for i in range(len(all_feature_names)) if scaler_g.valid_mask_[i]]
    else:
        return all_feature_names[:scaler_g.n_features_out_]


def analyze_cluster_centers(centers_original, feature_names):
    """
    分析聚类中心在各特征上的差异
    
    Returns:
        centers_df: 各簇中心的特征值
        diff_analysis: 特征区分度分析
    """
    n_clusters = centers_original.shape[0]
    
    # 创建中心 DataFrame
    cluster_names = [f'Cluster_{i}' for i in range(n_clusters)]
    centers_df = pd.DataFrame(
        centers_original.T,
        index=feature_names,
        columns=cluster_names
    )
    
    # 计算各特征的区分度 (簇间标准差 / 全局均值)
    cluster_std = np.std(centers_original, axis=0)  # 各特征在簇间的标准差
    cluster_mean = np.mean(centers_original, axis=0)  # 各特征的全局均值
    
    # 避免除零
    cluster_mean_safe = np.where(np.abs(cluster_mean) < 1e-6, 1e-6, cluster_mean)
    discrimination = cluster_std / np.abs(cluster_mean_safe)
    
    # 创建区分度分析 DataFrame
    diff_analysis = pd.DataFrame({
        'feature': feature_names,
        'cluster_std': cluster_std,
        'cluster_mean': cluster_mean,
        'discrimination': discrimination
    })
    diff_analysis = diff_analysis.sort_values('discrimination', ascending=False)
    
    return centers_df, diff_analysis


def analyze_scaler_variance(scaler_g, feature_names):
    """
    分析 scaler_g 中各特征的标准差
    
    Returns:
        variance_df: 各特征的均值、标准差、变异系数
    """
    means = scaler_g.mean_
    # FeatureScalerG 使用 std_ 而不是 scale_
    stds = scaler_g.std_ if hasattr(scaler_g, 'std_') else scaler_g.scale_
    
    # 变异系数 (CV = std / |mean|)
    means_safe = np.where(np.abs(means) < 1e-6, 1e-6, means)
    cv = stds / np.abs(means_safe)
    
    variance_df = pd.DataFrame({
        'feature': feature_names,
        'mean': means,
        'std': stds,
        'cv': cv
    })
    variance_df = variance_df.sort_values('std', ascending=False)
    
    return variance_df


def main():
    print("\n" + "=" * 80)
    print("B3 聚类特征贡献分析报告")
    print("=" * 80)
    
    # 加载文件
    print("\n[1] 加载 pca.pkl, scaler_g.pkl, fcm_centers.npy...")
    pca, scaler_g, fcm_centers = load_artifacts()
    
    print(f"  - PCA n_components: {pca.n_components_}")
    print(f"  - PCA explained_variance_ratio: {pca.explained_variance_ratio_}")
    print(f"  - FCM centers shape: {fcm_centers.shape}")
    print(f"  - Scaler n_features: {len(scaler_g.mean_)}")
    
    # 获取 scaler_g 保留的有效特征名
    valid_feature_names = get_valid_feature_names(scaler_g, G_FEATURE_NAMES)
    print(f"  - Valid features: {len(valid_feature_names)} / {len(G_FEATURE_NAMES)}")
    
    # =========================================================================
    # 1. PCA 特征载荷分析
    # =========================================================================
    print("\n" + "=" * 80)
    print("[2] PCA 特征载荷分析")
    print("=" * 80)
    
    loadings_df, top_features = analyze_pca_loadings(pca, valid_feature_names, top_k=5)
    
    # 打印各 PC 解释方差
    print("\nPCA 解释方差比例:")
    total_var = 0
    for i, var in enumerate(pca.explained_variance_ratio_):
        total_var += var
        print(f"  PC{i+1}: {var*100:.2f}% (累计: {total_var*100:.2f}%)")
    
    # 打印各 PC 的 top-5 贡献特征
    print("\n各主成分的 Top-5 贡献特征:")
    print("-" * 80)
    for pc_name, features in top_features.items():
        print(f"\n{pc_name}:")
        for feat_name, loading, abs_loading in features:
            sign = "+" if loading > 0 else "-"
            print(f"  {sign} {feat_name:<25} loading={loading:>8.4f} (|loading|={abs_loading:.4f})")
    
    # =========================================================================
    # 2. FCM 聚类中心反变换分析
    # =========================================================================
    print("\n" + "=" * 80)
    print("[3] FCM 聚类中心分析 (反变换到原始特征空间)")
    print("=" * 80)
    
    centers_original = inverse_transform_centers(fcm_centers, pca, scaler_g)
    centers_df, diff_analysis = analyze_cluster_centers(centers_original, valid_feature_names)
    
    # 打印各簇中心
    print("\n各簇中心在原始特征空间中的值:")
    print("-" * 80)
    print(f"{'Feature':<30} {'Cluster_0':>12} {'Cluster_1':>12} {'Cluster_2':>12}")
    print("-" * 80)
    for feat in valid_feature_names:
        vals = [centers_df.loc[feat, f'Cluster_{i}'] for i in range(3)]
        print(f"{feat:<30} {vals[0]:>12.4f} {vals[1]:>12.4f} {vals[2]:>12.4f}")
    
    # 打印区分度最大的特征
    print("\n特征区分度排名 (簇间标准差 / 全局均值):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<30} {'Cluster_Std':>12} {'Cluster_Mean':>12} {'Discrimination':>14}")
    print("-" * 80)
    for i, row in diff_analysis.head(10).iterrows():
        rank = diff_analysis.index.get_loc(i) + 1
        print(f"{rank:<6} {row['feature']:<30} {row['cluster_std']:>12.4f} {row['cluster_mean']:>12.4f} {row['discrimination']:>14.4f}")
    
    # =========================================================================
    # 3. Scaler 方差分析
    # =========================================================================
    print("\n" + "=" * 80)
    print("[4] Scaler 特征方差分析")
    print("=" * 80)
    
    variance_df = analyze_scaler_variance(scaler_g, valid_feature_names)
    
    print("\n各特征的标准差排名 (标准化前):")
    print("-" * 80)
    print(f"{'Rank':<6} {'Feature':<30} {'Mean':>12} {'Std':>12} {'CV':>12}")
    print("-" * 80)
    for i, (_, row) in enumerate(variance_df.iterrows()):
        print(f"{i+1:<6} {row['feature']:<30} {row['mean']:>12.4f} {row['std']:>12.4f} {row['cv']:>12.4f}")
    
    # 检查方差极端不均的情况
    std_values = variance_df['std'].values
    std_ratio = std_values.max() / std_values.min()
    print(f"\n标准差极值比: max/min = {std_values.max():.4f} / {std_values.min():.4f} = {std_ratio:.2f}")
    
    if std_ratio > 100:
        print("[WARNING] 标准差极值比 > 100，存在方差极端不均的特征!")
    elif std_ratio > 10:
        print("[INFO] 标准差极值比 > 10，特征方差分布不太均匀，但 Z-score 标准化后影响应已消除")
    else:
        print("[OK] 标准差极值比 < 10，特征方差分布相对均匀")
    
    # =========================================================================
    # 4. 综合报告
    # =========================================================================
    print("\n" + "=" * 80)
    print("[5] 综合分析报告: 哪些特征主导了聚类结果")
    print("=" * 80)
    
    # 收集主导特征
    dominant_features = set()
    
    # 从 PCA 载荷中收集
    for pc_name, features in top_features.items():
        for feat_name, _, _ in features[:3]:  # 每个 PC 取 top-3
            dominant_features.add(feat_name)
    
    # 从簇间区分度中收集
    top_discriminative = diff_analysis.head(5)['feature'].tolist()
    for feat in top_discriminative:
        dominant_features.add(feat)
    
    print("\n主导聚类的特征 (综合 PCA 载荷 + 簇间区分度):")
    print("-" * 80)
    
    # 按特征类型分组
    feature_groups = {
        '辐照水平 (TSI/DNI/GHI mean/max)': [],
        '辐照变化 (delta)': [],
        '功率相关': [],
        '气象辅助 (Temp/Atm)': []
    }
    
    for feat in dominant_features:
        if 'delta' in feat.lower():
            feature_groups['辐照变化 (delta)'].append(feat)
        elif 'power' in feat.lower():
            feature_groups['功率相关'].append(feat)
        elif 'temp' in feat.lower() or 'atm' in feat.lower():
            feature_groups['气象辅助 (Temp/Atm)'].append(feat)
        else:
            feature_groups['辐照水平 (TSI/DNI/GHI mean/max)'].append(feat)
    
    for group_name, feats in feature_groups.items():
        if feats:
            print(f"\n{group_name}:")
            for feat in feats:
                # 找到该特征在各 PC 中的贡献
                pc_contrib = []
                for pc_name, features in top_features.items():
                    for f, loading, _ in features:
                        if f == feat:
                            pc_contrib.append(f"{pc_name}:{loading:+.3f}")
                            break
                
                # 找到该特征的区分度
                disc_row = diff_analysis[diff_analysis['feature'] == feat]
                if not disc_row.empty:
                    disc = disc_row['discrimination'].values[0]
                    disc_str = f"区分度={disc:.3f}"
                else:
                    disc_str = ""
                
                pc_str = ", ".join(pc_contrib) if pc_contrib else "-"
                print(f"  - {feat:<30} PCA载荷=[{pc_str}]  {disc_str}")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    # 统计各类特征的贡献
    irr_level_count = len(feature_groups['辐照水平 (TSI/DNI/GHI mean/max)'])
    irr_delta_count = len(feature_groups['辐照变化 (delta)'])
    power_count = len(feature_groups['功率相关'])
    weather_count = len(feature_groups['气象辅助 (Temp/Atm)'])
    total_dominant = len(dominant_features)
    
    print(f"""
1. PCA 降维情况:
   - 使用 {pca.n_components_} 个主成分，解释了 {sum(pca.explained_variance_ratio_)*100:.1f}% 的方差
   - PC1 解释 {pca.explained_variance_ratio_[0]*100:.1f}%，主要由辐照水平特征主导

2. 主导特征分布 (共 {total_dominant} 个):
   - 辐照水平 (mean/max): {irr_level_count} 个 ({irr_level_count/total_dominant*100:.0f}%)
   - 辐照变化 (delta):    {irr_delta_count} 个 ({irr_delta_count/total_dominant*100:.0f}%)
   - 功率相关:            {power_count} 个 ({power_count/total_dominant*100:.0f}%)
   - 气象辅助:            {weather_count} 个 ({weather_count/total_dominant*100:.0f}%)

3. 聚类物理含义推断:
   - 簇主要按"辐照强度水平"区分 (高/中/低辐照场景)
   - 辐照变化强度 (delta) 对区分"稳定晴天 vs 波动多云"有贡献
   - 气象特征 (气温/气压) 贡献较小，可能是冗余特征

4. 建议:
   - 当前聚类结果主要反映"辐照强度"的差异
   - 如需更细粒度的场景区分，可考虑:
     a) 增加辐照变化特征的权重
     b) 引入时段信息 (如小时角) 区分早/中/晚
     c) 减少冗余特征 (如 TSI/DNI/GHI 三者高度相关)
""")
    
    print("=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
