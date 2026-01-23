# -*- coding: utf-8 -*-
"""
一键生成所有论文图片
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import setup_matplotlib, FIGURES_DIR

def main():
    """运行所有绘图脚本"""
    setup_matplotlib()
    
    print("=" * 60)
    print("FCM-MoE Paper Figure Generation")
    print(f"Output Directory: {FIGURES_DIR}")
    print("=" * 60)
    
    print("\n[1/4] Generating Ablation Study Figure...")
    from plot_ablation import plot_ablation_study
    plot_ablation_study()
    
    print("\n[2/4] Generating Cluster Performance Figure...")
    from plot_cluster_perf import plot_cluster_performance
    plot_cluster_performance()
    
    print("\n[3/4] Generating 3D State Space Figure...")
    from plot_state_space import plot_state_space_3d
    plot_state_space_3d()
    
    print("\n[4/4] Generating Regime Time Series Figure...")
    from plot_timeseries import plot_regime_timeseries
    plot_regime_timeseries()
    
    print("\n" + "=" * 60)
    print("All figures generated successfully!")
    print(f"Check: {FIGURES_DIR}")
    print("=" * 60)

if __name__ == '__main__':
    main()
