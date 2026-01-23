# -*- coding: utf-8 -*-
"""
Figure 2: 各簇多时域性能对比图
展示不同天气工况在1h/2h/4h预测时域的nRMSE对比
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import (
    setup_matplotlib, FIGURES_DIR, CLUSTER_PERF_DATA, CLUSTER_COLORS,
    DOUBLE_COL_WIDTH
)

def plot_cluster_performance():
    """绘制各簇性能对比图"""
    setup_matplotlib()
    
    clusters = CLUSTER_PERF_DATA['clusters']
    nrmse_1h = CLUSTER_PERF_DATA['1h_nrmse']
    nrmse_2h = CLUSTER_PERF_DATA['2h_nrmse']
    nrmse_4h = CLUSTER_PERF_DATA['4h_nrmse']
    
    x = np.arange(len(clusters))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.45))
    
    colors_1h = ['#3B82F6', '#3B82F6', '#3B82F6', '#1E40AF']
    colors_2h = ['#10B981', '#10B981', '#10B981', '#047857']
    colors_4h = ['#F59E0B', '#F59E0B', '#F59E0B', '#B45309']
    
    bars_1h = ax.bar(x - width, nrmse_1h, width, label='1-hour', color=colors_1h, edgecolor='white', linewidth=0.5)
    bars_2h = ax.bar(x, nrmse_2h, width, label='2-hour', color=colors_2h, edgecolor='white', linewidth=0.5)
    bars_4h = ax.bar(x + width, nrmse_4h, width, label='4-hour', color=colors_4h, edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Weather Regime')
    ax.set_ylabel('nRMSE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(clusters, fontsize=9)
    ax.legend(loc='upper left', frameon=True, fancybox=False, edgecolor='gray')
    
    ax.set_ylim(0, 20)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    def add_value_labels(bars, fontsize=8):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=fontsize)
    
    add_value_labels(bars_1h)
    add_value_labels(bars_2h)
    add_value_labels(bars_4h)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'cluster_performance.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    output_path_png = os.path.join(FIGURES_DIR, 'cluster_performance.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path_png}")
    
    plt.close()

if __name__ == '__main__':
    plot_cluster_performance()
