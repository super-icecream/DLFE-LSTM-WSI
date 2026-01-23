# -*- coding: utf-8 -*-
"""
Figure 1: 消融实验柱状图
展示B0->B6各阶段在1h/2h/4h预测时域的nRMSE对比
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from config import (
    setup_matplotlib, FIGURES_DIR, ABLATION_DATA, ABLATION_COLORS,
    SINGLE_COL_WIDTH, DOUBLE_COL_WIDTH, FIG_HEIGHT_RATIO
)

def plot_ablation_study():
    """绘制消融实验柱状图"""
    setup_matplotlib()
    
    stages = ABLATION_DATA['stages']
    labels = ABLATION_DATA['labels']
    nrmse_1h = ABLATION_DATA['1h_nrmse']
    nrmse_2h = ABLATION_DATA['2h_nrmse']
    nrmse_4h = ABLATION_DATA['4h_nrmse']
    
    x = np.arange(len(stages))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5))
    
    bars_1h = ax.bar(x - width, nrmse_1h, width, label='1-hour', color='#3B82F6', edgecolor='white', linewidth=0.5)
    bars_2h = ax.bar(x, nrmse_2h, width, label='2-hour', color='#10B981', edgecolor='white', linewidth=0.5)
    bars_4h = ax.bar(x + width, nrmse_4h, width, label='4-hour', color='#F59E0B', edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Model Configuration')
    ax.set_ylabel('nRMSE (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='gray')
    
    ax.set_ylim(0, 50)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    def add_value_labels(bars, fontsize=7):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 2),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=fontsize)
    
    add_value_labels(bars_1h)
    add_value_labels(bars_2h)
    add_value_labels(bars_4h)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'ablation_study.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    output_path_png = os.path.join(FIGURES_DIR, 'ablation_study.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path_png}")
    
    plt.close()

if __name__ == '__main__':
    plot_ablation_study()
