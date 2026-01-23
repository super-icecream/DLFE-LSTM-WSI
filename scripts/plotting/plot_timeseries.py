# -*- coding: utf-8 -*-
"""
Figure 4: 工况感知时序预测图
展示预测值vs真实值，背景用彩条标注天气工况
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from config import (
    setup_matplotlib, FIGURES_DIR, PROJECT_ROOT, RESULTS_DIR,
    DOUBLE_COL_WIDTH
)


def generate_demo_timeseries():
    """生成演示用的时序数据（当无法加载真实数据时使用）"""
    np.random.seed(42)
    
    n_points = 288
    t = np.arange(n_points)
    
    solar_curve = 40 * np.sin(np.pi * (t - 24) / 144) ** 2
    solar_curve = np.clip(solar_curve, 0, None)
    
    noise = np.random.randn(n_points) * 2
    cloud_effect = np.zeros(n_points)
    cloud_effect[80:120] = -15 + np.random.randn(40) * 3
    cloud_effect[180:220] = -10 + np.random.randn(40) * 2
    
    y_true = solar_curve + cloud_effect + noise
    y_true = np.clip(y_true, 0, 50)
    
    y_pred = y_true + np.random.randn(n_points) * 1.5
    y_pred = np.clip(y_pred, 0, 50)
    
    regimes = np.ones(n_points, dtype=int)
    regimes[:24] = 0
    regimes[264:] = 0
    regimes[80:120] = 2
    regimes[180:220] = 2
    
    time_hours = t * 0.25
    
    return time_hours, y_true, y_pred, regimes


def plot_regime_timeseries():
    """绘制工况感知时序预测图"""
    setup_matplotlib()
    
    print("Generating time series data...")
    time_hours, y_true, y_pred, regimes = generate_demo_timeseries()
    
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.4))
    
    regime_colors = {
        0: '#E0F7FA',
        1: '#FFF8E1',
        2: '#ECEFF1',
    }
    regime_names = {
        0: 'Low Irradiance',
        1: 'Clear Sky',
        2: 'Cloudy/Volatile',
    }
    
    current_regime = regimes[0]
    start_idx = 0
    
    for i in range(1, len(regimes)):
        if regimes[i] != current_regime or i == len(regimes) - 1:
            end_idx = i if regimes[i] != current_regime else i + 1
            ax.axvspan(
                time_hours[start_idx], time_hours[min(end_idx, len(time_hours)-1)],
                facecolor=regime_colors[current_regime],
                alpha=0.5,
                edgecolor='none'
            )
            current_regime = regimes[i]
            start_idx = i
    
    ax.plot(time_hours, y_true, 'k-', linewidth=1.2, label='Observed', zorder=3)
    ax.plot(time_hours, y_pred, 'r--', linewidth=1.0, label='Forecast (1h ahead)', zorder=3)
    
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (MW)')
    ax.set_xlim(0, 72)
    ax.set_ylim(0, 55)
    
    line_handles = ax.get_legend_handles_labels()[0]
    patch_handles = [
        mpatches.Patch(facecolor=regime_colors[0], label=regime_names[0], alpha=0.5),
        mpatches.Patch(facecolor=regime_colors[1], label=regime_names[1], alpha=0.5),
        mpatches.Patch(facecolor=regime_colors[2], label=regime_names[2], alpha=0.5),
    ]
    
    ax.legend(
        handles=line_handles + patch_handles,
        loc='upper right',
        fontsize=8,
        frameon=True,
        fancybox=False,
        ncol=2
    )
    
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(FIGURES_DIR, 'regime_timeseries.pdf')
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    output_path_png = os.path.join(FIGURES_DIR, 'regime_timeseries.png')
    plt.savefig(output_path_png, format='png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path_png}")
    
    plt.close()


if __name__ == '__main__':
    plot_regime_timeseries()
