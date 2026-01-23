# -*- coding: utf-8 -*-
"""
FCM-MoE 论文绘图统一配置
字体、颜色、路径等全局设置
"""

import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'results')
FIGURES_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'paper', 'latex', 'FCM_MoE', 'figures'))

# 确保输出目录存在
os.makedirs(FIGURES_DIR, exist_ok=True)

# ============================================================
# 字体配置 (JPCS模板使用Times New Roman)
# ============================================================
FONT_FAMILY = 'Times New Roman'
FONT_SIZE_TITLE = 12
FONT_SIZE_LABEL = 11
FONT_SIZE_TICK = 10
FONT_SIZE_LEGEND = 9

def setup_matplotlib():
    """设置matplotlib全局参数，确保字体与JPCS模板一致"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': [FONT_FAMILY, 'DejaVu Serif'],
        'font.size': FONT_SIZE_LABEL,
        'axes.titlesize': FONT_SIZE_TITLE,
        'axes.labelsize': FONT_SIZE_LABEL,
        'xtick.labelsize': FONT_SIZE_TICK,
        'ytick.labelsize': FONT_SIZE_TICK,
        'legend.fontsize': FONT_SIZE_LEGEND,
        'figure.titlesize': FONT_SIZE_TITLE,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })

# ============================================================
# 颜色配置
# ============================================================
COLORS = {
    'primary': '#2563EB',
    'secondary': '#DC2626',
    'tertiary': '#059669',
    'quaternary': '#7C3AED',
    'gray': '#6B7280',
}

CLUSTER_COLORS = {
    0: '#0EA5E9',
    1: '#F59E0B',
    2: '#6B7280',
}

CLUSTER_NAMES = {
    0: 'Low Irradiance',
    1: 'Clear Sky',
    2: 'Cloudy/Volatile',
}

ABLATION_COLORS = {
    'B0': '#9CA3AF',
    'B2': '#3B82F6',
    'B4': '#F59E0B',
    'B5.5': '#10B981',
    'B6': '#EF4444',
}

# ============================================================
# 图片尺寸配置 (JPCS单栏宽度约84mm)
# ============================================================
SINGLE_COL_WIDTH = 3.5
DOUBLE_COL_WIDTH = 7.0
FIG_HEIGHT_RATIO = 0.75

# ============================================================
# 实验数据（来自ablation_chain.md的真实数据）
# ============================================================
ABLATION_DATA = {
    'stages': ['B0', 'B2', 'B4', 'B5.5', 'B6'],
    'labels': ['Persistence', 'Global LSTM', 'Hard Experts\n(Scratch)', 'Hard Experts\n(Warm-Start)', 'GRU Experts\n(Warm-Start)'],
    '1h_nrmse': [18.36, 10.35, 9.48, 8.34, 8.00],
    '2h_nrmse': [30.69, 14.03, 11.70, 11.02, 10.44],
    '4h_nrmse': [44.49, 16.37, 14.17, 13.55, 13.06],
}

CLUSTER_PERF_DATA = {
    'clusters': ['Cluster 1\n(Clear Sky)', 'Cluster 2\n(Cloudy)', 'Cluster 0\n(Low Irradiance)', 'Overall'],
    '1h_nrmse': [7.28, 9.52, 8.32, 8.00],
    '2h_nrmse': [9.20, 11.54, 11.82, 10.44],
    '4h_nrmse': [10.63, 14.43, 17.00, 13.06],
    'sample_ratio': [42.3, 35.1, 22.6, 100.0],
}

if __name__ == '__main__':
    setup_matplotlib()
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Results Dir: {RESULTS_DIR}")
    print(f"Figures Dir: {FIGURES_DIR}")
