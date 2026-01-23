# FCM-MoE 论文绘图脚本

## 目录结构

```
scripts/plotting/
├── README.md           # 本文件
├── config.py           # 统一配置（字体、颜色、路径）
├── plot_ablation.py    # Figure 1: 消融实验柱状图
├── plot_cluster_perf.py # Figure 2: 各簇性能对比图
├── plot_state_space.py  # Figure 3: 3D状态空间嵌入图
├── plot_timeseries.py   # Figure 4: 工况感知时序图
└── run_all.py          # 一键生成所有图片
```

## 图片保存位置

`paper/latex/FCM_MoE/figures/`

## 字体要求

JPCS模板使用 **Times New Roman** 字体，图片中的文字需保持一致。

## 图片格式

- 输出格式: **PDF** (矢量图，LaTeX推荐)
- 备选格式: EPS, PNG (300 DPI)
- 宽度建议: 单栏 84mm, 双栏 174mm

## 运行方式

```bash
conda activate New-LSTM_A1
cd scripts/plotting
python run_all.py
```

## 图表清单

| Figure | 文件名 | 描述 | 状态 |
|--------|--------|------|------|
| Fig. 1 | ablation_study.pdf | 消融实验nRMSE对比（1h/2h/4h） | 待完成 |
| Fig. 2 | cluster_performance.pdf | 各簇多时域性能对比 | 待完成 |
| Fig. 3 | state_space_3d.pdf | PCA状态空间聚类可视化 | 待完成 |
| Fig. 4 | regime_timeseries.pdf | 工况感知时序预测图 | 待完成 |
