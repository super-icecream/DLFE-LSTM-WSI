# -*- coding: utf-8 -*-
"""检查数据集字段"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
file_path = PROJECT_ROOT / "datas/2. 甘肃光伏功率预测数据集/data_processed/solar_stations/Solar station site 1 (Nominal capacity-50MW).xlsx"

df = pd.read_excel(file_path)

print("=" * 60)
print("数据集字段分析")
print("=" * 60)

print("\n=== 数据列名 (共 {} 列) ===".format(len(df.columns)))
for i, col in enumerate(df.columns):
    print("  {}: [{}]".format(i, col))

print("\n=== 数据形状 ===")
print("  行数: {}, 列数: {}".format(df.shape[0], df.shape[1]))

print("\n=== 数据类型 ===")
for col in df.columns:
    print("  {}: {}".format(col, df[col].dtype))

# 检查时间列范围
time_col = "Time(year-month-day h:m:s)"
df[time_col] = pd.to_datetime(df[time_col])
print("\n=== 时间范围 ===")
print("  起始: {}".format(df[time_col].min()))
print("  结束: {}".format(df[time_col].max()))
