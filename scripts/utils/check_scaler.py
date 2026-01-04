# -*- coding: utf-8 -*-
"""检查 scaler_X 信息"""
import sys
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
scaler_path = PROJECT_ROOT / "experiments/results/scaler_X.pkl"

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print("=" * 60)
print("scaler_X 信息")
print("=" * 60)
print("n_features: {}".format(scaler.n_features_))
print("\nmean (均值):")
for i, m in enumerate(scaler.mean_):
    print("  Feature {}: {:.4f}".format(i, m))

print("\nstd (标准差):")
for i, s in enumerate(scaler.std_):
    print("  Feature {}: {:.4f}".format(i, s))
