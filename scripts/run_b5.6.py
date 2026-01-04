# -*- coding: utf-8 -*-
"""
B5.6 阶段：对比基线模型
添加 GRU, MLP, SVR, CNN (1D Conv) 进行性能对比
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_solar_data
from src.data.splitter import split_by_day
from src.data.window import generate_windows
from src.data.daylight import add_daylight_flag, filter_daylight_rows
from src.models.lstm import GlobalLSTM
from src.models.baselines_dl import GRUModel, MLPModel, CNN1DModel
from src.training.trainer import LSTMTrainer, TrainConfig
from src.evaluation.rolling_eval import RollingEvaluator, print_eval_table
from src.utils.seed import set_seed, print_seed_info


def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def samples_to_arrays(samples, nominal_capacity):
    X_list = []
    y_list = []
    for sample in samples:
        X_list.append(sample.X)
        y_list.append(sample.Y)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    
    # 归一化到 p.u.
    X = X / nominal_capacity
    y = y / nominal_capacity
    return X, y


def run_svr(X_train, y_train, X_test, capacity, horizons, horizon_names):
    """运行 SVR (MultiOutput)"""
    print("\n[SVR] 正在训练 Support Vector Regression...")
    
    # Flatten input: (N, Lx, F) -> (N, Lx*F)
    N_train, Lx, F = X_train.shape
    X_train_flat = X_train.reshape(N_train, -1)
    
    N_test = X_test.shape[0]
    X_test_flat = X_test.reshape(N_test, -1)
    
    # 定义模型: 多输出回归包装器 + SVR
    # SVR 训练较慢，这里限制 iter 或使用较小子集调试? 
    # 为了保证效果，还是完整训练，但可能需要一些时间。
    # SVR 对数据规模敏感，如果数据量很大(>10000)，可能非常慢。
    # 这里我们使用默认 RBF 核。
    
    # 注意: SVR 默认没有 early stopping，直接 fit
    svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, epsilon=0.1))
    
    # 为了加快演示速度，若数据量过大可截断 (可选，这里全量)
    print(f"  - Train data shape: {X_train_flat.shape}")
    
    start_time = datetime.now()
    svr.fit(X_train_flat, y_train)
    duration = datetime.now() - start_time
    print(f"  - SVR 训练耗时: {duration}")
    
    # 预测
    print("  - 正在预测...")
    y_pred_test = svr.predict(X_test_flat)
    
    # 转换为 MW
    y_true_mw = None # Evaluation usually uses samples directly, but here we have arrays
    # But RollingEvaluator expects predict_fn.
    
    return svr


def main():
    print("\n" + "=" * 80)
    print("B5.6 对比实验: LSTM vs GRU vs MLP vs CNN vs SVR")
    print("=" * 80)
    
    # 1. 加载配置
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)
    
    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    model_config = config.get('model', {})
    eval_config = config['evaluation']
    train_config_dict = config.get('training', {})
    
    capacity = data_config['nominal_capacity_mw']
    target_col = data_config['target_column']
    
    # 2. 数据准备 (复用 B1 逻辑)
    print("正在准备数据...")
    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )
    
    daylight_config = config.get('daylight_filter', {'enabled': False})
    daylight_enabled = daylight_config.get('enabled', False)
    daylight_mode = daylight_config.get('mode', 'mask') if daylight_enabled else None
    
    # 白天筛选
    daylight_result = add_daylight_flag(
        df=load_result.df,
        time_column=data_config['time_column'],
        dni_col=daylight_config['dni_col'] if daylight_enabled else "Total solar irradiance (W/m2)",
        threshold=daylight_config['threshold'] if daylight_enabled else 5
    )
    
    # 切分
    split_result = split_by_day(
        df=daylight_result.df,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )
    
    # 生成滑窗 (Mask Mode)
    use_mask_mode = daylight_enabled and daylight_mode == 'mask'
    train_windows = generate_windows(split_result.train_df, data_config['time_column'], target_col, 
                                   window_config['input_length'], window_config['max_horizon'], 
                                   data_config['time_interval_minutes'], daylight_mask_mode=use_mask_mode)
    val_windows = generate_windows(split_result.val_df, data_config['time_column'], target_col, 
                                 window_config['input_length'], window_config['max_horizon'], 
                                 data_config['time_interval_minutes'], daylight_mask_mode=use_mask_mode)
    test_windows = generate_windows(split_result.test_df, data_config['time_column'], target_col, 
                                  window_config['input_length'], window_config['max_horizon'], 
                                  data_config['time_interval_minutes'], daylight_mask_mode=use_mask_mode)
    
    # 转 numpy
    X_train, y_train = samples_to_arrays(train_windows.samples, capacity)
    X_val, y_val = samples_to_arrays(val_windows.samples, capacity)
    X_test, y_test = samples_to_arrays(test_windows.samples, capacity)
    
    # 3. 定义模型列表
    input_len = window_config['input_length']
    output_len = window_config['max_horizon']
    n_features = X_train.shape[2]
    
    models_to_run = {
        "Global LSTM": lambda: GlobalLSTM(input_size=n_features, output_steps=output_len, 
                                         hidden_size=model_config.get('hidden_size', 64),
                                         num_layers=model_config.get('num_layers', 2)),
        
        "GRU": lambda: GRUModel(input_size=n_features, output_steps=output_len,
                               hidden_size=model_config.get('hidden_size', 64),
                               num_layers=model_config.get('num_layers', 2)),
        
        "MLP": lambda: MLPModel(input_size=n_features, input_length=input_len, output_steps=output_len,
                                hidden_size=128, num_layers=3), # MLP 通常需要稍微宽一点或多一层
        
        "CNN-1D": lambda: CNN1DModel(input_size=n_features, input_length=input_len, output_steps=output_len,
                                     kernel_size=3, num_filters=64)
    }
    
    # 结果存储
    comparison_results = []
    
    # 通用参数
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    set_seed(seed)
    
    train_cfg = TrainConfig(
        lr=train_config_dict.get('lr', 1e-3),
        batch_size=train_config_dict.get('batch_size', 256),
        max_epochs=train_config_dict.get('max_epochs', 100), # 1500 数据量很少，100 足够
        patience=10,
        device=device,
        seed=seed
    )
    
    # 4. 循环训练深度学习模型
    final_metrics = {} # Name -> RollingEvalResult
    
    for name, model_factory in models_to_run.items():
        print(f"\n[{name}] 开始训练...")
        model = model_factory()
        
        trainer = LSTMTrainer(
            model=model,
            config=train_cfg,
            nominal_capacity=capacity,
            horizons=eval_config['horizons'],
            horizon_names=eval_config['horizon_names']
        )
        
        # 训练
        train_res = trainer.train(X_train, y_train, X_val, y_val, verbose=False) # 关闭详细 epoch 日志
        print(f"  - 训练完成. Best Epoch: {train_res.best_epoch}, Best Val 4h RMSE: {train_res.best_val_rmse_4h_pu:.4f}")
        
        # 评估
        evaluator = RollingEvaluator(eval_config['horizons'], eval_config['horizon_names'], capacity)
        
        def predict_wrapper(x):
            # x shape (Lx, F)
            x_pu = x.reshape(1, -1, x.shape[-1]) / capacity
            x_tensor = torch.FloatTensor(x_pu).to(device)
            model.eval()
            with torch.no_grad():
                out = model(x_tensor)
            return out.cpu().numpy().flatten() * capacity
            
        res = evaluator.evaluate(test_windows.samples, predict_wrapper)
        final_metrics[name] = res
        print(f"  - Test 4h nRMSE: {res.overall_metrics.nrmse:.4f}")
    
    # 5. 训练 SVR
    svr_model = run_svr(X_train, y_train, X_test, capacity, eval_config['horizons'], eval_config['horizon_names'])
    
    # SVR 评估
    evaluator_svr = RollingEvaluator(eval_config['horizons'], eval_config['horizon_names'], capacity)
    def predict_svr_wrapper(x):
        # x shape (Lx, F)
        x_pu = x.reshape(1, -1) / capacity # (1, Lx*F)
        out_pu = svr_model.predict(x_pu)
        return out_pu.flatten() * capacity
        
    res_svr = evaluator_svr.evaluate(test_windows.samples, predict_svr_wrapper)
    final_metrics["SVR"] = res_svr
    
    # 6. 汇总对比表格
    print("\n" + "="*90)
    print(f"{'Model':<15} | {'1h nRMSE':<10} | {'2h nRMSE':<10} | {'4h nRMSE':<10} | {'Overall nRMSE':<12}")
    print("-" * 90)
    
    # 获取主要 horizon key
    h1_key = 4  # 1h 
    h2_key = 8  # 2h
    h4_key = 16 # 4h
    
    for name, res in final_metrics.items():
        # 提取各个 metrics
        m1 = next((h.metrics.nrmse for h in res.horizon_results if h.horizon == h1_key), -1)
        m2 = next((h.metrics.nrmse for h in res.horizon_results if h.horizon == h2_key), -1)
        m4 = next((h.metrics.nrmse for h in res.horizon_results if h.horizon == h4_key), -1)
        overall = res.overall_metrics.nrmse
        
        print(f"{name:<15} | {m1:<10.4f} | {m2:<10.4f} | {m4:<10.4f} | {overall:<12.4f}")
    
    print("="*90)
    print(f"对比实验完成. 结果显式各模型在相同测试集上的性能差异。")


if __name__ == "__main__":
    main()
