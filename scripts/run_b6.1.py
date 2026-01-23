# -*- coding: utf-8 -*-
"""
B6.1 阶段集成脚本
用 GRU 模型作为基座，一个脚本跑通完整的消融实验链路

流程:
  B0 (Persistence) → B2-GRU (Global GRU) → B4-GRU (Scratch Experts) → B5.5-GRU (Warm-start Experts)

证明链:
  - GRU 序列建模有效: B2-GRU vs B0
  - 分专家训练有效: B4-GRU vs B2-GRU
  - Warm-start 有效: B5.5-GRU vs B4-GRU

全局配置: 使用 daylight_mode='mask' (B5.6 过渡期优化)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional

from src.data.loader import load_solar_data
from src.data.splitter import split_by_day
from src.data.window import generate_windows, WindowSample
from src.data.daylight import add_daylight_flag, filter_daylight_rows
from src.baselines.persistence import PersistenceBaseline
from src.models.baselines_dl import GRUModel
from src.utils.seed import set_seed, print_seed_info, get_dataloader_generator, worker_init_fn


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class HorizonMetrics:
    """单个 horizon 的指标"""
    mae: float
    rmse: float
    nrmse: float


@dataclass
class StageResult:
    """单个阶段的结果"""
    stage: str              # "B0", "B2-GRU", "B4-GRU", "B5.5-GRU"
    model_name: str         # "Persistence", "Global GRU", etc.
    routing_mode: str       # "-", "hard"
    metrics_1h: HorizonMetrics
    metrics_2h: HorizonMetrics
    metrics_4h: HorizonMetrics
    vs_prev_4h: Optional[float] = None  # 相对上一阶段的提升
    extra_info: Dict = field(default_factory=dict)


@dataclass
class ExpertTrainResult:
    """专家训练结果"""
    cluster_id: int
    best_epoch: int
    best_val_loss: float
    train_samples: int
    val_samples: int


# =============================================================================
# 工具函数
# =============================================================================

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, capacity: float) -> HorizonMetrics:
    """计算单个 horizon 的 MAE/RMSE/nRMSE"""
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    nrmse = rmse / capacity
    return HorizonMetrics(mae=mae, rmse=rmse, nrmse=nrmse)


def compute_improvement(curr_nrmse: float, prev_nrmse: float) -> float:
    """计算相对提升 (负数表示改善)"""
    if prev_nrmse == 0:
        return 0.0
    return (curr_nrmse - prev_nrmse) / prev_nrmse


# =============================================================================
# 状态向量 g_t 构建 (复用 B3/B4/B6 逻辑)
# =============================================================================

def compute_state_vector(sample: WindowSample, feature_names: List[str], capacity: float) -> np.ndarray:
    """计算单个样本的状态向量 g_t"""
    X = sample.X
    g_values = []

    def get_col_idx(partial_name):
        for i, name in enumerate(feature_names):
            if partial_name.lower() in name.lower():
                return i
        return None

    tsi_idx = get_col_idx('Total solar')
    dni_idx = get_col_idx('Direct normal')
    ghi_idx = get_col_idx('Global horizontal')
    temp_idx = get_col_idx('temperature')
    atm_idx = get_col_idx('Atmosphere')
    power_idx = get_col_idx('Power')

    # 辐照统计特征
    irr_indices = [('TSI', tsi_idx), ('DNI', dni_idx), ('GHI', ghi_idx)]
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            g_values.extend([vals.mean(), vals.std(), vals.min(), vals.max()])

    # 辐照变化特征
    for name, idx in irr_indices:
        if idx is not None:
            vals = X[:, idx]
            delta = np.diff(vals)
            if len(delta) > 0:
                g_values.extend([np.mean(np.abs(delta)), np.std(delta), np.max(np.abs(delta))])

    # 功率统计
    if power_idx is not None:
        power_pu = X[:, power_idx] / capacity
        g_values.extend([power_pu.mean(), power_pu.std()])
        delta_power = np.diff(power_pu)
        if len(delta_power) > 0:
            g_values.append(np.mean(np.abs(delta_power)))

    # 环境变量
    if temp_idx is not None:
        temp = X[:, temp_idx]
        g_values.extend([temp.mean(), temp.std()])

    if atm_idx is not None:
        atm = X[:, atm_idx]
        g_values.extend([atm.mean(), atm.std()])

    return np.array(g_values, dtype=np.float32)


def compute_fcm_membership(z: np.ndarray, centers: np.ndarray, m: float = 2.0) -> np.ndarray:
    """计算 FCM 隶属度向量"""
    K = centers.shape[0]
    distances = np.zeros(K)
    for k in range(K):
        diff = z - centers[k]
        distances[k] = np.sum(diff ** 2)
    distances = np.maximum(distances, 1e-10)
    power = 2.0 / (m - 1)
    u = np.zeros(K)
    for k in range(K):
        u[k] = 1.0 / np.sum((distances[k] / distances) ** power)
    return u


# =============================================================================
# GRU 模型训练函数
# =============================================================================

def train_gru_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    lr: float = 0.001,
    batch_size: int = 256,
    max_epochs: int = 100,
    patience: int = 10,
    seed: int = 42,
    model_name: str = "GRU",
    init_model: Optional[GRUModel] = None,
    lr_scale: float = 1.0
) -> Tuple[GRUModel, int, float]:
    """
    训练 GRU 模型

    Args:
        init_model: 用于 warm-start 的初始化模型
        lr_scale: 学习率缩放因子

    Returns:
        (model, best_epoch, best_val_loss)
    """
    input_size = X_train.shape[2]
    output_steps = y_train.shape[1]

    if init_model is not None:
        # Warm-start: 复制模型结构和权重
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=output_steps,
            dropout=dropout
        )
        model.load_state_dict(init_model.state_dict())
    else:
        model = GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_steps=output_steps,
            dropout=dropout
        )

    model.to(device)

    # DataLoader
    generator = get_dataloader_generator(seed)

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        generator=generator, worker_init_fn=worker_init_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    actual_lr = lr * lr_scale
    optimizer = torch.optim.Adam(model.parameters(), lr=actual_lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    no_improve = 0

    print(f"    [{model_name}] Train: {len(X_train)}, Val: {len(X_val)}, LR: {actual_lr:.6f}")

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        val_loss = np.mean(val_losses)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
            if epoch <= 5 or epoch % 10 == 0:
                print(f"      Epoch {epoch:3d}: val_loss={val_loss:.6f} *")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"      Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    print(f"    Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")

    return model, best_epoch, best_val_loss


# =============================================================================
# 评估函数
# =============================================================================

def evaluate_model(
    model,
    test_samples: List[WindowSample],
    X_test: np.ndarray,
    capacity: float,
    device: str,
    horizons: List[int] = [4, 8, 16]
) -> Dict[str, HorizonMetrics]:
    """
    评估模型在测试集上的性能

    Returns:
        Dict: {"1h": HorizonMetrics, "2h": HorizonMetrics, "4h": HorizonMetrics}
    """
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            X_tensor = torch.FloatTensor(X_test[i:i+1]).to(device)
            pred_pu = model(X_tensor).cpu().numpy().flatten()
            pred_mw = pred_pu * capacity
            preds.append(pred_mw)
            targets.append(sample.Y)

    preds = np.array(preds)
    targets = np.array(targets)

    horizon_names = {4: '1h', 8: '2h', 16: '4h'}
    results = {}

    for h in horizons:
        name = horizon_names.get(h, f"{h}step")
        pred_h = preds[:, :h]
        target_h = targets[:, :h]
        results[name] = compute_metrics(target_h.flatten(), pred_h.flatten(), capacity)

    return results


def evaluate_experts_hard(
    experts: Dict[int, GRUModel],
    test_samples: List[WindowSample],
    X_test: np.ndarray,
    labels_test: np.ndarray,
    capacity: float,
    device: str,
    horizons: List[int] = [4, 8, 16]
) -> Dict[str, HorizonMetrics]:
    """
    Hard 路由评估专家模型
    """
    preds = []
    targets = []

    with torch.no_grad():
        for i, sample in enumerate(test_samples):
            k_t = labels_test[i]
            X_tensor = torch.FloatTensor(X_test[i:i+1]).to(device)
            experts[k_t].eval()
            pred_pu = experts[k_t](X_tensor).cpu().numpy().flatten()
            pred_mw = pred_pu * capacity
            preds.append(pred_mw)
            targets.append(sample.Y)

    preds = np.array(preds)
    targets = np.array(targets)

    horizon_names = {4: '1h', 8: '2h', 16: '4h'}
    results = {}

    for h in horizons:
        name = horizon_names.get(h, f"{h}step")
        pred_h = preds[:, :h]
        target_h = targets[:, :h]
        results[name] = compute_metrics(target_h.flatten(), pred_h.flatten(), capacity)

    return results


# =============================================================================
# 打印函数
# =============================================================================

def print_stage_header(stage: str, description: str):
    """打印阶段标题"""
    print(f"\n{'='*70}")
    print(f"[{stage}] {description}")
    print(f"{'='*70}")


def print_summary_table(results: List[StageResult]):
    """打印汇总对比表"""
    print("\n" + "=" * 90)
    print("B6.1 消融实验汇总表 - GRU 基座模型")
    print("=" * 90)

    print(f"\n{'Stage':<12} {'Model':<25} {'1h nRMSE':>12} {'2h nRMSE':>12} {'4h nRMSE':>12} {'vs Prev':>10}")
    print("-" * 90)

    for r in results:
        vs_prev_str = "-" if r.vs_prev_4h is None else f"{r.vs_prev_4h*100:+.1f}%"
        print(f"{r.stage:<12} {r.model_name:<25} "
              f"{r.metrics_1h.nrmse*100:>11.2f}% "
              f"{r.metrics_2h.nrmse*100:>11.2f}% "
              f"{r.metrics_4h.nrmse*100:>11.2f}% "
              f"{vs_prev_str:>10}")

    print("=" * 90)

    # 证明链总结
    if len(results) >= 2:
        print("\n[证明链总结]")

        # B2-GRU vs B0
        if len(results) >= 2:
            imp = compute_improvement(results[1].metrics_4h.nrmse, results[0].metrics_4h.nrmse)
            print(f"  GRU 序列建模有效: {results[1].stage} vs {results[0].stage}, "
                  f"4h nRMSE 降低 {abs(imp)*100:.1f}%")

        # B4-GRU vs B2-GRU
        if len(results) >= 3:
            imp = compute_improvement(results[2].metrics_4h.nrmse, results[1].metrics_4h.nrmse)
            print(f"  分专家训练有效: {results[2].stage} vs {results[1].stage}, "
                  f"4h nRMSE 降低 {abs(imp)*100:.1f}%")

        # B5.5-GRU vs B4-GRU
        if len(results) >= 4:
            imp = compute_improvement(results[3].metrics_4h.nrmse, results[2].metrics_4h.nrmse)
            print(f"  Warm-start 有效: {results[3].stage} vs {results[2].stage}, "
                  f"4h nRMSE 降低 {abs(imp)*100:.1f}%")

        # 累计提升
        if len(results) >= 2:
            total_imp = compute_improvement(results[-1].metrics_4h.nrmse, results[0].metrics_4h.nrmse)
            print(f"  累计提升: {results[-1].stage} vs {results[0].stage}, "
                  f"4h nRMSE 降低 {abs(total_imp)*100:.1f}%")


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("B6.1 消融实验 - GRU 基座模型完整流程")
    print("B0 -> B2-GRU -> B4-GRU -> B5.5-GRU")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_results: List[StageResult] = []

    # =========================================================================
    # [1/10] 加载配置 + 初始化
    # =========================================================================
    print("\n[1/10] 加载配置...")

    config_path = PROJECT_ROOT / "config" / "config.yaml"
    config = load_config(config_path)

    data_config = config['data']
    split_config = config['split']
    window_config = config['window']
    eval_config = config['evaluation']
    output_config = config['output']
    model_config = config.get('model', {})
    train_config = config.get('training', {})
    daylight_config = config.get('daylight_filter', {'enabled': False})

    # 强制使用 mask 模式 (B5.6 过渡期优化)
    daylight_enabled = True
    daylight_mode = 'mask'

    target_col = data_config['target_column']
    capacity = data_config['nominal_capacity_mw']
    n_clusters = 3
    horizons = eval_config['horizons']

    # 随机种子
    seed = train_config.get('seed', 42)
    deterministic = train_config.get('deterministic', True)
    set_seed(seed, deterministic=deterministic)
    print_seed_info(seed, deterministic)

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Daylight mode: {daylight_mode} (mask模式，保留过渡期样本)")

    results_dir = PROJECT_ROOT / output_config['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # [2/10] 数据读取 + 白天筛选
    # =========================================================================
    print("\n[2/10] 数据读取 + 白天筛选...")

    load_result = load_solar_data(
        file_path=data_config['file_path'],
        time_column=data_config['time_column'],
        time_format=data_config['time_format'],
        time_interval_minutes=data_config['time_interval_minutes'],
        project_root=PROJECT_ROOT
    )

    feature_names = list(load_result.df.select_dtypes(include=[np.number]).columns)
    feature_names = [c for c in feature_names if c != 'is_daylight']

    # 添加白天标记
    daylight_result = add_daylight_flag(
        df=load_result.df,
        time_column=data_config['time_column'],
        dni_col=daylight_config.get('dni_col', 'Total solar irradiance (W/m2)'),
        threshold=daylight_config.get('threshold', 5.0)
    )
    df_for_split = daylight_result.df

    print(f"  Total rows: {len(df_for_split)}")
    print(f"  Daylight rows: {daylight_result.stats.daylight_rows}")

    # =========================================================================
    # [3/10] 按天切分 + 滑窗样本生成
    # =========================================================================
    print("\n[3/10] 按天切分 + 滑窗样本生成...")

    split_result = split_by_day(
        df=df_for_split,
        time_column=data_config['time_column'],
        train_ratio=split_config['train_ratio'],
        val_ratio=split_config['val_ratio'],
        test_ratio=split_config['test_ratio']
    )

    print(f"  Train days: {split_result.train_days}, Val days: {split_result.val_days}, Test days: {split_result.test_days}")

    # 生成滑窗样本
    train_window_result = generate_windows(
        df=split_result.train_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True  # mask 模式
    )

    val_window_result = generate_windows(
        df=split_result.val_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )

    test_window_result = generate_windows(
        df=split_result.test_df,
        time_column=data_config['time_column'],
        target_column=target_col,
        input_length=window_config['input_length'],
        max_horizon=window_config['max_horizon'],
        daylight_mask_mode=True
    )

    train_samples = train_window_result.samples
    val_samples = val_window_result.samples
    test_samples = test_window_result.samples

    print(f"  Train samples: {len(train_samples)}, Val samples: {len(val_samples)}, Test samples: {len(test_samples)}")

    # 数据一致性断言
    assert len(test_samples) > 0, "测试集样本数为 0!"

    # =========================================================================
    # [4/10] 加载 Router 组件 (复用 B3 产物)
    # =========================================================================
    print("\n[4/10] 加载 Router 组件...")

    with open(results_dir / "router_config.json", 'r', encoding='utf-8') as f:
        router_config_data = json.load(f)
    cluster_semantic_map = {int(k): v for k, v in router_config_data['cluster_semantic_map'].items()}

    with open(results_dir / "scaler_g.pkl", 'rb') as f:
        scaler_g = pickle.load(f)

    with open(results_dir / "pca.pkl", 'rb') as f:
        pca = pickle.load(f)

    fcm_centers = np.load(results_dir / "fcm_centers.npy")

    with open(results_dir / "scaler_X.pkl", 'rb') as f:
        scaler_X = pickle.load(f)

    print(f"  Loaded: scaler_g.pkl, pca.pkl, fcm_centers.npy, scaler_X.pkl")
    print(f"  Cluster semantic map: {cluster_semantic_map}")

    # =========================================================================
    # [5/10] B0: Persistence Baseline 评估
    # =========================================================================
    print_stage_header("B0", "Persistence Baseline 评估")

    baseline = PersistenceBaseline(max_horizon=window_config['max_horizon'])

    preds_b0 = []
    targets_b0 = []
    for sample in test_samples:
        pred = baseline.predict(sample.X)
        preds_b0.append(pred)
        targets_b0.append(sample.Y)

    preds_b0 = np.array(preds_b0)
    targets_b0 = np.array(targets_b0)

    metrics_b0 = {}
    horizon_names = {4: '1h', 8: '2h', 16: '4h'}
    for h in horizons:
        name = horizon_names.get(h, f"{h}step")
        metrics_b0[name] = compute_metrics(
            targets_b0[:, :h].flatten(),
            preds_b0[:, :h].flatten(),
            capacity
        )

    print(f"  1h nRMSE: {metrics_b0['1h'].nrmse*100:.2f}%")
    print(f"  2h nRMSE: {metrics_b0['2h'].nrmse*100:.2f}%")
    print(f"  4h nRMSE: {metrics_b0['4h'].nrmse*100:.2f}%")

    stage_results.append(StageResult(
        stage="B0",
        model_name="Persistence",
        routing_mode="-",
        metrics_1h=metrics_b0['1h'],
        metrics_2h=metrics_b0['2h'],
        metrics_4h=metrics_b0['4h'],
        vs_prev_4h=None
    ))

    # =========================================================================
    # [6/10] B2-GRU: Global GRU 训练 + 评估
    # =========================================================================
    print_stage_header("B2-GRU", "Global GRU 训练 + 评估")

    # 准备数据 (归一化)
    def prepare_data(samples, scaler_X, capacity):
        X_list, y_list = [], []
        for sample in samples:
            X_scaled = scaler_X.transform(sample.X)
            y_pu = sample.Y / capacity
            X_list.append(X_scaled)
            y_list.append(y_pu)
        return np.array(X_list), np.array(y_list)

    X_train_all, y_train_all = prepare_data(train_samples, scaler_X, capacity)
    X_val_all, y_val_all = prepare_data(val_samples, scaler_X, capacity)
    X_test_all, y_test_all = prepare_data(test_samples, scaler_X, capacity)

    hidden_size = model_config.get('hidden_size', 64)
    num_layers = model_config.get('num_layers', 2)
    dropout = model_config.get('dropout', 0.1)
    lr = train_config.get('lr', 0.001)
    batch_size = train_config.get('batch_size', 256)
    max_epochs = train_config.get('max_epochs', 100)
    patience = train_config.get('patience', 10)

    global_gru, global_best_epoch, global_best_loss = train_gru_model(
        X_train=X_train_all,
        y_train=y_train_all,
        X_val=X_val_all,
        y_val=y_val_all,
        device=device,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        lr=lr,
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        seed=seed,
        model_name="Global GRU"
    )

    # 保存 Global GRU
    torch.save(global_gru.state_dict(), results_dir / "model_global_gru_b6.1.pt")

    # 评估 Global GRU
    metrics_b2 = evaluate_model(global_gru, test_samples, X_test_all, capacity, device, horizons)

    print(f"\n  [评估结果]")
    print(f"  1h nRMSE: {metrics_b2['1h'].nrmse*100:.2f}%")
    print(f"  2h nRMSE: {metrics_b2['2h'].nrmse*100:.2f}%")
    print(f"  4h nRMSE: {metrics_b2['4h'].nrmse*100:.2f}%")

    vs_prev = compute_improvement(metrics_b2['4h'].nrmse, stage_results[-1].metrics_4h.nrmse)

    stage_results.append(StageResult(
        stage="B2-GRU",
        model_name="Global GRU",
        routing_mode="-",
        metrics_1h=metrics_b2['1h'],
        metrics_2h=metrics_b2['2h'],
        metrics_4h=metrics_b2['4h'],
        vs_prev_4h=vs_prev,
        extra_info={"best_epoch": global_best_epoch, "best_val_loss": global_best_loss}
    ))

    # =========================================================================
    # [7/10] 路由分簇 (计算 g_t → FCM 标签)
    # =========================================================================
    print("\n[7/10] 路由分簇...")

    def route_samples(samples, scaler_g, pca, fcm_centers):
        labels = []
        memberships = []
        for sample in samples:
            g_t = compute_state_vector(sample, feature_names, capacity)
            g_scaled = scaler_g.transform(g_t.reshape(1, -1))[0]
            z_t = pca.transform(g_scaled.reshape(1, -1))[0]
            u_t = compute_fcm_membership(z_t, fcm_centers, m=2.0)
            labels.append(np.argmax(u_t))
            memberships.append(u_t)
        return np.array(labels), np.array(memberships)

    labels_train, memb_train = route_samples(train_samples, scaler_g, pca, fcm_centers)
    labels_val, memb_val = route_samples(val_samples, scaler_g, pca, fcm_centers)
    labels_test, memb_test = route_samples(test_samples, scaler_g, pca, fcm_centers)

    print(f"  Cluster distribution (test):")
    for k in range(n_clusters):
        n_k = np.sum(labels_test == k)
        semantic = cluster_semantic_map.get(k, 'unknown')
        print(f"    Cluster {k} ({semantic}): {n_k} samples")

    # =========================================================================
    # [8/10] B4-GRU: Scratch 专家训练 + Hard 路由评估
    # =========================================================================
    print_stage_header("B4-GRU", "Scratch 专家训练 + Hard 路由评估")

    experts_scratch = {}
    expert_train_results_scratch = []

    for k in range(n_clusters):
        print(f"\n  Training Expert {k} ({cluster_semantic_map.get(k, 'unknown')}) from scratch...")

        train_mask = labels_train == k
        val_mask = labels_val == k

        X_train_k = X_train_all[train_mask]
        y_train_k = y_train_all[train_mask]
        X_val_k = X_val_all[val_mask]
        y_val_k = y_val_all[val_mask]

        model_k, best_epoch_k, best_loss_k = train_gru_model(
            X_train=X_train_k,
            y_train=y_train_k,
            X_val=X_val_k,
            y_val=y_val_k,
            device=device,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed + k,
            model_name=f"Expert_{k}_Scratch"
        )

        experts_scratch[k] = model_k
        expert_train_results_scratch.append(ExpertTrainResult(
            cluster_id=k,
            best_epoch=best_epoch_k,
            best_val_loss=best_loss_k,
            train_samples=len(X_train_k),
            val_samples=len(X_val_k)
        ))

        # 保存模型
        torch.save(model_k.state_dict(), results_dir / f"model_expert_{k}_gru_scratch.pt")

    # Hard 路由评估
    metrics_b4 = evaluate_experts_hard(experts_scratch, test_samples, X_test_all, labels_test, capacity, device, horizons)

    print(f"\n  [B4-GRU 评估结果] Hard Routing")
    print(f"  1h nRMSE: {metrics_b4['1h'].nrmse*100:.2f}%")
    print(f"  2h nRMSE: {metrics_b4['2h'].nrmse*100:.2f}%")
    print(f"  4h nRMSE: {metrics_b4['4h'].nrmse*100:.2f}%")

    vs_prev = compute_improvement(metrics_b4['4h'].nrmse, stage_results[-1].metrics_4h.nrmse)

    stage_results.append(StageResult(
        stage="B4-GRU",
        model_name="Experts (Scratch)",
        routing_mode="hard",
        metrics_1h=metrics_b4['1h'],
        metrics_2h=metrics_b4['2h'],
        metrics_4h=metrics_b4['4h'],
        vs_prev_4h=vs_prev
    ))

    # =========================================================================
    # [9/10] B5.5-GRU: Warm-start 专家训练 + Hard 路由评估
    # =========================================================================
    print_stage_header("B5.5-GRU", "Warm-start 专家训练 + Hard 路由评估")

    lr_scale_finetune = config.get('warm_start', {}).get('lr_scale_finetune', 0.1)

    experts_warm = {}
    expert_train_results_warm = []

    for k in range(n_clusters):
        print(f"\n  Training Expert {k} ({cluster_semantic_map.get(k, 'unknown')}) with warm-start...")

        train_mask = labels_train == k
        val_mask = labels_val == k

        X_train_k = X_train_all[train_mask]
        y_train_k = y_train_all[train_mask]
        X_val_k = X_val_all[val_mask]
        y_val_k = y_val_all[val_mask]

        model_k, best_epoch_k, best_loss_k = train_gru_model(
            X_train=X_train_k,
            y_train=y_train_k,
            X_val=X_val_k,
            y_val=y_val_k,
            device=device,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed + k + 100,
            model_name=f"Expert_{k}_Warm",
            init_model=global_gru,  # Warm-start
            lr_scale=lr_scale_finetune
        )

        experts_warm[k] = model_k
        expert_train_results_warm.append(ExpertTrainResult(
            cluster_id=k,
            best_epoch=best_epoch_k,
            best_val_loss=best_loss_k,
            train_samples=len(X_train_k),
            val_samples=len(X_val_k)
        ))

        # 保存模型
        torch.save(model_k.state_dict(), results_dir / f"model_expert_{k}_gru_warm.pt")

    # Hard 路由评估
    metrics_b55 = evaluate_experts_hard(experts_warm, test_samples, X_test_all, labels_test, capacity, device, horizons)

    print(f"\n  [B5.5-GRU 评估结果] Hard Routing")
    print(f"  1h nRMSE: {metrics_b55['1h'].nrmse*100:.2f}%")
    print(f"  2h nRMSE: {metrics_b55['2h'].nrmse*100:.2f}%")
    print(f"  4h nRMSE: {metrics_b55['4h'].nrmse*100:.2f}%")

    vs_prev = compute_improvement(metrics_b55['4h'].nrmse, stage_results[-1].metrics_4h.nrmse)

    stage_results.append(StageResult(
        stage="B5.5-GRU",
        model_name="Experts (Warm-start)",
        routing_mode="hard",
        metrics_1h=metrics_b55['1h'],
        metrics_2h=metrics_b55['2h'],
        metrics_4h=metrics_b55['4h'],
        vs_prev_4h=vs_prev
    ))

    # =========================================================================
    # [10/10] 汇总对比表 + 保存结果
    # =========================================================================
    print("\n[10/10] 汇总对比表 + 保存结果...")

    # 打印汇总表
    print_summary_table(stage_results)

    # 保存 JSON
    summary_data = {
        'timestamp': timestamp,
        'experiment': 'B6.1 GRU Ablation Chain',
        'daylight_mode': daylight_mode,
        'n_test_samples': len(test_samples),
        'stages': []
    }

    for r in stage_results:
        stage_data = {
            'stage': r.stage,
            'model_name': r.model_name,
            'routing_mode': r.routing_mode,
            '1h': {'mae': r.metrics_1h.mae, 'rmse': r.metrics_1h.rmse, 'nrmse': r.metrics_1h.nrmse},
            '2h': {'mae': r.metrics_2h.mae, 'rmse': r.metrics_2h.rmse, 'nrmse': r.metrics_2h.nrmse},
            '4h': {'mae': r.metrics_4h.mae, 'rmse': r.metrics_4h.rmse, 'nrmse': r.metrics_4h.nrmse},
            'vs_prev_4h': r.vs_prev_4h
        }
        if r.extra_info:
            stage_data['extra_info'] = r.extra_info
        summary_data['stages'].append(stage_data)

    json_path = results_dir / "metrics_B6.1_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path.name}")

    # 保存 CSV
    csv_path = results_dir / "metrics_B6.1_summary.csv"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("stage,model,routing,1h_mae,1h_rmse,1h_nrmse,2h_mae,2h_rmse,2h_nrmse,4h_mae,4h_rmse,4h_nrmse,vs_prev_4h\n")
        for r in stage_results:
            vs_prev_str = "" if r.vs_prev_4h is None else f"{r.vs_prev_4h:.4f}"
            f.write(f"{r.stage},{r.model_name},{r.routing_mode},"
                    f"{r.metrics_1h.mae:.4f},{r.metrics_1h.rmse:.4f},{r.metrics_1h.nrmse:.4f},"
                    f"{r.metrics_2h.mae:.4f},{r.metrics_2h.rmse:.4f},{r.metrics_2h.nrmse:.4f},"
                    f"{r.metrics_4h.mae:.4f},{r.metrics_4h.rmse:.4f},{r.metrics_4h.nrmse:.4f},"
                    f"{vs_prev_str}\n")
    print(f"  Saved: {csv_path.name}")

    # 保存详细结果 JSON
    detail_data = {
        'timestamp': timestamp,
        'config': {
            'daylight_mode': daylight_mode,
            'n_clusters': n_clusters,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'lr': lr,
            'lr_scale_finetune': lr_scale_finetune,
            'batch_size': batch_size,
            'patience': patience
        },
        'cluster_semantic_map': cluster_semantic_map,
        'sample_counts': {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples)
        },
        'global_gru_training': {
            'best_epoch': global_best_epoch,
            'best_val_loss': global_best_loss
        },
        'expert_training_scratch': [asdict(r) for r in expert_train_results_scratch],
        'expert_training_warm': [asdict(r) for r in expert_train_results_warm]
    }

    detail_path = results_dir / "metrics_B6.1_detail.json"
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detail_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {detail_path.name}")

    # 保存日志
    logs_dir = PROJECT_ROOT / output_config.get('logs_dir', 'experiments/logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train_log_b6.1_{timestamp}.json"
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'stage': 'B6.1',
            'n_test_samples': len(test_samples),
            'final_4h_nrmse': stage_results[-1].metrics_4h.nrmse,
            'total_improvement': compute_improvement(
                stage_results[-1].metrics_4h.nrmse,
                stage_results[0].metrics_4h.nrmse
            )
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {log_path.name}")

    # 完成
    print("\n" + "=" * 70)
    print("B6.1 消融实验完成!")
    print(f"  最终 4h nRMSE: {stage_results[-1].metrics_4h.nrmse*100:.2f}%")
    total_imp = compute_improvement(stage_results[-1].metrics_4h.nrmse, stage_results[0].metrics_4h.nrmse)
    print(f"  累计提升 (vs Persistence): {abs(total_imp)*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
