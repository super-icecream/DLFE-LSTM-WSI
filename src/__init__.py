"""DLFE-LSTM-WSI Core Package
================================

该包聚合了项目的核心模块（数据处理、特征工程、模型、训练、评估与工具），
并提供统一的初始化入口，便于按照《DLFE-LSTM-WSI-4 - 最终方案总结》中的
整体架构搭建完整的光伏功率预测流水线。

主要能力：
    * 加载并校验项目配置
    * 准备日志、检查点等运行基础设施
    * 运行环境检测（GPU / 字体 / 依赖）
    * 暴露各功能子模块的核心类 / 函数

典型用法::

    from src import initialize_project

    runtime = initialize_project()
    cfg = runtime["config"]
    logger = runtime["logger"]
    env_status = runtime["environment"]

    # 后续可使用 DataLoader / DPSR / ModelBuilder 等组件完成全流程
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Any

from .data_processing.data_loader import DataLoader
from .data_processing.data_splitter import DataSplitter
from .data_processing.preprocessor import Preprocessor
from .data_processing.vmd_decomposer import VMDDecomposer
from .feature_engineering.weather_classifier import WeatherClassifier
from .feature_engineering.dpsr import DPSR
from .feature_engineering.dlfe import DLFE
from .models.lstm_model import LSTMPredictor
from .models.model_builder import ModelBuilder
from .models.multi_weather_model import MultiWeatherModel
from .training.trainer import GPUOptimizedTrainer
from .training.validator import Validator, MultiModelValidator
from .training.adaptive_optimizer import AdaptiveOptimizer
from .evaluation import (
    ModelEvaluator,
    PerformanceMetrics,
    PerformanceVisualizer,
    MetricsResult,
    export_metrics_bundle,
    export_weather_distribution,
    generate_markdown_report,
)
from .utils.config_loader import ConfigLoader
from .utils.logger import get_logger
from .utils.checkpoint import CheckpointManager


__all__ = [
    # 数据处理
    "DataLoader",
    "DataSplitter",
    "Preprocessor",
    "VMDDecomposer",
    # 特征工程
    "WeatherClassifier",
    "DPSR",
    "DLFE",
    # 模型
    "LSTMPredictor",
    "ModelBuilder",
    "MultiWeatherModel",
    # 训练与自适应
    "GPUOptimizedTrainer",
    "Validator",
    "MultiModelValidator",
    "AdaptiveOptimizer",
    # 评估
    "ModelEvaluator",
    "PerformanceMetrics",
    "PerformanceVisualizer",
    "MetricsResult",
    "export_metrics_bundle",
    "export_weather_distribution",
    "generate_markdown_report",
    # 初始化与工具
    "initialize_project",
    "load_project_config",
    "get_package_metadata",
    "get_evaluation_config",
]


__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "Dynamic Locally Featured Embedding + LSTM Weather-State Inference platform"


# 路径常量 --------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def get_package_metadata() -> Dict[str, str]:
    """返回包的元信息。"""

    return {
        "name": "DLFE-LSTM-WSI",
        "version": __version__,
        "author": __author__,
        "description": __description__,
    }


def load_project_config(
    config_path: Optional[str | Path] = None,
    *,
    override_with_env: bool = True,
    validate: bool = True,
) -> Dict[str, Any]:
    """加载并校验项目配置。"""

    loader = ConfigLoader(config_dir=str(PROJECT_ROOT / "config"))

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    return loader.load(path, override_with_env=override_with_env, validate=validate)


def _ensure_runtime_directories(config: Dict[str, Any]) -> None:
    """根据配置创建必要的运行目录。"""

    experiments_root = Path(config.get("project", {}).get("experiments_dir", PROJECT_ROOT / "experiments"))
    results_dir = Path(config.get("evaluation", {}).get("output", experiments_root / "results"))
    logs_dir = Path(config.get("project", {}).get("log_dir", PROJECT_ROOT / "experiments" / "logs"))
    checkpoints_dir = Path(config.get("project", {}).get("checkpoint_dir", experiments_root / "checkpoints"))

    for directory in {experiments_root, results_dir, logs_dir, checkpoints_dir}:
        directory.mkdir(parents=True, exist_ok=True)


def initialize_project(
    *,
    config_path: Optional[str | Path] = None,
    log_level: str = "INFO",
    create_dirs: bool = True,
    checkpoint_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """项目统一初始化入口。

    返回的字典包含配置、记录器、检查点管理器、评估配置以及运行环境状况。
    """

    config = load_project_config(config_path)

    if create_dirs:
        _ensure_runtime_directories(config)

    logger = get_logger(
        name=config.get("project", {}).get("name", "DLFE-LSTM-WSI"),
        log_dir=str(config.get("project", {}).get("log_dir", PROJECT_ROOT / "experiments" / "logs")),
        log_level=log_level,
    )

    checkpoint_dir = config.get("project", {}).get("checkpoint_dir", PROJECT_ROOT / "experiments" / "checkpoints")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(checkpoint_dir),
        **(checkpoint_kwargs or {}),
    )

    device = config.get("project", {}).get("device", "cuda:")
    evaluation_cfg = get_evaluation_config(device=device.split(":")[0] if ":" in device else device)
    env_status = check_environment(verbose=False)

    logger.info("项目初始化完成", extra={"config_path": str(config_path or DEFAULT_CONFIG_PATH)})

    return {
        "config": config,
        "logger": logger,
        "checkpoint_manager": checkpoint_manager,
        "evaluation_config": evaluation_cfg,
        "environment": env_status,
    }


