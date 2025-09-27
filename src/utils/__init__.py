# -*- coding: utf-8 -*-
"""
DLFE-LSTM-WSI 工具模块
提供配置管理、日志记录和检查点管理等基础功能

Components:
    ConfigLoader: 配置文件加载和管理
    ExperimentLogger: 实验日志和TensorBoard集成
    CheckpointManager: 模型检查点管理
"""

from .config_loader import ConfigLoader, ConfigSchema
from .logger import ExperimentLogger, TensorBoardLogger, get_logger
from .checkpoint import CheckpointManager

# 版本信息
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Development Team"

# 导出的公共接口
__all__ = [
    'ConfigLoader',
    'ConfigSchema',
    'ExperimentLogger',
    'TensorBoardLogger',
    'get_logger',
    'CheckpointManager',
    'setup_experiment',
    'load_config',
    'create_logger',
    'create_checkpoint_manager'
]


def setup_experiment(experiment_name: str,
                    config_path: str = './config/config.yaml',
                    log_level: str = 'INFO') -> tuple:
    """
    快速设置实验环境
    
    Args:
        experiment_name: 实验名称
        config_path: 配置文件路径
        log_level: 日志级别
        
    Returns:
        (config, logger, ckpt_manager) 元组
    """
    # 加载配置
    config_loader = ConfigLoader()
    config = config_loader.load(config_path)
    
    # 创建日志器
    logger = ExperimentLogger(
        name=experiment_name,
        log_level=log_level,
        use_tensorboard=True
    )
    logger.log_config(config)
    
    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        checkpoint_dir=f'./experiments/checkpoints/{experiment_name}',
        monitor_metric=config.get('evaluation', {}).get('monitor_metric', 'val_loss')
    )
    
    return config, logger, ckpt_manager


def load_config(config_path: str = './config/config.yaml') -> dict:
    """快速加载配置文件"""
    loader = ConfigLoader()
    return loader.load(config_path)


def create_logger(name: str = 'DLFE-LSTM-WSI', **kwargs) -> ExperimentLogger:
    """快速创建日志器"""
    return ExperimentLogger(name=name, **kwargs)


def create_checkpoint_manager(checkpoint_dir: str = './experiments/checkpoints',
                             **kwargs) -> CheckpointManager:
    """快速创建检查点管理器"""
    return CheckpointManager(checkpoint_dir=checkpoint_dir, **kwargs)


# 模块初始化检查
def _check_dependencies():
    """检查必要的依赖"""
    import importlib
    
    dependencies = {
        'torch': '深度学习框架',
        'yaml': '配置文件解析',
        'numpy': '数值计算'
    }
    
    missing = []
    for module, description in dependencies.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(f"{module} ({description})")
    
    if missing:
        print("⚠️ 缺少以下依赖:")
        for dep in missing:
            print(f"  - {dep}")
        print("\n请运行: pip install torch pyyaml numpy")
    else:
        print("✅ 工具模块初始化成功")


# 执行依赖检查
if __name__ != '__main__':
    _check_dependencies()