# -*- coding: utf-8 -*-
"""
配置加载器模块
负责加载、验证和管理YAML配置文件
支持环境变量替换和配置合并功能
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime
import copy

logger = logging.getLogger(__name__)


@dataclass
class ConfigSchema:
    """配置模式验证类"""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, type] = field(default_factory=dict)
    field_ranges: Dict[str, tuple] = field(default_factory=dict)


class ConfigLoader:
    """
    配置加载器
    
    功能：
    - 加载YAML/JSON配置文件
    - 支持环境变量替换
    - 配置验证和类型检查
    - 配置合并和覆盖
    - 配置缓存管理
    """
    
    def __init__(self, 
                 config_dir: str = './config',
                 env_prefix: str = 'DLFE',
                 cache_enabled: bool = True):
        """
        初始化配置加载器
        
        Args:
            config_dir: 配置文件目录
            env_prefix: 环境变量前缀
            cache_enabled: 是否启用配置缓存
        """
        self.config_dir = Path(config_dir)
        self.env_prefix = env_prefix
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, Dict] = {}
        
        # 确保配置目录存在
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义默认配置模式
        self.schema = self._define_schema()
    
    def _define_schema(self) -> ConfigSchema:
        """定义配置文件模式"""
        schema = ConfigSchema()
        
        # 必需字段
        schema.required_fields = [
            'project.name',
            'project.version',
            'data.dataset',
            'model.lstm.hidden_sizes',
            'training.batch_size',
            'training.epochs'
        ]
        
        # 字段类型
        schema.field_types = {
            'project.name': str,
            'project.version': str,
            'data.train_ratio': float,
            'data.val_ratio': float,
            'data.test_ratio': float,
            'model.lstm.hidden_sizes': list,
            'model.lstm.dropout_rates': list,
            'training.batch_size': int,
            'training.epochs': int,
            'training.learning_rate': float
        }
        
        # 字段范围
        schema.field_ranges = {
            'data.train_ratio': (0.0, 1.0),
            'data.val_ratio': (0.0, 1.0),
            'data.test_ratio': (0.0, 1.0),
            'training.batch_size': (1, 1024),
            'training.epochs': (1, 10000),
            'training.learning_rate': (1e-6, 1.0)
        }
        
        return schema
    
    def load(self, 
             config_path: Union[str, Path],
             override_with_env: bool = True,
             validate: bool = True) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            override_with_env: 是否用环境变量覆盖
            validate: 是否验证配置
            
        Returns:
            配置字典
        """
        config_path = Path(config_path)
        
        # 检查缓存
        cache_key = str(config_path.absolute())
        if self.cache_enabled and cache_key in self._cache:
            logger.debug(f"从缓存加载配置: {cache_key}")
            return copy.deepcopy(self._cache[cache_key])
        
        # 加载配置文件
        if not config_path.exists():
            # 尝试在配置目录中查找
            config_path = self.config_dir / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        logger.info(f"加载配置文件: {config_path}")
        
        # 根据文件扩展名选择加载器
        if config_path.suffix in ['.yaml', '.yml']:
            config = self._load_yaml(config_path)
        elif config_path.suffix == '.json':
            config = self._load_json(config_path)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 环境变量替换
        if override_with_env:
            config = self._override_with_env(config)
        
        # 配置验证
        if validate:
            self.validate(config)
        
        # 缓存配置
        if self.cache_enabled:
            self._cache[cache_key] = copy.deepcopy(config)
        
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """加载YAML文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """加载JSON文件"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用环境变量覆盖配置
        
        环境变量格式: DLFE_SECTION_KEY=value
        例如: DLFE_TRAINING_BATCH_SIZE=128
        """
        def update_nested_dict(d: dict, keys: list, value: Any):
            """更新嵌套字典"""
            key = keys[0]
            if len(keys) == 1:
                # 尝试类型转换
                if key in d and isinstance(d[key], bool):
                    value = value.lower() in ['true', '1', 'yes']
                elif key in d and isinstance(d[key], int):
                    value = int(value)
                elif key in d and isinstance(d[key], float):
                    value = float(value)
                d[key] = value
            else:
                if key not in d:
                    d[key] = {}
                update_nested_dict(d[key], keys[1:], value)
        
        # 扫描环境变量
        for env_key, env_value in os.environ.items():
            if env_key.startswith(self.env_prefix + '_'):
                # 解析环境变量键
                config_keys = env_key[len(self.env_prefix)+1:].lower().split('_')
                try:
                    update_nested_dict(config, config_keys, env_value)
                    logger.debug(f"环境变量覆盖: {env_key}={env_value}")
                except Exception as e:
                    logger.warning(f"环境变量覆盖失败: {env_key}={env_value}, 错误: {e}")
        
        return config
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        验证配置
        
        Args:
            config: 配置字典
            
        Returns:
            是否验证通过
            
        Raises:
            ValueError: 配置验证失败
        """
        def get_nested_value(d: dict, keys: str):
            """获取嵌套字典值"""
            keys_list = keys.split('.')
            value = d
            for key in keys_list:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        
        # 检查必需字段
        for field in self.schema.required_fields:
            value = get_nested_value(config, field)
            if value is None:
                raise ValueError(f"缺少必需配置字段: {field}")
        
        # 检查字段类型
        for field, expected_type in self.schema.field_types.items():
            value = get_nested_value(config, field)
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(
                    f"配置字段类型错误: {field}, "
                    f"期望 {expected_type.__name__}, "
                    f"实际 {type(value).__name__}"
                )
        
        # 检查字段范围
        for field, (min_val, max_val) in self.schema.field_ranges.items():
            value = get_nested_value(config, field)
            if value is not None:
                if isinstance(value, (int, float)):
                    if not (min_val <= value <= max_val):
                        raise ValueError(
                            f"配置字段超出范围: {field}={value}, "
                            f"有效范围: [{min_val}, {max_val}]"
                        )
        
        # 特殊验证：数据集划分比例
        train_ratio = get_nested_value(config, 'data.train_ratio')
        val_ratio = get_nested_value(config, 'data.val_ratio')
        test_ratio = get_nested_value(config, 'data.test_ratio')
        
        if train_ratio and val_ratio and test_ratio:
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 1e-6:
                raise ValueError(
                    f"数据集划分比例之和必须为1.0, 当前: {total_ratio}"
                )
        
        logger.info("配置验证通过")
        return True
    
    def merge_configs(self, 
                     base_config: Dict[str, Any],
                     *override_configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并多个配置
        
        Args:
            base_config: 基础配置
            override_configs: 覆盖配置（可多个）
            
        Returns:
            合并后的配置
        """
        def deep_merge(base: dict, override: dict) -> dict:
            """深度合并字典"""
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = copy.deepcopy(value)
            return result
        
        merged = copy.deepcopy(base_config)
        for override in override_configs:
            merged = deep_merge(merged, override)
        
        return merged
    
    def save(self, 
             config: Dict[str, Any],
             save_path: Union[str, Path],
             format: str = 'yaml'):
        """
        保存配置文件
        
        Args:
            config: 配置字典
            save_path: 保存路径
            format: 保存格式 ('yaml' 或 'json')
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            if format == 'yaml':
                yaml.dump(config, f, default_flow_style=False, 
                         allow_unicode=True, sort_keys=False)
            elif format == 'json':
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"配置已保存至: {save_path}")
    
    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        default_config = {
            'project': {
                'name': 'DLFE-LSTM-WSI',
                'version': '1.0.0',
                'device': 'cuda:0'
            },
            'data': {
                'dataset': '甘肃光伏功率预测数据集',
                'train_ratio': 0.7,
                'val_ratio': 0.2,
                'test_ratio': 0.1,
                'sequence_length': 30,
                'sampling_rate': '5T'
            },
            'preprocessing': {
                'vmd': {
                    'n_modes': 5,
                    'alpha': 2000,
                    'tolerance': 1e-6
                },
                'normalization': {
                    'method': 'minmax',
                    'feature_range': [0, 1]
                }
            },
            'feature_engineering': {
                'ci_thresholds': [0.2, 0.6],
                'wsi_thresholds': [0.3, 0.7],
                'fusion_weights': {
                    'ci': 0.7,
                    'wsi': 0.3
                },
                'dpsr': {
                    'embedding_dim': 30,
                    'neighborhood_size': 10
                },
                'dlfe': {
                    'target_dim': 30,
                    'alpha': 2**-10,
                    'beta': 0.1
                }
            },
            'model': {
                'lstm': {
                    'hidden_sizes': [100, 50],
                    'dropout_rates': [0.3, 0.2],
                    'activation': 'tanh'
                }
            },
            'training': {
                'batch_size': 64,
                'epochs': 100,
                'learning_rate': 0.05,
                'optimizer': 'SGDM',
                'scheduler': {
                    'type': 'StepLR',
                    'step_size': 30,
                    'gamma': 0.1
                }
            },
            'adaptive': {
                'error_window': 100,
                'trigger_thresholds': {
                    'level1': 0.1,
                    'level2': 0.15,
                    'level3': 0.05,
                    'level4': 5
                }
            },
            'evaluation': {
                'metrics': ['RMSE', 'MAE', 'NRMSE', 'R2', 'MAPE'],
                'horizons': [1, 3, 6],
                'confidence_level': 0.95
            }
        }
        
        return default_config
    
    def clear_cache(self):
        """清空配置缓存"""
        self._cache.clear()
        logger.debug("配置缓存已清空")


# 单元测试
if __name__ == "__main__":
    # 创建配置加载器
    loader = ConfigLoader()
    
    # 获取默认配置
    default_config = loader.get_default_config()
    print("默认配置:")
    print(yaml.dump(default_config, default_flow_style=False))
    
    # 保存默认配置
    loader.save(default_config, './config/default.yaml')
    
    # 加载配置
    config = loader.load('./config/default.yaml')
    print("\n加载的配置:")
    print(f"项目名称: {config['project']['name']}")
    print(f"批大小: {config['training']['batch_size']}")
    
    # 合并配置
    override = {'training': {'batch_size': 128}}
    merged = loader.merge_configs(config, override)
    print(f"\n合并后的批大小: {merged['training']['batch_size']}")