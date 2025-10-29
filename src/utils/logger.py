# -*- coding: utf-8 -*-
"""
日志工具模块
提供统一的日志记录、TensorBoard集成和实验追踪功能
"""

import os
import sys
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Union, List
import threading
from queue import Queue
import traceback

import numpy as np
import torch


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""
    
    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m', # 紫色
    }
    RESET = '\033[0m'
    
    def format(self, record):
        # 添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
        
        # 格式化消息
        formatted = super().format(record)
        
        # 对特定关键字添加颜色
        keywords = {
            'GPU': '\033[93m',      # 亮黄色
            'CUDA': '\033[93m',
            'Epoch': '\033[94m',    # 亮蓝色
            'Loss': '\033[91m',     # 亮红色
            'Accuracy': '\033[92m', # 亮绿色
            'RMSE': '\033[91m',
            'MAE': '\033[91m',
        }
        
        for keyword, color in keywords.items():
            formatted = formatted.replace(keyword, f"{color}{keyword}{self.RESET}")
        
        return formatted


class TensorBoardLogger:
    """TensorBoard日志记录器"""
    
    def __init__(self, log_dir: str = './experiments/runs'):
        """
        初始化TensorBoard记录器
        
        Args:
            log_dir: TensorBoard日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard未安装，禁用TensorBoard日志")
            self.writer = None
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """记录多个标量"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: int):
        """记录直方图"""
        if self.enabled:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """记录图像"""
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_model_graph(self, model: torch.nn.Module, input_sample: torch.Tensor):
        """记录模型计算图"""
        if self.enabled:
            self.writer.add_graph(model, input_sample)
    
    def log_text(self, tag: str, text: str, step: int):
        """记录文本"""
        if self.enabled:
            self.writer.add_text(tag, text, step)
    
    def flush(self):
        """刷新缓冲区"""
        if self.enabled:
            self.writer.flush()
    
    def close(self):
        """关闭记录器"""
        if self.enabled:
            self.writer.close()


class ExperimentLogger:
    """
    实验日志管理器
    
    功能：
    - 多级日志记录
    - 文件和控制台输出
    - TensorBoard集成
    - 实验元数据追踪
    - 异步日志写入
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """单例模式"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
            return cls._instance
    
    def __init__(self, 
                 name: str = 'DLFE-LSTM-WSI',
                 log_dir: str = './experiments/logs',
                 log_level: str = 'INFO',
                 use_tensorboard: bool = True,
                 async_logging: bool = False):
        """
        初始化日志管理器
        
        Args:
            name: 日志器名称
            log_dir: 日志目录
            log_level: 日志级别
            use_tensorboard: 是否使用TensorBoard
            async_logging: 是否启用异步日志
        """
        # 避免重复初始化
        if hasattr(self, '_initialized'):
            return
        self._initialized = True
        
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建时间戳目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.log_dir / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.handlers.clear()  # 清除已有处理器
        self.logger.propagate = False  # 不传播到root logger，避免重复输出
        
        # 控制台处理器（简洁输出）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = ColoredFormatter(
            '%(message)s',  # 只显示消息内容，无时间戳和日志级别前缀
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器（详细日志）
        file_handler = logging.FileHandler(
            self.run_dir / 'experiment.log', 
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 错误日志单独文件
        error_handler = logging.FileHandler(
            self.run_dir / 'errors.log',
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # TensorBoard记录器
        if use_tensorboard:
            self.tb_logger = TensorBoardLogger(
                log_dir=str(self.run_dir / 'tensorboard')
            )
        else:
            self.tb_logger = None
        
        # 异步日志队列
        if async_logging:
            self.log_queue = Queue()
            self._start_async_logging()
        else:
            self.log_queue = None
        
        # 实验元数据
        self.metadata = {
            'experiment_name': name,
            'start_time': self.timestamp,
            'log_dir': str(self.run_dir),
            'config': {}
        }
        
        # 记录初始化信息
        self.info(f"实验日志初始化: {name}")
        self.info(f"日志目录: {self.run_dir}")
    
    def _start_async_logging(self):
        """启动异步日志线程"""
        def async_log_worker():
            while True:
                record = self.log_queue.get()
                if record is None:  # 退出信号
                    break
                self.logger.handle(record)
        
        self.log_thread = threading.Thread(target=async_log_worker, daemon=True)
        self.log_thread.start()
    
    def debug(self, message: str, *args, **kwargs):
        """DEBUG级别日志"""
        self._log(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """INFO级别日志"""
        self._log(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """WARNING级别日志"""
        self._log(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, exc_info: bool = False, **kwargs):
        """ERROR级别日志"""
        if exc_info:
            message += f"\n{traceback.format_exc()}"
        self._log(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """CRITICAL级别日志"""
        self._log(logging.CRITICAL, message, *args, **kwargs)
    
    def _log(self, level: int, message: str, *args, **kwargs):
        """内部日志方法
        
        支持标准Python logging格式化：
        - logger.info("Message %s", value)  # 使用 % 格式化
        - logger.info("Message", extra={'key': 'value'})  # 关键字参数
        """
        # 如果有格式化参数，先格式化消息
        if args:
            message = message % args
        
        # 格式化额外参数
        if kwargs:
            extra_info = ' | '.join([f"{k}={v}" for k, v in kwargs.items()])
            message = f"{message} | extra={{{extra_info}}}"
        
        if self.log_queue is not None:
            # 异步日志
            record = self.logger.makeRecord(
                self.logger.name, level, '', 0, 
                message, (), None
            )
            self.log_queue.put(record)
        else:
            # 同步日志
            self.logger.log(level, message)
    
    def log_config(self, config: Dict[str, Any]):
        """记录配置信息"""
        self.metadata['config'] = config
        config_path = self.run_dir / 'config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.info(f"配置已保存至: {config_path}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int, phase: str = 'train'):
        """
        记录评估指标
        
        Args:
            metrics: 指标字典
            step: 当前步数
            phase: 阶段 ('train', 'val', 'test')
        """
        # 日志记录
        metrics_str = ' | '.join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.info(f"[{phase.upper()}] Step {step} | {metrics_str}")
        
        # TensorBoard记录
        if self.tb_logger:
            for name, value in metrics.items():
                self.tb_logger.log_scalar(f'{phase}/{name}', value, step)
    
    def log_epoch_summary(self, 
                         epoch: int,
                         train_metrics: Dict[str, float],
                         val_metrics: Optional[Dict[str, float]] = None,
                         time_elapsed: float = 0):
        """
        记录epoch总结
        
        Args:
            epoch: 当前epoch
            train_metrics: 训练指标
            val_metrics: 验证指标
            time_elapsed: 耗时（秒）
        """
        self.info(f"{'='*60}")
        self.info(f"Epoch {epoch} 完成 | 耗时: {time_elapsed:.2f}秒")
        
        # 训练指标
        train_str = ' | '.join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
        self.info(f"训练指标: {train_str}")
        
        # 验证指标
        if val_metrics:
            val_str = ' | '.join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
            self.info(f"验证指标: {val_str}")
        
        self.info(f"{'='*60}")
    
    def log_model_info(self, model: torch.nn.Module):
        """记录模型信息"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.info(f"模型参数统计:")
        self.info(f"  总参数量: {total_params:,}")
        self.info(f"  可训练参数: {trainable_params:,}")
        self.info(f"  不可训练参数: {total_params - trainable_params:,}")
        
        # 保存模型结构
        model_path = self.run_dir / 'model_architecture.txt'
        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(str(model))
    
    def log_gpu_memory(self):
        """记录GPU内存使用"""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                self.debug(
                    f"GPU {i} 内存: "
                    f"已分配={allocated:.2f}GB, "
                    f"已预留={reserved:.2f}GB"
                )
    
    def save_checkpoint(self, 
                       state: Dict[str, Any],
                       filename: str = 'checkpoint.pth'):
        """保存检查点"""
        checkpoint_path = self.run_dir / filename
        torch.save(state, checkpoint_path)
        self.info(f"检查点已保存: {checkpoint_path}")
        return checkpoint_path
    
    def close(self):
        """关闭日志器"""
        # 保存元数据
        metadata_path = self.run_dir / 'metadata.json'
        self.metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        # 关闭TensorBoard
        if self.tb_logger:
            self.tb_logger.close()
        
        # 停止异步日志
        if self.log_queue is not None:
            self.log_queue.put(None)
            self.log_thread.join()
        
        self.info("实验日志已关闭")


def get_logger(name: Optional[str] = None, **kwargs) -> ExperimentLogger:
    """获取全局日志器，支持自定义目录和设置"""
    if name is None:
        name = 'DLFE-LSTM-WSI'

    log_dir = kwargs.pop('log_dir', './experiments/logs')
    log_level = kwargs.pop('log_level', 'INFO')
    return ExperimentLogger(name=name, log_dir=log_dir, log_level=log_level, **kwargs)


# 单元测试
if __name__ == "__main__":
    # 创建日志器
    logger = ExperimentLogger(
        name='test_experiment',
        log_level='DEBUG',
        use_tensorboard=True
    )
    
    # 测试不同级别日志
    logger.debug("调试信息")
    logger.info("一般信息")
    logger.warning("警告信息")
    logger.error("错误信息", exc_info=True)
    
    # 测试指标记录
    metrics = {
        'loss': 0.5,
        'accuracy': 0.95,
        'RMSE': 12.3,
        'MAE': 8.7
    }
    logger.log_metrics(metrics, step=10, phase='train')
    
    # 测试epoch总结
    logger.log_epoch_summary(
        epoch=1,
        train_metrics={'loss': 0.5, 'acc': 0.9},
        val_metrics={'loss': 0.6, 'acc': 0.85},
        time_elapsed=120.5
    )
    
    # 测试GPU内存记录
    logger.log_gpu_memory()
    
    # 关闭日志器
    logger.close()