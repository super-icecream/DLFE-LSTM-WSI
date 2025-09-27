# -*- coding: utf-8 -*-
"""
模型检查点管理模块
负责模型的保存、加载和版本管理
支持断点续训和最佳模型追踪
"""

import os
import json
import shutil
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import hashlib
import pickle


class CheckpointManager:
    """
    检查点管理器
    
    功能：
    - 模型权重保存/加载
    - 训练状态保存/恢复
    - 最佳模型管理
    - 断点续训支持
    - 模型版本控制
    """
    
    def __init__(self, 
                 checkpoint_dir: str = './experiments/checkpoints',
                 max_checkpoints: int = 5,
                 save_best_only: bool = False,
                 monitor_metric: str = 'val_loss',
                 mode: str = 'min'):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
            max_checkpoints: 最多保留的检查点数量
            save_best_only: 是否只保存最佳模型
            monitor_metric: 监控的指标名称
            mode: 'min'或'max'，指标优化方向
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.monitor_metric = monitor_metric
        self.mode = mode
        
        # 最佳指标追踪
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        # 检查点历史
        self.checkpoint_history = []
        self._load_history()
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       metrics: Dict[str, float],
                       scheduler: Optional[Any] = None,
                       scaler: Optional[Any] = None,
                       extra_info: Optional[Dict] = None) -> Optional[Path]:
        """
        保存检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            epoch: 当前epoch
            metrics: 评估指标
            scheduler: 学习率调度器
            scaler: 混合精度缩放器
            extra_info: 额外信息
            
        Returns:
            保存路径（如果保存）
        """
        # 检查是否需要保存
        current_metric = metrics.get(self.monitor_metric, 0)
        is_best = self._is_better(current_metric, self.best_metric)
        
        if self.save_best_only and not is_best:
            return None
        
        # 更新最佳指标
        if is_best:
            self.best_metric = current_metric
            self.best_epoch = epoch
        
        # 构建检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'best_metric': self.best_metric,
            'best_epoch': self.best_epoch,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': self._get_model_config(model),
        }
        
        # 添加可选组件
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if scaler is not None:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        if extra_info is not None:
            checkpoint['extra_info'] = extra_info
        
        # 计算检查点哈希（用于版本控制）
        checkpoint['hash'] = self._calculate_hash(checkpoint['model_state_dict'])
        
        # 保存检查点
        if is_best:
            save_path = self.checkpoint_dir / 'best_model.pth'
        else:
            save_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pth'
        
        torch.save(checkpoint, save_path)
        
        # 记录历史
        self._add_to_history({
            'path': str(save_path),
            'epoch': epoch,
            'metrics': metrics,
            'is_best': is_best,
            'timestamp': checkpoint['timestamp'],
            'hash': checkpoint['hash']
        })
        
        # 清理旧检查点
        if not self.save_best_only:
            self._cleanup_old_checkpoints()
        
        print(f"{'🌟 最佳' if is_best else '✅'} 检查点已保存: {save_path}")
        
        return save_path
    
    def load_checkpoint(self,
                       model: nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[Any] = None,
                       scaler: Optional[Any] = None,
                       checkpoint_path: Optional[str] = None,
                       load_best: bool = False,
                       strict: bool = True) -> Dict[str, Any]:
        """
        加载检查点
        
        Args:
            model: 模型
            optimizer: 优化器
            scheduler: 学习率调度器
            scaler: 混合精度缩放器
            checkpoint_path: 指定检查点路径
            load_best: 是否加载最佳模型
            strict: 是否严格匹配模型参数
            
        Returns:
            检查点信息字典
        """
        # 确定加载路径
        if checkpoint_path:
            load_path = Path(checkpoint_path)
        elif load_best:
            load_path = self.checkpoint_dir / 'best_model.pth'
        else:
            # 加载最新检查点
            load_path = self._get_latest_checkpoint()
        
        if not load_path or not load_path.exists():
            raise FileNotFoundError(f"检查点不存在: {load_path}")
        
        print(f"📂 加载检查点: {load_path}")
        
        # 加载检查点
        checkpoint = torch.load(load_path, map_location='cpu')
        
        # 恢复模型状态
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # 恢复优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复调度器状态
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复缩放器状态
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 更新最佳指标
        if 'best_metric' in checkpoint:
            self.best_metric = checkpoint['best_metric']
            self.best_epoch = checkpoint.get('best_epoch', 0)
        
        print(f"✅ 检查点加载成功 | Epoch: {checkpoint['epoch']}")
        
        # 验证哈希
        if 'hash' in checkpoint:
            current_hash = self._calculate_hash(checkpoint['model_state_dict'])
            if current_hash != checkpoint['hash']:
                print("⚠️ 警告: 检查点哈希不匹配，模型可能已被修改")
        
        return checkpoint
    
    def save_model_only(self,
                       model: nn.Module,
                       save_path: Optional[str] = None,
                       model_name: Optional[str] = None) -> Path:
        """
        仅保存模型权重
        
        Args:
            model: 模型
            save_path: 保存路径
            model_name: 模型名称
            
        Returns:
            保存路径
        """
        if save_path:
            save_path = Path(save_path)
        else:
            model_name = model_name or 'model'
            save_path = self.checkpoint_dir / f'{model_name}.pth'
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': self._get_model_config(model),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }, save_path)
        
        print(f"💾 模型已保存: {save_path}")
        return save_path
    
    def load_model_only(self,
                       model: nn.Module,
                       model_path: str,
                       strict: bool = True) -> nn.Module:
        """
        仅加载模型权重
        
        Args:
            model: 模型
            model_path: 模型路径
            strict: 是否严格匹配
            
        Returns:
            加载后的模型
        """
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        else:
            # 兼容直接保存的state_dict
            model.load_state_dict(checkpoint, strict=strict)
        
        print(f"✅ 模型加载成功: {model_path}")
        return model
    
    def export_model(self,
                    model: nn.Module,
                    export_path: str,
                    input_shape: Tuple[int, ...],
                    export_format: str = 'onnx'):
        """
        导出模型到其他格式
        
        Args:
            model: 模型
            export_path: 导出路径
            input_shape: 输入形状
            export_format: 导出格式 ('onnx', 'torchscript')
        """
        model.eval()
        dummy_input = torch.randn(1, *input_shape)
        
        if export_format == 'onnx':
            torch.onnx.export(
                model,
                dummy_input,
                export_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'},
                            'output': {0: 'batch_size'}}
            )
            print(f"📦 模型已导出为ONNX: {export_path}")
            
        elif export_format == 'torchscript':
            traced = torch.jit.trace(model, dummy_input)
            traced.save(export_path)
            print(f"📦 模型已导出为TorchScript: {export_path}")
            
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")
    
    def get_checkpoint_info(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        获取检查点信息（不加载权重）
        
        Args:
            checkpoint_path: 检查点路径
            
        Returns:
            检查点信息
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 移除大的张量数据
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'metrics': checkpoint.get('metrics', {}),
            'timestamp': checkpoint.get('timestamp', 'Unknown'),
            'best_metric': checkpoint.get('best_metric', None),
            'best_epoch': checkpoint.get('best_epoch', None),
            'hash': checkpoint.get('hash', None),
            'extra_info': checkpoint.get('extra_info', {})
        }
        
        return info
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        列出所有检查点
        
        Returns:
            检查点信息列表
        """
        checkpoints = []
        
        for ckpt_path in self.checkpoint_dir.glob('*.pth'):
            try:
                info = self.get_checkpoint_info(ckpt_path)
                info['path'] = str(ckpt_path)
                info['size_mb'] = ckpt_path.stat().st_size / 1024 / 1024
                checkpoints.append(info)
            except Exception as e:
                print(f"⚠️ 无法读取检查点 {ckpt_path}: {e}")
        
        # 按epoch排序
        checkpoints.sort(key=lambda x: x.get('epoch', 0))
        
        return checkpoints
    
    def compare_checkpoints(self,
                           ckpt1_path: str,
                           ckpt2_path: str) -> Dict[str, Any]:
        """
        比较两个检查点
        
        Args:
            ckpt1_path: 第一个检查点路径
            ckpt2_path: 第二个检查点路径
            
        Returns:
            比较结果
        """
        info1 = self.get_checkpoint_info(ckpt1_path)
        info2 = self.get_checkpoint_info(ckpt2_path)
        
        comparison = {
            'checkpoint_1': ckpt1_path,
            'checkpoint_2': ckpt2_path,
            'epoch_diff': info2['epoch'] - info1['epoch'],
            'metrics_comparison': {}
        }
        
        # 比较指标
        for metric in set(info1.get('metrics', {}).keys()) | set(info2.get('metrics', {}).keys()):
            val1 = info1.get('metrics', {}).get(metric, None)
            val2 = info2.get('metrics', {}).get(metric, None)
            
            if val1 is not None and val2 is not None:
                diff = val2 - val1
                improvement = diff if metric != 'loss' else -diff
                comparison['metrics_comparison'][metric] = {
                    'ckpt1': val1,
                    'ckpt2': val2,
                    'difference': diff,
                    'improvement': improvement,
                    'improvement_pct': (improvement / abs(val1)) * 100 if val1 != 0 else 0
                }
        
        return comparison
    
    def _is_better(self, current: float, best: float) -> bool:
        """判断当前指标是否更好"""
        if self.mode == 'min':
            return current < best
        else:
            return current > best
    
    def _get_model_config(self, model: nn.Module) -> Dict:
        """获取模型配置"""
        config = {
            'class_name': model.__class__.__name__,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # 尝试获取模型的配置属性
        if hasattr(model, 'config'):
            config['model_config'] = model.config
        
        return config
    
    def _calculate_hash(self, state_dict: Dict) -> str:
        """计算状态字典哈希值"""
        # 将状态字典转换为字节
        state_bytes = pickle.dumps(
            {k: v.cpu().numpy() for k, v in state_dict.items()},
            protocol=pickle.HIGHEST_PROTOCOL
        )
        
        # 计算MD5哈希
        return hashlib.md5(state_bytes).hexdigest()
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """获取最新的检查点"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if not checkpoints:
            return None
        
        # 按修改时间排序
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        return checkpoints[-1]
    
    def _cleanup_old_checkpoints(self):
        """清理旧的检查点"""
        checkpoints = list(self.checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # 按修改时间排序
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # 删除最旧的检查点
        for ckpt in checkpoints[:-self.max_checkpoints]:
            ckpt.unlink()
            print(f"🗑️ 删除旧检查点: {ckpt}")
            
            # 从历史中移除
            self.checkpoint_history = [
                h for h in self.checkpoint_history 
                if h['path'] != str(ckpt)
            ]
    
    def _load_history(self):
        """加载检查点历史"""
        history_path = self.checkpoint_dir / 'checkpoint_history.json'
        
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                self.checkpoint_history = json.load(f)
    
    def _add_to_history(self, record: Dict):
        """添加到历史记录"""
        self.checkpoint_history.append(record)
        
        # 保存历史
        history_path = self.checkpoint_dir / 'checkpoint_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint_history, f, indent=2, ensure_ascii=False)
    
    def clear_all_checkpoints(self, confirm: bool = False):
        """
        清除所有检查点
        
        Args:
            confirm: 确认删除
        """
        if not confirm:
            print("⚠️ 警告: 此操作将删除所有检查点!")
            print("如需继续，请设置 confirm=True")
            return
        
        # 删除所有.pth文件
        for ckpt_path in self.checkpoint_dir.glob('*.pth'):
            ckpt_path.unlink()
            print(f"🗑️ 已删除: {ckpt_path}")
        
        # 清空历史
        self.checkpoint_history = []
        history_path = self.checkpoint_dir / 'checkpoint_history.json'
        if history_path.exists():
            history_path.unlink()
        
        # 重置最佳指标
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.best_epoch = 0
        
        print("✅ 所有检查点已清除")


# 单元测试
if __name__ == "__main__":
    import torch.nn as nn
    import torch.optim as optim
    
    # 创建简单模型用于测试
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # 创建检查点管理器
    ckpt_manager = CheckpointManager(
        checkpoint_dir='./test_checkpoints',
        max_checkpoints=3,
        monitor_metric='val_loss',
        mode='min'
    )
    
    # 创建模型和优化器
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 模拟训练过程
    for epoch in range(5):
        # 模拟训练...
        metrics = {
            'train_loss': 0.5 - epoch * 0.1,
            'val_loss': 0.6 - epoch * 0.08,
            'accuracy': 0.8 + epoch * 0.03
        }
        
        # 保存检查点
        ckpt_manager.save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            extra_info={'learning_rate': 0.001}
        )
    
    # 列出所有检查点
    print("\n所有检查点:")
    for ckpt in ckpt_manager.list_checkpoints():
        print(f"  Epoch {ckpt['epoch']}: {ckpt['metrics']}")
    
    # 加载最佳模型
    print("\n加载最佳模型:")
    checkpoint = ckpt_manager.load_checkpoint(
        model=model,
        optimizer=optimizer,
        load_best=True
    )
    print(f"  最佳Epoch: {checkpoint['epoch']}")
    print(f"  最佳指标: {checkpoint['metrics']}")