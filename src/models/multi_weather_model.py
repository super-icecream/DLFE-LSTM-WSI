"""
DLFE-LSTM-WSI项目 - GPU并行的多天气模型管理器
严格按照指导文件第453-730行实现

GPU优化特性：
- 模型并行推理
- GPU内存池管理
- 异步模型更新
- CUDA图优化
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from .model_builder import ModelBuilder


class MultiWeatherModel:
    """
    GPU优化的多天气模型管理器
    - 模型并行推理
    - GPU内存池管理
    - 异步模型更新
    """

    def __init__(self, model_builder: ModelBuilder,
                 use_model_parallel: bool = False):
        """
        参数：
        - use_model_parallel: 是否使用模型并行（多个模型在不同GPU）
        """
        self.model_builder = model_builder
        self.device = model_builder.device
        self.gpu_config = model_builder.gpu_config
        self.use_model_parallel = use_model_parallel

        # 创建三个天气子模型
        self.models = self._create_models()

        # GPU内存池（避免频繁分配）
        if self.gpu_config['available']:
            self._setup_memory_pool()

    def _create_models(self) -> Dict[str, nn.Module]:
        """创建GPU优化的模型"""
        models = {}

        if self.use_model_parallel and self.gpu_config['device_count'] >= 3:
            # 每个模型在不同GPU上
            for i, weather in enumerate(['sunny', 'cloudy', 'overcast']):
                with torch.cuda.device(i):
                    model = self.model_builder.build_model(weather)
                    model = model.to(f'cuda:{i}')
                    models[weather] = model
                print(f"{weather}模型部署在GPU {i}")
        else:
            # 所有模型在同一GPU
            for weather in ['sunny', 'cloudy', 'overcast']:
                models[weather] = self.model_builder.build_model(weather)

        return models

    def _setup_memory_pool(self):
        """设置GPU内存池优化"""
        # 预分配内存池减少碎片
        torch.cuda.set_per_process_memory_fraction(0.9)

        # 启用内存缓存
        torch.cuda.empty_cache()

        # 设置cuDNN确定性（可重现但稍慢）
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    @torch.cuda.amp.autocast()  # 混合精度推理
    def predict_gpu_optimized(self,
                             features: torch.Tensor,
                             weather_type: Optional[int] = None,
                             weather_prob: Optional[torch.Tensor] = None,
                             use_ensemble: bool = False) -> torch.Tensor:
        """
        GPU优化的预测

        关键优化：
        - 混合精度推理
        - CUDA图加速（静态图优化）
        - 异步执行
        """
        # 确保输入在GPU上
        if not features.is_cuda:
            features = features.to(self.device, non_blocking=True)

        if weather_prob is not None and not weather_prob.is_cuda:
            weather_prob = weather_prob.to(self.device, non_blocking=True)

        if use_ensemble:
            return self._ensemble_predict_parallel(features, weather_prob)
        else:
            model_name = self._get_model_name(weather_type)
            model = self.models[model_name]

            # 使用CUDA图加速（对于固定大小的输入）
            if not hasattr(self, f'_graph_{model_name}'):
                self._create_cuda_graph(model, model_name, features)

            return self._run_cuda_graph(model_name, features)

    def _get_model_name(self, weather_type: int) -> str:
        """根据天气类型获取模型名称"""
        weather_map = {0: 'sunny', 1: 'cloudy', 2: 'overcast'}
        return weather_map.get(weather_type, 'sunny')

    def _ensemble_predict_parallel(self,
                                  features: torch.Tensor,
                                  weather_prob: torch.Tensor) -> torch.Tensor:
        """
        并行集成预测（GPU优化）
        使用CUDA流实现真正的并行
        """
        # 创建CUDA流
        streams = [torch.cuda.Stream() for _ in range(3)]
        predictions = []

        model_names = ['sunny', 'cloudy', 'overcast']

        for i, (stream, model_name) in enumerate(zip(streams, model_names)):
            with torch.cuda.stream(stream):
                model = self.models[model_name]
                model.eval()

                with torch.no_grad():
                    pred, _ = model(features)
                    # 在GPU上加权
                    weighted_pred = pred * weather_prob[:, i:i+1]
                    predictions.append(weighted_pred)

        # 同步所有流
        for stream in streams:
            stream.synchronize()

        # GPU上求和
        ensemble_prediction = torch.stack(predictions).sum(dim=0)

        return ensemble_prediction

    def _create_cuda_graph(self, model: nn.Module,
                          model_name: str,
                          sample_input: torch.Tensor):
        """创建CUDA图以加速固定大小的推理"""
        # 预热
        model.eval()
        for _ in range(3):
            with torch.no_grad():
                _ = model(sample_input)

        # 记录CUDA图
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_input = sample_input.clone()
            static_output = model(static_input)[0]

        setattr(self, f'_graph_{model_name}', graph)
        setattr(self, f'_static_input_{model_name}', static_input)
        setattr(self, f'_static_output_{model_name}', static_output)

    def _run_cuda_graph(self, model_name: str,
                       features: torch.Tensor) -> torch.Tensor:
        """运行CUDA图"""
        graph = getattr(self, f'_graph_{model_name}')
        static_input = getattr(self, f'_static_input_{model_name}')
        static_output = getattr(self, f'_static_output_{model_name}')

        static_input.copy_(features)
        graph.replay()

        return static_output.clone()

    def train_gpu_optimized(self, model_type: int,
                           train_loader: DataLoader,
                           val_loader: DataLoader,
                           epochs: int = 100):
        """
        GPU优化的训练流程

        关键优化：
        - 混合精度训练
        - 梯度累积
        - 异步数据加载
        """
        model_name = self._get_model_name(model_type)
        model = self.models[model_name]

        # 优化器和调度器
        optimizer = self.model_builder.create_optimizer_gpu(model)
        scaler = torch.cuda.amp.GradScaler()

        # 损失函数（在GPU上）
        criterion = nn.MSELoss().to(self.device)

        # 梯度累积设置
        accumulation_steps = 4

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            # 异步预取下一批数据
            data_iter = iter(train_loader)
            next_batch = next(data_iter, None)

            for batch_idx in range(len(train_loader)):
                # 当前批次
                if next_batch is not None:
                    features, targets = next_batch
                    features = features.to(self.device, non_blocking=True)
                    targets = targets.to(self.device, non_blocking=True)

                    # 预取下一批（异步）
                    try:
                        next_batch = next(data_iter)
                    except StopIteration:
                        next_batch = None

                    # 混合精度前向传播
                    with torch.cuda.amp.autocast():
                        predictions, _ = model(features)
                        loss = criterion(predictions, targets)
                        loss = loss / accumulation_steps

                    # 混合精度反向传播
                    scaler.scale(loss).backward()

                    # 梯度累积
                    if (batch_idx + 1) % accumulation_steps == 0:
                        # 梯度裁剪
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                        # 优化器步进
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

                    train_loss += loss.item() * accumulation_steps

            # 验证（GPU上进行）
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for features, targets in val_loader:
                        features = features.to(self.device, non_blocking=True)
                        targets = targets.to(self.device, non_blocking=True)
                        predictions, _ = model(features)
                        loss = criterion(predictions, targets)
                        val_loss += loss.item()

            # 清理GPU缓存
            if epoch % 10 == 0:
                torch.cuda.empty_cache()

            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    def adaptive_update_gpu(self, model_type: int,
                           recent_data: Tuple,
                           recent_errors: torch.Tensor):
        """
        GPU加速的自适应更新
        使用二阶优化方法加速收敛
        """
        model_name = self._get_model_name(model_type)
        model = self.models[model_name]

        # 确保数据在GPU上
        features, targets = recent_data
        if not features.is_cuda:
            features = features.to(self.device, non_blocking=True)
        if not targets.is_cuda:
            targets = targets.to(self.device, non_blocking=True)
        if not recent_errors.is_cuda:
            recent_errors = recent_errors.to(self.device, non_blocking=True)

        # 使用L-BFGS优化器（二阶方法，更快收敛）
        optimizer = torch.optim.LBFGS(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.1,
            max_iter=20
        )

        def closure():
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                predictions, _ = model(features)
                loss = nn.MSELoss()(predictions, targets)
                weighted_loss = (loss * recent_errors).mean()
            weighted_loss.backward()
            return weighted_loss

        # 执行优化
        optimizer.step(closure)