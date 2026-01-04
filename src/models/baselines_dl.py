# -*- coding: utf-8 -*-
"""
基线深度学习模型
1D)
"""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    多层感知机 (MLP) 基线模型
    输入展平后通过全连接层，不考虑时序结构
    """
    
    def __init__(
        self,
        input_size: int,
        input_length: int,
        output_steps: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: 特征维度 (F)
            input_length: 输入序列长度 (Lx)
            output_steps: 输出步长 (Hmax)
            hidden_size: 隐藏层节点数
        """
        super().__init__()
        
        # 输入维度 = Lx * F
        self.flat_input_dim = input_length * input_size
        
        layers = []
        
        # 输入层
        layers.append(nn.Linear(self.flat_input_dim, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
        # 输出层
        layers.append(nn.Linear(hidden_size, output_steps))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, Lx, F)
        Returns:
            (batch, Hmax)
        """
        batch_size = x.size(0)
        # Flatten: (batch, Lx*F)
        x_flat = x.reshape(batch_size, -1)
        return self.net(x_flat)


class GRUModel(nn.Module):
    """
    GRU 基线模型
    结构类似于 GlobalLSTM，只是将 LSTM 换为 GRU
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_steps: int = 16,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.fc = nn.Linear(hidden_size, output_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, Lx, F)
        """
        # GRU 输出: output, h_n
        # output: (batch, Lx, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        _, h_n = self.gru(x)
        
        # 取最后一层最后时刻的隐状态
        last_hidden = h_n[-1]
        
        return self.fc(last_hidden)


class CNN1DModel(nn.Module):
    """
    1D CNN 基线模型
    使用一维卷积提取时序特征
    """
    
    def __init__(
        self,
        input_size: int,
        input_length: int,
        output_steps: int,
        kernel_size: int = 3,
        num_filters: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: 特征维度 (Data channels, e.g. 1)
            input_length: 序列长度 (Lx)
            num_filters: 卷积核数量 (Out channels)
        """
        super().__init__()
        
        # PyTorch Conv1d 输入: (batch, C_in, L_in)
        # 我们的输入 x: (batch, Lx, F) -> 需要转置为 (batch, F, Lx)
        
        self.conv1 = nn.Conv1d(
            in_channels=input_size,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # 不使用池化，保持时间维度或使用 Global Average Pooling
        # 这里为了简单，使用 Flatten + Linear
        
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1
        )
        
        # 计算 Flatten 后的维度: Lx * num_filters
        self.flatten_dim = input_length * num_filters
        
        self.fc = nn.Linear(self.flatten_dim, output_steps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, Lx, F)
        """
        # (batch, F, Lx)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return self.fc(x)
