"""
DLFE-LSTM-WSI !‹!W
GPU„ñ¦f`!‹ž°

!WÄö:
- LSTMPredictor: GPU„LSTM„KQÜ
- ModelBuilder: GPUå„!‹„úh
- MultiWeatherModel: GPUvL„)!‹¡h
"""

from .lstm_model import LSTMPredictor
from .model_builder import ModelBuilder
from .multi_weather_model import MultiWeatherModel

__all__ = [
    'LSTMPredictor',
    'ModelBuilder',
    'MultiWeatherModel'
]

# H,áo
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "GPU„IŸ‡„K!‹!W"

# GPUÀå
import torch
if torch.cuda.is_available():
    print(f"=€ ÀK0 {torch.cuda.device_count()} *GPU¾ò/(GPU ")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("   *ÀK0CUDA¾(CPU!")