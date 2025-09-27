"""
DLFE-LSTM-WSI !�!W
GPU��f`!���

!W��:
- LSTMPredictor: GPU�LSTM�KQ�
- ModelBuilder: GPU�!���h
- MultiWeatherModel: GPUvL�)!��h
"""

from .lstm_model import LSTMPredictor
from .model_builder import ModelBuilder
from .multi_weather_model import MultiWeatherModel

__all__ = [
    'LSTMPredictor',
    'ModelBuilder',
    'MultiWeatherModel'
]

# H,�o
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "GPU�I���K!�!W"

# GPU��
import torch
if torch.cuda.is_available():
    print(f"=� �K0 {torch.cuda.device_count()} *GPU��/(GPU�")
    for i in range(torch.cuda.device_count()):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("�  *�K0CUDA�(CPU!")