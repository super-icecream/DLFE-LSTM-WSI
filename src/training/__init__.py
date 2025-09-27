"""
DLFE-LSTM-WSI ��!W
GPU����������

!W�:
- GPUOptimizedTrainer: GPU���h
- Validator: !���h�\:6	
- MultiModelValidator: !���h
- AdaptiveOptimizer: ���ph
"""

from .trainer import GPUOptimizedTrainer
from .validator import Validator, MultiModelValidator
from .adaptive_optimizer import AdaptiveOptimizer

__all__ = [
    'GPUOptimizedTrainer',
    'Validator',
    'MultiModelValidator',
    'AdaptiveOptimizer'
]

# H,�o
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "GPU�I���K��!W"

# �KGPU
import torch
if torch.cuda.is_available():
    print(f"��!W: �K0 {torch.cuda.device_count()} *GPU�")
    print(f"   ;GPU: {torch.cuda.get_device_name(0)}")
    print("   �����: �/(")
else:
    print("��!W: (CPU!")