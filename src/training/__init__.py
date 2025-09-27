"""
DLFE-LSTM-WSI ­Ã!W
GPU„­ÃŒÁŒê”Äö

!WÄ:
- GPUOptimizedTrainer: GPU ­Ãh
- Validator: !‹ŒÁhé\:6	
- MultiModelValidator: !‹ŒÁh
- AdaptiveOptimizer: ê”Âph
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

# H,áo
__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = "GPU„IŸ‡„K­Ã!W"

# ÀKGPU
import torch
if torch.cuda.is_available():
    print(f"­Ã!W: ÀK0 {torch.cuda.device_count()} *GPU¾")
    print(f"   ;GPU: {torch.cuda.get_device_name(0)}")
    print("   ÷¾¦­Ã: ò/(")
else:
    print("­Ã!W: (CPU!")