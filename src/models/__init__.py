"""Model package exports for DLFE-LSTM-WSI."""

from .lstm_model import LSTMPredictor
from .model_builder import ModelBuilder
from .multi_weather_model import MultiWeatherModel

__all__ = [
    "LSTMPredictor",
    "ModelBuilder",
    "MultiWeatherModel",
]

__version__ = "1.0.0"
__author__ = "DLFE-LSTM-WSI Team"
__description__ = (
    "GPU-accelerated LSTM predictors and builders for the DLFE-LSTM-WSI project"
)

