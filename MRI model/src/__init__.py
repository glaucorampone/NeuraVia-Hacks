"""
NeuraVia: Deep Learning MRI Alzheimer's Prediction

A production-ready system for Alzheimer's disease prediction from MRI scans.
"""

__version__ = "1.0.0"
__description__ = "Deep Learning MRI Alzheimer's Prediction for NeuraVia Hack Challenge"

from .neurovia_network import create_neurovia_model
from .neurovia_inference import NeuraViaPredictor
from .neurovia_config import get_model_path, interpret_score

__all__ = [
    'create_neurovia_model',
    'NeuraViaPredictor', 
    'get_model_path',
    'interpret_score'
]
