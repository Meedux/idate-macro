"""Machine Learning module for iDate detection."""
from .yolo_detector import YOLODetector
from .data_generator import generate_training_data

__all__ = ['YOLODetector', 'generate_training_data']
