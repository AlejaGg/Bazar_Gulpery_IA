"""
Utilidades y funciones auxiliares del sistema
"""

from .logger import setup_logger
from .video_capture import VideoCapture

__all__ = ['setup_logger', 'VideoCapture']
