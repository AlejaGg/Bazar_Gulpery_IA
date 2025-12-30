"""
Sistema de logging configurado para el proyecto
"""

import logging
import logging.config
from pathlib import Path
from typing import Optional

from ..config import LOGGING_CONFIG, LOGS_DIR


def setup_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configura y retorna un logger para el sistema
    
    Args:
        name: Nombre del logger (None para root logger)
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Logger configurado
    """
    # Asegurar que existe el directorio de logs
    LOGS_DIR.mkdir(exist_ok=True)
    
    # Configurar logging
    logging.config.dictConfig(LOGGING_CONFIG)
    
    # Obtener logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    return logger


def log_system_info(logger: logging.Logger):
    """
    Registra información del sistema en el log
    
    Args:
        logger: Logger a utilizar
    """
    import platform
    import sys
    import torch
    import cv2
    
    logger.info("=" * 80)
    logger.info("INFORMACIÓN DEL SISTEMA")
    logger.info("=" * 80)
    logger.info(f"Sistema Operativo: {platform.system()} {platform.release()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"OpenCV: {cv2.__version__}")
    logger.info(f"CUDA Disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"Dispositivo CUDA: {torch.cuda.get_device_name(0)}")
    logger.info("=" * 80)
