"""
Módulos principales del sistema
Contiene la lógica de negocio y componentes core
"""

from .detector import ProductDetector
from .database_manager import DatabaseManager
from .pos_system import POSSystem

__all__ = ['ProductDetector', 'DatabaseManager', 'POSSystem']
