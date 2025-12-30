"""
Módulo de captura de video optimizado
Maneja la captura desde cámara o stream
"""

import cv2
import numpy as np
import threading
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    Clase optimizada para captura de video con buffering en thread separado
    Evita lag en la captura de frames
    """
    
    def __init__(self, source: str, buffer_size: int = 1):
        """
        Inicializa la captura de video
        
        Args:
            source: Fuente de video (índice de cámara o URL de stream)
            buffer_size: Tamaño del buffer (1 = más reciente)
        """
        self.source = source
        self.buffer_size = buffer_size
        
        # Inicializar captura
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video: {source}")
        
        # Configurar propiedades
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, buffer_size)
        
        # Variables de estado
        self.frame = None
        self.grabbed = False
        self.running = True
        
        # Iniciar thread de lectura
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        
        logger.info(f"✅ Captura de video inicializada: {source}")
    
    def _update(self):
        """Thread que lee frames continuamente"""
        while self.running:
            if self.cap.isOpened():
                self.grabbed, self.frame = self.cap.read()
            else:
                self.running = False
                break
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lee el frame más reciente
        
        Returns:
            Tupla (success, frame)
        """
        return self.grabbed, self.frame
    
    def release(self):
        """Libera recursos"""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.cap.release()
        logger.info("Captura de video liberada")
    
    def is_opened(self) -> bool:
        """Verifica si la captura está activa"""
        return self.cap.isOpened() and self.running
    
    def get_fps(self) -> float:
        """Obtiene FPS de la fuente"""
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def get_resolution(self) -> Tuple[int, int]:
        """Obtiene resolución (ancho, alto)"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return width, height
