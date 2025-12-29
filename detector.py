"""
Módulo de Detección de Productos con YOLOv11
Maneja la inferencia del modelo y procesamiento de detecciones
"""

import cv2
import numpy as np
import torch
import warnings

# Monkey-patch torch.load para usar weights_only=False (compatibilidad PyTorch 2.6+)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO
from typing import List, Dict, Tuple
import logging
from collections import Counter
from config import MODEL_CONFIG, PRODUCT_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductDetector:
    """
    Detector de productos basado en YOLOv11
    Realiza inferencia en tiempo real y procesa detecciones
    """
    
    def __init__(self, model_path: str = None):
        """
        Inicializa el detector con el modelo YOLO
        
        Args:
            model_path: Ruta al modelo entrenado (best.pt)
        """
        self.model_path = model_path or MODEL_CONFIG['model_path']
        self.confidence_threshold = MODEL_CONFIG['confidence_threshold']
        self.iou_threshold = MODEL_CONFIG['iou_threshold']
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"✅ Modelo YOLO cargado: {self.model_path}")
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo YOLO: {e}")
            raise
        
        # Caché de detecciones para estabilidad
        self.detection_cache = []
        self.cache_size = 5
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Realiza detección en un frame
        
        Args:
            frame: Frame de entrada (imagen BGR)
            
        Returns:
            Tupla con (frame anotado, lista de detecciones)
        """
        try:
            # Realizar inferencia
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Obtener frame anotado
            annotated_frame = results[0].plot()
            
            # Extraer detecciones estructuradas
            detections = self._parse_detections(results[0])
            
            # Aplicar filtro de estabilidad
            stable_detections = self._stabilize_detections(detections)
            
            return annotated_frame, stable_detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return frame, []
    
    def _parse_detections(self, result) -> List[Dict]:
        """
        Parsea los resultados de YOLO a una estructura más manejable
        
        Args:
            result: Objeto de resultado de YOLO
            
        Returns:
            Lista de diccionarios con información de cada detección
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()  # Coordenadas [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # Confianzas
        class_ids = result.boxes.cls.cpu().numpy().astype(int)  # IDs de clase
        
        for i in range(len(boxes)):
            class_id = class_ids[i]
            
            # Verificar que el ID de clase sea válido
            if 0 <= class_id < len(PRODUCT_CLASSES):
                detection = {
                    'class_id': class_id,
                    'class_name': PRODUCT_CLASSES[class_id],
                    'confidence': float(confidences[i]),
                    'bbox': boxes[i].tolist(),  # [x1, y1, x2, y2]
                    'center': [
                        (boxes[i][0] + boxes[i][2]) / 2,
                        (boxes[i][1] + boxes[i][3]) / 2
                    ]
                }
                detections.append(detection)
        
        return detections
    
    def _stabilize_detections(self, detections: List[Dict]) -> List[Dict]:
        """
        Estabiliza las detecciones usando un caché temporal
        Filtra detecciones intermitentes
        
        Args:
            detections: Lista de detecciones actuales
            
        Returns:
            Lista de detecciones estabilizadas
        """
        # Agregar detecciones actuales al caché
        self.detection_cache.append(detections)
        
        # Mantener solo las últimas N detecciones
        if len(self.detection_cache) > self.cache_size:
            self.detection_cache.pop(0)
        
        # Contar frecuencia de cada producto
        product_counter = Counter()
        for frame_detections in self.detection_cache:
            for detection in frame_detections:
                product_counter[detection['class_name']] += 1
        
        # Filtrar solo productos que aparecen en mayoría de frames
        threshold = len(self.detection_cache) * 0.6  # 60% de los frames
        stable_products = {
            product for product, count in product_counter.items() 
            if count >= threshold
        }
        
        # Retornar solo detecciones estables
        return [
            detection for detection in detections 
            if detection['class_name'] in stable_products
        ]
    
    def get_unique_products(self, detections: List[Dict]) -> List[str]:
        """
        Obtiene lista única de productos detectados
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Lista de nombres únicos de productos
        """
        return list(set(detection['class_name'] for detection in detections))
    
    def count_products(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Cuenta la cantidad de cada producto detectado
        
        Args:
            detections: Lista de detecciones
            
        Returns:
            Diccionario con conteo de productos
        """
        counter = Counter(detection['class_name'] for detection in detections)
        return dict(counter)
    
    def reset_cache(self):
        """Limpia el caché de detecciones"""
        self.detection_cache = []
        logger.info("🔄 Caché de detecciones reiniciado")


class VideoCapture:
    """
    Manejador de captura de video con reconexión automática
    """
    
    def __init__(self, source: str):
        """
        Inicializa la captura de video
        
        Args:
            source: URL o índice de la cámara
        """
        self.source = source
        self.cap = None
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        
        self._initialize_capture()
    
    def _initialize_capture(self):
        """Inicializa la captura de video"""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise Exception("No se pudo abrir la fuente de video")
            logger.info(f"✅ Captura de video inicializada: {self.source}")
        except Exception as e:
            logger.error(f"❌ Error al inicializar captura: {e}")
            raise
    
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Lee un frame del video
        
        Returns:
            Tupla (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            if self.reconnect_attempts < self.max_reconnect_attempts:
                logger.warning("⚠️ Intentando reconectar...")
                self.reconnect_attempts += 1
                self._initialize_capture()
            else:
                return False, None
        
        ret, frame = self.cap.read()
        
        if ret:
            self.reconnect_attempts = 0  # Reset contador si lectura exitosa
        
        return ret, frame
    
    def release(self):
        """Libera los recursos de captura"""
        if self.cap:
            self.cap.release()
            logger.info("🔒 Captura de video liberada")
    
    def is_opened(self) -> bool:
        """Verifica si la captura está activa"""
        return self.cap is not None and self.cap.isOpened()
