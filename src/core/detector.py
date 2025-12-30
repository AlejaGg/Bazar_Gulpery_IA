"""
Módulo de Detección de Productos con YOLOv11
Maneja la inferencia del modelo y procesamiento de detecciones
"""

import cv2
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional
from collections import Counter
from pathlib import Path

# Monkey-patch torch.load para compatibilidad PyTorch 2.6+
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

from ..config import MODEL_CONFIG, PRODUCT_CLASSES, SYSTEM_CONSTANTS

logger = logging.getLogger(__name__)


class Detection:
    """Representa una detección individual"""
    
    def __init__(self, class_id: int, class_name: str, confidence: float, 
                 bbox: Tuple[int, int, int, int]):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
    
    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f})"
    
    def to_dict(self) -> Dict:
        """Convierte la detección a diccionario"""
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': self.confidence,
            'bbox': self.bbox
        }


class ProductDetector:
    """
    Detector de productos basado en YOLOv11
    Realiza inferencia en tiempo real y procesa detecciones
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Inicializa el detector con el modelo YOLO
        
        Args:
            model_path: Ruta al modelo entrenado (best.pt)
        """
        self.model_path = model_path or MODEL_CONFIG['model_path']
        self.confidence_threshold = MODEL_CONFIG['confidence_threshold']
        self.iou_threshold = MODEL_CONFIG['iou_threshold']
        self.device = MODEL_CONFIG['device']
        
        # Verificar que existe el modelo
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"No se encontró el modelo en: {self.model_path}\n"
                "Asegúrate de entrenar el modelo primero o copiar best.pt al directorio raíz."
            )
        
        # Cargar modelo
        try:
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            logger.info(f"✅ Modelo YOLO cargado: {self.model_path}")
            logger.info(f"   Dispositivo: {self.device}")
            logger.info(f"   Clases: {len(PRODUCT_CLASSES)}")
        except Exception as e:
            logger.error(f"❌ Error al cargar modelo YOLO: {e}")
            raise
        
        # Caché de detecciones para estabilidad
        self.detection_cache: List[List[Detection]] = []
        self.cache_size = SYSTEM_CONSTANTS['DETECTION_CACHE_SIZE']
        
        # Estadísticas
        self.total_inferences = 0
        self.total_detections = 0
    
    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
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
                verbose=False,
                device=self.device
            )
            
            # Actualizar estadísticas
            self.total_inferences += 1
            
            # Obtener frame anotado
            annotated_frame = results[0].plot()
            
            # Extraer detecciones estructuradas
            detections = self._parse_detections(results[0])
            self.total_detections += len(detections)
            
            # Aplicar filtro de estabilidad
            stable_detections = self._stabilize_detections(detections)
            
            return annotated_frame, stable_detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return frame, []
    
    def _parse_detections(self, result) -> List[Detection]:
        """
        Parsea los resultados de YOLO a objetos Detection
        
        Args:
            result: Objeto de resultado de YOLO
            
        Returns:
            Lista de objetos Detection
        """
        detections = []
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.cpu().numpy()
        
        for box in boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Validar que el ID de clase es válido
            if 0 <= class_id < len(PRODUCT_CLASSES):
                class_name = PRODUCT_CLASSES[class_id]
                
                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2)
                )
                detections.append(detection)
        
        return detections
    
    def _stabilize_detections(self, current_detections: List[Detection]) -> List[Detection]:
        """
        Aplica filtro de estabilidad temporal a las detecciones
        Solo retorna detecciones que aparecen consistentemente
        
        Args:
            current_detections: Detecciones del frame actual
            
        Returns:
            Lista de detecciones estables
        """
        # Agregar al caché
        self.detection_cache.append(current_detections)
        
        # Mantener tamaño del caché
        if len(self.detection_cache) > self.cache_size:
            self.detection_cache.pop(0)
        
        # Si el caché no está lleno, retornar detecciones actuales
        if len(self.detection_cache) < self.cache_size:
            return current_detections
        
        # Contar frecuencia de cada clase en el caché
        class_counter = Counter()
        for frame_detections in self.detection_cache:
            for detection in frame_detections:
                class_counter[detection.class_name] += 1
        
        # Filtrar detecciones que aparecen en al menos X% de los frames
        threshold = self.cache_size * SYSTEM_CONSTANTS['STABILITY_THRESHOLD']
        stable_classes = {
            class_name for class_name, count in class_counter.items()
            if count >= threshold
        }
        
        # Retornar solo detecciones de clases estables
        stable_detections = [
            det for det in current_detections
            if det.class_name in stable_classes
        ]
        
        return stable_detections
    
    def get_statistics(self) -> Dict:
        """Retorna estadísticas del detector"""
        return {
            'total_inferences': self.total_inferences,
            'total_detections': self.total_detections,
            'avg_detections_per_frame': (
                self.total_detections / self.total_inferences 
                if self.total_inferences > 0 else 0
            ),
            'cache_size': len(self.detection_cache),
        }
    
    def reset_statistics(self):
        """Reinicia las estadísticas"""
        self.total_inferences = 0
        self.total_detections = 0
        self.detection_cache.clear()
