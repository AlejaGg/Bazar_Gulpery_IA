"""
Configuración centralizada del Sistema de Punto de Venta con IA
Autor: Sistema de Arquitectura de IA
Fecha: 2025-12-28
"""

# ==================== CONFIGURACIÓN DE BASE DE DATOS ====================
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': 'gulpery2025',
    'database': 'bazar_gulpery'
}

# ==================== CONFIGURACIÓN DEL MODELO YOLO ====================
MODEL_CONFIG = {
    'model_path': 'best.pt',
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'img_size': 640
}

# ==================== CLASES DE PRODUCTOS ====================
PRODUCT_CLASSES = [
    'Borrador de ballena',
    'Borrador de sirena',
    'Esfero Negro',
    'Flash Kingston 4GB',
    'Flash Verbatim 16Gb',
    'Pasador Cabello Minimouse',
    'Resaltador',
    'cartera',
    'perfume'
]

# ==================== CONFIGURACIÓN DE CÁMARA ====================
CAMERA_CONFIG = {
    'source': 'http://192.168.100.11:8080/video',
    'fps': 30,
    'width': 1280,
    'height': 720
}

# ==================== CONFIGURACIÓN DE VOZ ====================
VOICE_CONFIG = {
    'keyword': 'LISTO',
    'language': 'es-ES',
    'tts_rate': 150,
    'tts_volume': 0.9
}

# ==================== CONFIGURACIÓN DE INTERFAZ ====================
import cv2  # Importar aquí para evitar errores circulares

UI_CONFIG = {
    'window_name': 'Bazar Gulpery - Sistema POS con IA',
    'font': cv2.FONT_HERSHEY_SIMPLEX,
    'font_scale': 0.6,
    'font_thickness': 2,
    'bbox_color': (0, 255, 0),  # Verde
    'text_color': (255, 255, 255),  # Blanco
    'cart_bg_color': (50, 50, 50),  # Gris oscuro
    'cart_position': (10, 50)
}

# ==================== CONFIGURACIÓN DE ROBOFLOW ====================
ROBOFLOW_CONFIG = {
    'api_key': 'vxVrVEeW04cCPGwwCTod',
    'workspace': 'bazarmg',
    'project': 'my-first-project-fiobt',
    'version': 2
}

# ==================== CONFIGURACIÓN DE ENTRENAMIENTO ====================
TRAINING_CONFIG = {
    'base_model': 'yolov8n.pt',
    'epochs': 250,
    'batch_size': 16,
    'img_size': 640,
    'data_yaml': '/content/My-First-Project-2/data.yaml'
}
