"""
Configuración centralizada del Sistema de Punto de Venta con IA
Autor: Sistema de Arquitectura de IA
Fecha: 2025-12-30
"""

import os
from pathlib import Path

# ==================== RUTAS DEL PROYECTO ====================
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_DIR = PROJECT_ROOT / "dataset"
LOGS_DIR = PROJECT_ROOT / "logs"
DOCS_DIR = PROJECT_ROOT / "documentacion"

# Crear directorios si no existen
LOGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ==================== CONFIGURACIÓN DE BASE DE DATOS ====================
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'gulpery2025'),
    'database': os.getenv('DB_NAME', 'bazar_gulpery'),
    'pool_size': 5,
    'max_overflow': 10,
}

# ==================== CONFIGURACIÓN DEL MODELO YOLO ====================
MODEL_CONFIG = {
    'model_path': str(PROJECT_ROOT / 'best.pt'),
    'confidence_threshold': 0.5,
    'iou_threshold': 0.45,
    'img_size': 640,
    'device': 'cuda' if os.getenv('USE_GPU', 'true').lower() == 'true' else 'cpu',
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

# Mapeo de IDs a nombres de productos
PRODUCT_ID_TO_NAME = {i: name for i, name in enumerate(PRODUCT_CLASSES)}
PRODUCT_NAME_TO_ID = {name: i for i, name in enumerate(PRODUCT_CLASSES)}

# ==================== CONFIGURACIÓN DE CÁMARA ====================
CAMERA_CONFIG = {
    'source': os.getenv('CAMERA_SOURCE', 'http://192.168.100.11:8080/video'),
    'fps': 30,
    'width': 1280,
    'height': 720,
    'buffer_size': 1,
}

# ==================== CONFIGURACIÓN DE VOZ ====================
VOICE_CONFIG = {
    'keyword': 'LISTO',
    'language': 'es-ES',
    'tts_rate': 150,
    'tts_volume': 0.9,
    'recognition_timeout': 5,
}

# ==================== CONFIGURACIÓN DE ENTRENAMIENTO ====================
TRAINING_CONFIG = {
    'base_model': 'yolo11n.pt',
    'epochs': 200,
    'img_size': 640,
    'batch_size': 16,
    'learning_rate': 0.01,
    'optimizer': 'AdamW',
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3,
    'patience': 50,
    'augmentation': {
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
    }
}

# ==================== CONFIGURACIÓN DE INTERFAZ UI ====================
UI_CONFIG = {
    'window_name': 'Sistema POS - Bazar Gulpery',
    'font_scale': 0.7,
    'font_thickness': 2,
    'line_thickness': 2,
    'colors': {
        'primary': (0, 255, 0),      # Verde
        'secondary': (255, 165, 0),  # Naranja
        'accent': (255, 0, 255),     # Magenta
        'text': (255, 255, 255),     # Blanco
        'background': (50, 50, 50),  # Gris oscuro
    }
}

# ==================== CONFIGURACIÓN DE LOGGING ====================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': str(LOGS_DIR / 'pos_system.log'),
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers': ['console', 'file']
        }
    }
}

# ==================== CONSTANTES DEL SISTEMA ====================
SYSTEM_CONSTANTS = {
    'MAX_DETECTION_AGE': 3.0,        # segundos
    'DETECTION_CACHE_SIZE': 5,        # frames
    'MIN_CONFIDENCE_DISPLAY': 0.3,    # umbral mínimo para mostrar
    'STABILITY_THRESHOLD': 0.7,       # umbral de estabilidad
    'FPS_UPDATE_INTERVAL': 1.0,       # segundos
}
