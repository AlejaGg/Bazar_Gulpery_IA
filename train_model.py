"""
Script de Entrenamiento del Modelo YOLOv11
Entrena el modelo usando el dataset local

Ejecutar en Google Colab o entorno con GPU:
    python train_model.py
"""

from ultralytics import YOLO
import logging
import os
from pathlib import Path
from config import TRAINING_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataset_path():
    """
    Obtiene la ruta al dataset local
    
    Returns:
        Ruta al archivo data.yaml
    """
    logger.info("📁 Usando dataset local...")
    
    try:
        # Ruta al dataset local
        dataset_dir = Path(__file__).parent / 'dataset'
        data_yaml_path = dataset_dir / 'data.yaml'
        
        if not data_yaml_path.exists():
            raise FileNotFoundError(
                f"❌ No se encontró el archivo data.yaml en: {data_yaml_path}\n"
                f"Asegúrate de que la carpeta 'dataset' contenga el dataset descargado."
            )
        
        logger.info(f"✅ Dataset encontrado en: {dataset_dir}")
        return str(data_yaml_path)
        
    except Exception as e:
        logger.error(f"❌ Error al acceder al dataset: {e}")
        raise


def train_model(data_yaml_path: str):
    """
    Entrena el modelo YOLO
    
    Args:
        data_yaml_path: Ruta al archivo data.yaml del dataset
    """
    logger.info("🚀 Iniciando entrenamiento del modelo...")
    logger.info("=" * 60)
    logger.info(f"Modelo base: {TRAINING_CONFIG['base_model']}")
    logger.info(f"Épocas: {TRAINING_CONFIG['epochs']}")
    logger.info(f"Tamaño de imagen: {TRAINING_CONFIG['img_size']}")
    logger.info(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    logger.info("=" * 60)
    
    try:
        # Cargar modelo base
        model = YOLO(TRAINING_CONFIG['base_model'])
        
        # Entrenar
        results = model.train(
            data=data_yaml_path,
            epochs=TRAINING_CONFIG['epochs'],
            imgsz=TRAINING_CONFIG['img_size'],
            batch=TRAINING_CONFIG['batch_size'],
            name='bazar_gulpery_detector',
            project='yolo_training',
            patience=50,  # Early stopping
            save=True,
            device='0',  # GPU 0
            workers=8,
            verbose=True
        )
        
        logger.info("✅ Entrenamiento completado")
        logger.info(f"Mejor modelo guardado en: runs/detect/bazar_gulpery_detector/weights/best.pt")
        
        # Validar modelo
        logger.info("\n📊 Validando modelo...")
        metrics = model.val()
        
        logger.info("=" * 60)
        logger.info("MÉTRICAS DEL MODELO:")
        logger.info("=" * 60)
        logger.info(f"mAP50: {metrics.box.map50:.4f}")
        logger.info(f"mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"Precision: {metrics.box.mp:.4f}")
        logger.info(f"Recall: {metrics.box.mr:.4f}")
        logger.info("=" * 60)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        raise


def main():
    """Función principal"""
    logger.info("=" * 60)
    logger.info("🧠 ENTRENAMIENTO DE MODELO YOLO PARA BAZAR GULPERY")
    logger.info("=" * 60 + "\n")
    
    try:
        # 1. Obtener ruta del dataset local
        data_yaml_path = get_dataset_path()
        
        # 2. Entrenar modelo
        train_model(data_yaml_path)
        
        print("\n" + "=" * 60)
        print("✅ ENTRENAMIENTO COMPLETADO")
        print("=" * 60)
        print("\nRecuerda copiar el archivo 'best.pt' al directorio del proyecto:")
        print("  cp runs/detect/bazar_gulpery_detector/weights/best.pt .")
        print("\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR EN EL ENTRENAMIENTO")
        print("=" * 60)
        print(f"\n{str(e)}\n")


if __name__ == "__main__":
    main()
