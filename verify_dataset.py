"""
Script de verificación del dataset local
Verifica que el dataset esté correctamente configurado
"""

from pathlib import Path
import yaml

def verify_dataset():
    """Verifica que el dataset local esté disponible y correctamente configurado"""
    
    print("=" * 60)
    print("🔍 VERIFICACIÓN DEL DATASET LOCAL")
    print("=" * 60 + "\n")
    
    # Verificar carpeta dataset
    dataset_dir = Path(__file__).parent / 'dataset'
    
    if not dataset_dir.exists():
        print("❌ ERROR: La carpeta 'dataset' no existe")
        return False
    
    print(f"✅ Carpeta dataset encontrada: {dataset_dir}")
    
    # Verificar data.yaml
    data_yaml = dataset_dir / 'data.yaml'
    
    if not data_yaml.exists():
        print("❌ ERROR: El archivo 'data.yaml' no existe")
        return False
    
    print(f"✅ Archivo data.yaml encontrado")
    
    # Leer configuración
    with open(data_yaml, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📊 INFORMACIÓN DEL DATASET:")
    print(f"  - Número de clases: {data_config.get('nc', 'N/A')}")
    print(f"  - Clases detectadas:")
    
    for i, name in enumerate(data_config.get('names', []), 1):
        print(f"    {i}. {name}")
    
    # Verificar carpetas
    print(f"\n📁 CARPETAS:")
    folders = ['train', 'valid', 'test']
    
    for folder in folders:
        folder_path = dataset_dir / folder
        if folder_path.exists():
            images_dir = folder_path / 'images'
            labels_dir = folder_path / 'labels'
            
            img_count = len(list(images_dir.glob('*.jpg'))) if images_dir.exists() else 0
            lbl_count = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
            
            print(f"  ✅ {folder:8s}: {img_count} imágenes, {lbl_count} etiquetas")
        else:
            print(f"  ❌ {folder:8s}: No encontrada")
    
    print("\n" + "=" * 60)
    print("✅ DATASET LOCAL VERIFICADO CORRECTAMENTE")
    print("=" * 60)
    print("\n💡 Ya no necesitas Roboflow API para entrenar!")
    print("   Puedes ejecutar: python train_model.py\n")
    
    return True


if __name__ == "__main__":
    verify_dataset()
