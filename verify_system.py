"""
🔍 VERIFICACIÓN RÁPIDA DEL SISTEMA
Script de verificación automática - Ejecutar antes de iniciar el sistema
"""

import sys
import os
from pathlib import Path
import torch

# Monkey-patch torch.load para usar weights_only=False (compatibilidad PyTorch 2.6+)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

# Colores para terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_header():
    print("\n" + "=" * 70)
    print(f"{BLUE}🔍 VERIFICACIÓN RÁPIDA DEL SISTEMA POS{RESET}")
    print("=" * 70 + "\n")

def check_python_version():
    """Verifica versión de Python"""
    print(f"{BLUE}[1/7]{RESET} Verificando versión de Python...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 8:
        print(f"  {GREEN}✓{RESET} Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"  {RED}✗{RESET} Python {version.major}.{version.minor}.{version.micro} - Requiere 3.8+")
        return False

def check_files():
    """Verifica que existan todos los archivos necesarios"""
    print(f"\n{BLUE}[2/7]{RESET} Verificando archivos del proyecto...")
    
    required_files = [
        'app.py',
        'config.py',
        'database.py',
        'detector.py',
        'voice_assistant.py',
        'ui.py',
        'init_database.py',
        'best.pt',
        'requirements.txt'
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"  {GREEN}✓{RESET} {file}")
        else:
            print(f"  {RED}✗{RESET} {file} - NO ENCONTRADO")
            all_exist = False
    
    return all_exist

def check_dependencies():
    """Verifica dependencias instaladas"""
    print(f"\n{BLUE}[3/7]{RESET} Verificando dependencias...")
    
    dependencies = {
        'cv2': 'opencv-python',
        'ultralytics': 'ultralytics',
        'psycopg2': 'psycopg2-binary',
        'speech_recognition': 'SpeechRecognition',
        'pyttsx3': 'pyttsx3',
        'numpy': 'numpy'
    }
    
    all_installed = True
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  {GREEN}✓{RESET} {package}")
        except ImportError:
            print(f"  {RED}✗{RESET} {package} - NO INSTALADO")
            all_installed = False
    
    if not all_installed:
        print(f"\n  {YELLOW}💡 Instalar con:{RESET} pip install -r requirements.txt")
    
    return all_installed

def check_model():
    """Verifica modelo YOLO"""
    print(f"\n{BLUE}[4/7]{RESET} Verificando modelo YOLO...")
    
    if not Path('best.pt').exists():
        print(f"  {RED}✗{RESET} best.pt - NO ENCONTRADO")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO('best.pt')
        print(f"  {GREEN}✓{RESET} Modelo cargado correctamente")
        return True
    except Exception as e:
        print(f"  {RED}✗{RESET} Error al cargar modelo: {e}")
        return False

def check_database():
    """Verifica conexión a base de datos"""
    print(f"\n{BLUE}[5/7]{RESET} Verificando base de datos...")
    
    try:
        import psycopg2  # type: ignore
        from config import DATABASE_CONFIG
        
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database='postgres'
        )
        conn.close()
        print(f"  {GREEN}✓{RESET} Conexión a PostgreSQL exitosa")
        
        # Verificar si existe la BD del sistema
        conn = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['user'],
            password=DATABASE_CONFIG['password'],
            database='postgres'
        )
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE_CONFIG['database'],))
        exists = cur.fetchone()
        cur.close()
        conn.close()
        
        if exists:
            print(f"  {GREEN}✓{RESET} Base de datos '{DATABASE_CONFIG['database']}' existe")
            return True
        else:
            print(f"  {YELLOW}⚠{RESET} Base de datos '{DATABASE_CONFIG['database']}' no existe")
            print(f"  {YELLOW}💡 Ejecutar:{RESET} python init_database.py")
            return False
            
    except Exception as e:
        print(f"  {RED}✗{RESET} Error de conexión: {e}")
        print(f"  {YELLOW}💡 Verificar:{RESET}")
        print(f"     - PostgreSQL está corriendo")
        print(f"     - Credenciales en config.py son correctas")
        return False

def check_camera():
    """Verifica cámara"""
    print(f"\n{BLUE}[6/7]{RESET} Verificando cámara...")
    
    try:
        import cv2
        from config import CAMERA_CONFIG
        
        cap = cv2.VideoCapture(CAMERA_CONFIG['source'])
        
        if not cap.isOpened():
            print(f"  {YELLOW}⚠{RESET} No se puede abrir cámara: {CAMERA_CONFIG['source']}")
            print(f"  {YELLOW}💡 Para usar webcam:{RESET} Cambiar 'source' a 0 en config.py")
            cap.release()
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            print(f"  {GREEN}✓{RESET} Cámara conectada - {frame.shape[1]}x{frame.shape[0]}")
            return True
        else:
            print(f"  {YELLOW}⚠{RESET} No se puede leer frame de la cámara")
            return False
            
    except Exception as e:
        print(f"  {RED}✗{RESET} Error: {e}")
        return False

def check_microphone():
    """Verifica micrófono"""
    print(f"\n{BLUE}[7/7]{RESET} Verificando micrófono...")
    
    try:
        import speech_recognition as sr
        
        recognizer = sr.Recognizer()
        mic = sr.Microphone()
        
        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
        
        print(f"  {GREEN}✓{RESET} Micrófono detectado")
        return True
        
    except Exception as e:
        print(f"  {YELLOW}⚠{RESET} Micrófono no disponible: {e}")
        print(f"  {YELLOW}💡 El sistema funcionará sin asistente de voz{RESET}")
        return False

def print_summary(results):
    """Imprime resumen de verificación"""
    print("\n" + "=" * 70)
    print(f"{BLUE}📊 RESUMEN DE VERIFICACIÓN{RESET}")
    print("=" * 70 + "\n")
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = f"{GREEN}✓ PASS{RESET}" if result else f"{RED}✗ FAIL{RESET}"
        print(f"  {check:20s}: {status}")
    
    print("\n" + "-" * 70)
    print(f"  Total: {passed}/{total} verificaciones pasadas")
    print("-" * 70 + "\n")
    
    if passed == total:
        print(f"{GREEN}✅ SISTEMA LISTO PARA USAR{RESET}")
        print(f"\nEjecutar: {BLUE}python app.py{RESET}\n")
        return True
    elif passed >= 5:
        print(f"{YELLOW}⚠️  SISTEMA PARCIALMENTE LISTO{RESET}")
        print(f"\nAlgunas funcionalidades pueden no estar disponibles.")
        print(f"Revisa los errores arriba y corrígelos si es necesario.\n")
        return False
    else:
        print(f"{RED}❌ SISTEMA NO LISTO{RESET}")
        print(f"\nRevisa y corrige los errores antes de ejecutar el sistema.\n")
        return False

def main():
    """Función principal"""
    print_header()
    
    # Ejecutar todas las verificaciones
    results = {
        'Python Version': check_python_version(),
        'Project Files': check_files(),
        'Dependencies': check_dependencies(),
        'YOLO Model': check_model(),
        'Database': check_database(),
        'Camera': check_camera(),
        'Microphone': check_microphone()
    }
    
    # Mostrar resumen
    ready = print_summary(results)
    
    return 0 if ready else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
