"""
Utilidades y Herramientas del Sistema
Scripts de ayuda para mantenimiento y diagnóstico
"""

import logging
from database import db_manager
from config import PRODUCT_CLASSES
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_database_connection():
    """Prueba la conexión a la base de datos"""
    logger.info("🧪 Probando conexión a base de datos...")
    
    try:
        productos = db_manager.get_all_products()
        logger.info(f"✅ Conexión exitosa. {len(productos)} productos encontrados")
        
        for producto in productos:
            logger.info(f"  • {producto['nombre_producto']}: ${producto['precio']:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error de conexión: {e}")
        return False


def test_camera_connection(source='http://192.168.100.11:8080/video'):
    """Prueba la conexión a la cámara"""
    logger.info(f"🧪 Probando conexión a cámara: {source}")
    
    try:
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            logger.error("❌ No se pudo abrir la cámara")
            return False
        
        ret, frame = cap.read()
        
        if not ret:
            logger.error("❌ No se pudo leer frame")
            cap.release()
            return False
        
        logger.info(f"✅ Cámara conectada. Resolución: {frame.shape[1]}x{frame.shape[0]}")
        
        # Mostrar frame de prueba
        cv2.imshow("Test Camera", frame)
        logger.info("Presiona cualquier tecla para cerrar...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        cap.release()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al probar cámara: {e}")
        return False


def test_voice_assistant():
    """Prueba el asistente de voz"""
    logger.info("🧪 Probando asistente de voz...")
    
    try:
        from voice_assistant import VoiceAssistant
        
        assistant = VoiceAssistant()
        
        # Probar TTS
        logger.info("🔊 Probando síntesis de voz...")
        assistant.speak("Hola, soy el asistente de Bazar Gulpery. Sistema operativo.")
        
        # Probar STT
        logger.info("🎤 Probando reconocimiento de voz...")
        logger.info("Di algo (tienes 5 segundos)...")
        text = assistant.listen_once()
        
        if text:
            logger.info(f"✅ Reconocido: {text}")
            assistant.speak(f"Escuché: {text}")
        else:
            logger.warning("⚠️ No se reconoció ningún texto")
        
        assistant.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al probar asistente de voz: {e}")
        return False


def test_yolo_model(model_path='best.pt'):
    """Prueba que el modelo YOLO funcione"""
    logger.info(f"🧪 Probando modelo YOLO: {model_path}")
    
    try:
        from ultralytics import YOLO
        
        model = YOLO(model_path)
        logger.info(f"✅ Modelo cargado correctamente")
        logger.info(f"Clases configuradas: {len(PRODUCT_CLASSES)}")
        
        for i, clase in enumerate(PRODUCT_CLASSES):
            logger.info(f"  {i}: {clase}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error al cargar modelo: {e}")
        return False


def add_product_to_inventory(nombre: str, precio: float, stock: int = 100):
    """
    Añade un producto al inventario
    
    Args:
        nombre: Nombre del producto
        precio: Precio del producto
        stock: Stock inicial
    """
    logger.info(f"➕ Añadiendo producto: {nombre}")
    
    success = db_manager.add_product(nombre, precio, stock)
    
    if success:
        logger.info(f"✅ Producto añadido: {nombre} - ${precio:.2f}")
    else:
        logger.error(f"❌ Error al añadir producto")


def update_product_price(nombre: str, nuevo_precio: float):
    """
    Actualiza el precio de un producto
    
    Args:
        nombre: Nombre del producto
        nuevo_precio: Nuevo precio
    """
    logger.info(f"💰 Actualizando precio de: {nombre}")
    
    try:
        producto = db_manager.get_product_by_name(nombre)
        
        if not producto:
            logger.error(f"❌ Producto no encontrado: {nombre}")
            return
        
        # Aquí necesitarías implementar update_product_price en database.py
        # Por ahora solo mostramos info
        logger.info(f"Precio actual: ${producto['precio']:.2f}")
        logger.info(f"Nuevo precio: ${nuevo_precio:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")


def show_sales_report(days: int = 7):
    """
    Muestra reporte de ventas
    
    Args:
        days: Número de días a consultar
    """
    logger.info(f"📊 Reporte de ventas (últimos {days} días)")
    logger.info("=" * 60)
    
    try:
        ventas = db_manager.get_sales_history(limit=100)
        
        if not ventas:
            logger.info("No hay ventas registradas")
            return
        
        total_general = 0
        
        for venta in ventas[:20]:  # Mostrar últimas 20
            logger.info(
                f"{venta['fecha'].strftime('%Y-%m-%d %H:%M')} | "
                f"${venta['total_pago']:6.2f} | "
                f"{venta['productos']}"
            )
            total_general += venta['total_pago']
        
        logger.info("=" * 60)
        logger.info(f"Total de ventas mostradas: ${total_general:.2f}")
        
        # Total del día
        total_dia = db_manager.get_daily_sales_total()
        logger.info(f"Total de ventas de HOY: ${total_dia:.2f}")
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")


def run_system_diagnostics():
    """Ejecuta diagnóstico completo del sistema"""
    logger.info("\n" + "=" * 60)
    logger.info("🔍 DIAGNÓSTICO DEL SISTEMA")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Base de Datos", test_database_connection),
        ("Modelo YOLO", lambda: test_yolo_model()),
        ("Cámara", lambda: test_camera_connection()),
        ("Asistente de Voz", test_voice_assistant),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Prueba: {test_name}")
        logger.info('=' * 60)
        
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Error en prueba: {e}")
            results[test_name] = False
    
    # Resumen
    logger.info("\n" + "=" * 60)
    logger.info("📋 RESUMEN DE DIAGNÓSTICO")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name:20s}: {status}")
    
    logger.info("=" * 60 + "\n")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("✅ Todos los tests pasaron. Sistema listo para usar.")
    else:
        logger.warning("⚠️ Algunos tests fallaron. Revisa los errores arriba.")


def main():
    """Menú principal de utilidades"""
    print("\n" + "=" * 60)
    print("🛠️  UTILIDADES DEL SISTEMA POS")
    print("=" * 60)
    print("\nOpciones:")
    print("  1. Ejecutar diagnóstico completo")
    print("  2. Probar conexión a base de datos")
    print("  3. Probar cámara")
    print("  4. Probar asistente de voz")
    print("  5. Probar modelo YOLO")
    print("  6. Mostrar reporte de ventas")
    print("  7. Añadir producto al inventario")
    print("  0. Salir")
    print("=" * 60)
    
    while True:
        try:
            opcion = input("\nSelecciona una opción: ").strip()
            
            if opcion == '1':
                run_system_diagnostics()
            elif opcion == '2':
                test_database_connection()
            elif opcion == '3':
                source = input("URL de cámara (Enter para default): ").strip()
                test_camera_connection(source if source else 'http://192.168.100.11:8080/video')
            elif opcion == '4':
                test_voice_assistant()
            elif opcion == '5':
                test_yolo_model()
            elif opcion == '6':
                show_sales_report()
            elif opcion == '7':
                nombre = input("Nombre del producto: ")
                precio = float(input("Precio: "))
                stock = int(input("Stock inicial (default 100): ") or "100")
                add_product_to_inventory(nombre, precio, stock)
            elif opcion == '0':
                print("\n👋 ¡Hasta luego!\n")
                break
            else:
                print("❌ Opción inválida")
                
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!\n")
            break
        except Exception as e:
            logger.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
