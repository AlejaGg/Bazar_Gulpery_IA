"""
Sistema de Punto de Venta Automatizado con IA
Bazar Gulpery - 2025

Autor: Arquitecto de Soluciones de IA
Versión: 2.0

Características:
- Detección de productos con YOLOv11
- Gestión de precios con PostgreSQL
- Asistente de voz interactivo
- Interfaz visual en tiempo real
"""

import torch
# Monkey-patch torch.load para usar weights_only=False (compatibilidad PyTorch 2.6+)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import cv2
import time
import logging
from typing import Dict
import threading
from typing import Optional, Union

# Módulos personalizados
from config import CAMERA_CONFIG, PRODUCT_CLASSES
from detector import ProductDetector, VideoCapture
from database import db_manager
from voice_assistant import VoiceAssistant
from ui import UIRenderer, ShoppingCart
from menu import MenuPrincipal, PanelAdmin

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _normalize_camera_source(user_value: str, default_source: Union[str, int]) -> Union[str, int]:
    """Normaliza la entrada del usuario a un source válido para OpenCV."""
    value = (user_value or "").strip()
    if not value:
        return default_source

    # Permitir webcam local (ej: 0, 1, 2)
    if value.isdigit():
        return int(value)

    # Corregir typo común: "IP::PUERTO" -> "IP:PUERTO"
    value = value.replace("::", ":")

    # Si ya es URL completa
    if value.startswith("http://") or value.startswith("https://"):
        return value

    # Si viene como IP o IP:PUERTO
    if "/" not in value:
        # Si el usuario solo ingresa IP, usar puerto 8080 por defecto
        if ":" not in value:
            value = f"{value}:8080"
        return f"http://{value}/video"

    return f"http://{value}"


class POSSystem:
    """
    Sistema de Punto de Venta Principal
    Coordina todos los módulos del sistema
    """
    
    def __init__(self, camera_source: Optional[Union[str, int]] = None):
        """Inicializa el sistema POS"""
        logger.info("=" * 60)
        logger.info("🚀 Inicializando Sistema de Punto de Venta con IA")
        logger.info("=" * 60)
        
        # Componentes principales
        self.detector = None
        self.video_capture = None
        self.voice_assistant = None
        self.ui_renderer = None
        self.shopping_cart = None
        
        # Mapa de precios (caché)
        self.prices_map: Dict[str, float] = {}
        
        # Control de estado
        self.is_running = False
        self.fps = 0.0
        self.current_frame = None  # Frame actual para captura de imágenes

        # Fuente de cámara (si el usuario la especifica al iniciar)
        self.camera_source = camera_source
        
        # Inicializar componentes
        self._initialize_components()
        
        logger.info("✅ Sistema POS inicializado correctamente")
    
    def _initialize_components(self):
        """Inicializa todos los componentes del sistema"""
        try:
            # 1. Detector YOLO
            logger.info("📦 Cargando detector YOLO...")
            self.detector = ProductDetector()
            
            # 2. Captura de video
            logger.info("📹 Inicializando captura de video...")
            source = self.camera_source if self.camera_source is not None else CAMERA_CONFIG['source']
            self.video_capture = VideoCapture(source)
            
            # 3. Renderizador UI
            logger.info("🖥️ Configurando interfaz visual...")
            self.ui_renderer = UIRenderer()
            self.ui_renderer.create_window()
            
            # 4. Carrito de compras
            logger.info("🛒 Inicializando carrito de compras...")
            self.shopping_cart = ShoppingCart()
            
            # 5. Cargar precios desde base de datos
            logger.info("💰 Cargando precios desde base de datos...")
            self._load_prices()
            
            # 6. Asistente de voz
            logger.info("🎤 Inicializando asistente de voz...")
            self.voice_assistant = VoiceAssistant()
            self.voice_assistant.on_keyword_detected = self._handle_checkout
            
        except Exception as e:
            logger.error(f"❌ Error al inicializar componentes: {e}")
            raise
    
    def _load_prices(self):
        """Carga los precios de productos desde la base de datos"""
        try:
            productos = db_manager.get_all_products()
            
            for producto in productos:
                nombre = producto['nombre_producto']
                precio = float(producto['precio'])
                self.prices_map[nombre] = precio
            
            logger.info(f"✅ {len(self.prices_map)} precios cargados")
            
            # Si no hay precios, mostrar advertencia
            if not self.prices_map:
                logger.warning("⚠️ No se encontraron precios en la base de datos")
                logger.warning("⚠️ Asegúrate de ejecutar init_database.py primero")
            
        except Exception as e:
            logger.error(f"❌ Error al cargar precios: {e}")
            logger.warning("⚠️ El sistema funcionará sin precios")
    
    def _handle_checkout(self):
        """
        Maneja el proceso de checkout cuando se detecta la palabra clave LISTO
        Captura imagen, procesa venta y continua con siguiente compra
        """
        logger.info("🎤 Palabra clave 'LISTO' detectada - Procesando checkout...")
        
        try:
            # Verificar si hay productos en el carrito
            if self.shopping_cart.is_empty():
                logger.warning("⚠️ Carrito vacío - No hay nada que procesar")
                self.voice_assistant.speak("No hay productos en el carrito.")
                return
            
            # Obtener productos del carrito
            productos = self.shopping_cart.get_product_list()
            total = self.shopping_cart.get_total()
            
            # CAPTURAR IMAGEN del frame actual
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            imagen_path = f"ventas/venta_{timestamp}.jpg"
            
            # Crear directorio si no existe
            import os
            os.makedirs("ventas", exist_ok=True)
            
            # Guardar imagen del frame actual con el carrito visible
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                cv2.imwrite(imagen_path, self.current_frame)
                logger.info(f"📸 Imagen capturada: {imagen_path}")
            
            # Registrar venta en base de datos
            success = db_manager.register_sale(productos, total)
            
            if success:
                logger.info(f"✅ Venta registrada - Total: ${total:.2f}")
                
                # Anunciar venta completada
                self.voice_assistant.announce_sale(productos, total)
                
                # Esperar a que termine de hablar
                time.sleep(2)
                
                # Limpiar carrito para la siguiente compra
                self.shopping_cart.clear()
                self.detector.reset_cache()
                
                logger.info("🛒 Carrito limpiado - Listo para siguiente compra")
                self.voice_assistant.speak("Lista para la siguiente compra.")
                
                # NO CERRAR LA APP - Continuar funcionando
            else:
                logger.error("❌ Error al registrar venta")
                self.voice_assistant.speak("Error al procesar la venta.")
        
        except Exception as e:
            logger.error(f"❌ Error en checkout: {e}")
            self.voice_assistant.speak("Error al procesar la compra.")
    
    def _process_frame(self, frame):
        """
        Procesa un frame: detección, actualización de carrito y renderizado
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame procesado
        """
        # Guardar frame actual para captura de imagen
        self.current_frame = frame.copy()
        
        # 1. Realizar detección
        annotated_frame, detections = self.detector.detect(frame)
        
        # 2. Obtener lista de productos únicos
        productos_detectados = self.detector.get_unique_products(detections)
        
        # 3. Actualizar carrito
        self.shopping_cart.update(productos_detectados, self.prices_map)
        
        # 4. Dibujar información en el frame
        # (Aquí usamos el frame original, no el anotado, para tener control total)
        processed_frame = self.ui_renderer.draw_detection_info(
            frame, detections, self.prices_map
        )
        
        # 5. Dibujar carrito de compras
        processed_frame = self.ui_renderer.draw_shopping_cart(
            processed_frame,
            self.shopping_cart.items,
            self.shopping_cart.total
        )
        
        # 6. Dibujar barra de estado
        processed_frame = self.ui_renderer.draw_status_bar(
            processed_frame,
            listening=self.voice_assistant.is_listening,
            fps=self.fps,
            detection_count=len(detections)
        )
        
        return processed_frame
    
    def run(self):
        """
        Loop principal del sistema
        """
        logger.info("🎬 Iniciando loop principal...")
        logger.info("=" * 60)
        logger.info("Presiona ESC para salir")
        logger.info("Di 'LISTO' para finalizar compra actual y continuar")
        logger.info("Presiona 'C' para limpiar carrito")
        logger.info("Presiona 'V' para activar/desactivar voz")
        logger.info("=" * 60)
        
        self.is_running = True
        
        # Iniciar asistente de voz
        self.voice_assistant.start_listening()
        
        # Variables para calcular FPS
        frame_count = 0
        start_time = time.time()
        
        try:
            while self.is_running:
                # Leer frame
                ret, frame = self.video_capture.read()
                
                if not ret:
                    logger.warning("⚠️ No se pudo leer frame")
                    break
                
                # Procesar frame
                processed_frame = self._process_frame(frame)
                
                # Calcular FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 1.0:
                    self.fps = frame_count / elapsed_time
                    frame_count = 0
                    start_time = time.time()
                
                # Mostrar frame
                self.ui_renderer.show_frame(processed_frame)
                
                # Manejar input del teclado
                key = self.ui_renderer.wait_key(1)
                
                if key == 27:  # ESC
                    logger.info("👋 Saliendo del sistema...")
                    break
                elif key == ord('c') or key == ord('C'):
                    self.shopping_cart.clear()
                    self.detector.reset_cache()
                    logger.info("🛒 Carrito limpiado manualmente")
                elif key == ord('v') or key == ord('V'):
                    if self.voice_assistant.is_listening:
                        self.voice_assistant.stop_listening()
                        logger.info("🔇 Asistente de voz desactivado")
                    else:
                        self.voice_assistant.start_listening()
                        logger.info("🔊 Asistente de voz activado")
        
        except KeyboardInterrupt:
            logger.info("⚠️ Interrupción por teclado")
        except Exception as e:
            logger.error(f"❌ Error en loop principal: {e}")
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Apaga el sistema y libera recursos"""
        logger.info("=" * 60)
        logger.info("🔒 Apagando sistema...")
        
        self.is_running = False
        
        # Apagar asistente de voz
        if self.voice_assistant:
            self.voice_assistant.shutdown()
        
        # Liberar captura de video
        if self.video_capture:
            self.video_capture.release()
        
        # Cerrar ventanas
        if self.ui_renderer:
            self.ui_renderer.destroy_window()
        
        # Cerrar conexiones de base de datos
        db_manager.close_all_connections()
        
        logger.info("✅ Sistema apagado correctamente")
        logger.info("=" * 60)


def main():
    """
    Función principal
    """
    try:
        # Mostrar menú principal
        menu = MenuPrincipal()
        seleccion = menu.mostrar()

        camera_text = None
        if isinstance(seleccion, tuple) and len(seleccion) == 2:
            seleccion, camera_text = seleccion
        
        if seleccion == 'admin':
            # Abrir panel de administración
            panel = PanelAdmin()
            panel.mostrar()
            # Volver a mostrar menú después de cerrar admin
            main()
            
        elif seleccion == 'iniciar':
            default_source = CAMERA_CONFIG['source']
            camera_source = _normalize_camera_source(camera_text or "", default_source)
            logger.info(f"📷 Fuente de cámara seleccionada: {camera_source}")

            # Iniciar sistema POS
            pos_system = POSSystem(camera_source=camera_source)
            pos_system.run()
            
        elif seleccion == 'salir':
            logger.info("👋 Hasta pronto!")
            
    except Exception as e:
        logger.error(f"❌ Error fatal: {e}")
        raise


if __name__ == "__main__":
    main()