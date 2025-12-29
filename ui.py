"""
Módulo de Interfaz Visual
Maneja la visualización de detecciones y el carrito de compras
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from config import UI_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UIRenderer:
    """
    Renderizador de interfaz de usuario para el sistema POS
    Dibuja detecciones, carrito de compras e información del sistema
    """
    
    def __init__(self):
        """Inicializa el renderizador"""
        self.window_name = UI_CONFIG['window_name']
        self.font = cv2.FONT_HERSHEY_DUPLEX  # Fuente más nítida
        self.font_scale = 0.8  # Aumentado de 0.6
        self.font_thickness = 2  # Aumentado de 1
        self.bbox_color = UI_CONFIG['bbox_color']
        self.text_color = (255, 255, 255)  # Blanco puro
        
        # Estado de la interfaz
        self.show_cart = True
        self.show_prices = True
        self.listening_status = False
    
    def draw_detection_info(self, frame: np.ndarray, detections: List[Dict], 
                           prices_map: Dict[str, float]) -> np.ndarray:
        """
        Dibuja información de detecciones sobre el frame
        (Bounding boxes, nombres y precios)
        
        Args:
            frame: Frame de entrada
            detections: Lista de detecciones
            prices_map: Mapa de productos a precios
            
        Returns:
            Frame con anotaciones
        """
        annotated_frame = frame.copy()
        
        for detection in detections:
            class_name = detection['class_name']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Extraer coordenadas
            x1, y1, x2, y2 = map(int, bbox)
            
            # Obtener precio del mapa
            precio = prices_map.get(class_name, 0.0)
            
            # Dibujar bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), self.bbox_color, 2)
            
            # Preparar etiqueta
            if self.show_prices and precio > 0:
                label = f"{class_name} - ${precio:.2f} ({confidence:.2f})"
            else:
                label = f"{class_name} ({confidence:.2f})"
            
            # Calcular tamaño del texto
            (text_width, text_height), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Dibujar fondo para el texto
            cv2.rectangle(
                annotated_frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                self.bbox_color,
                -1  # Relleno
            )
            
            # Dibujar texto
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - baseline - 5),
                self.font,
                self.font_scale,
                self.text_color,
                self.font_thickness
            )
        
        return annotated_frame
    
    def draw_shopping_cart(self, frame: np.ndarray, cart_items: Dict[str, Dict],
                          total: float) -> np.ndarray:
        """
        Dibuja el carrito de compras en el frame
        
        Args:
            frame: Frame de entrada
            cart_items: Diccionario de items {nombre: {cantidad, precio, subtotal}}
            total: Total de la compra
            
        Returns:
            Frame con el carrito dibujado
        """
        if not self.show_cart:
            return frame
        
        h, w = frame.shape[:2]
        cart_width = 350
        cart_height = min(400, h - 100)
        
        # Posición del carrito (esquina superior derecha)
        cart_x = w - cart_width - 10
        cart_y = 10
        
        # Crear overlay para semi-transparencia
        overlay = frame.copy()
        
        # Dibujar fondo del carrito
        cv2.rectangle(
            overlay,
            (cart_x, cart_y),
            (cart_x + cart_width, cart_y + cart_height),
            UI_CONFIG['cart_bg_color'],
            -1
        )
        
        # Aplicar transparencia
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Título del carrito
        title = "CARRITO DE COMPRAS"
        cv2.putText(
            frame,
            title,
            (cart_x + 10, cart_y + 35),
            self.font,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA  # Anti-aliasing
        )
        
        # Línea separadora
        cv2.line(
            frame,
            (cart_x + 10, cart_y + 40),
            (cart_x + cart_width - 10, cart_y + 40),
            (255, 255, 255),
            1
        )
        
        # Dibujar items
        y_offset = cart_y + 60
        line_height = 25
        
        if not cart_items:
            cv2.putText(
                frame,
                "Carrito vacio",
                (cart_x + 10, y_offset),
                self.font,
                0.7,
                (200, 200, 200),
                2,
                cv2.LINE_AA
            )
        else:
            for producto, info in cart_items.items():
                cantidad = info['cantidad']
                precio = info['precio']
                subtotal = info['subtotal']
                
                # Texto del producto (truncar si es muy largo)
                producto_text = producto[:25] + "..." if len(producto) > 25 else producto
                
                # Línea 1: Nombre y cantidad
                cv2.putText(
                    frame,
                    f"{producto_text} x{cantidad}",
                    (cart_x + 10, y_offset),
                    self.font,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # Línea 2: Subtotal
                cv2.putText(
                    frame,
                    f"${subtotal:.2f}",
                    (cart_x + cart_width - 90, y_offset),
                    self.font,
                    0.65,
                    (0, 255, 0),
                    1
                )
                
                y_offset += line_height
                
                # Verificar si se excede el espacio
                if y_offset > cart_y + cart_height - 80:
                    cv2.putText(
                        frame,
                        "...",
                        (cart_x + 10, y_offset),
                        self.font,
                        0.5,
                        (200, 200, 200),
                        1
                    )
                    break
        
        # Línea separadora antes del total
        cv2.line(
            frame,
            (cart_x + 10, cart_y + cart_height - 60),
            (cart_x + cart_width - 10, cart_y + cart_height - 60),
            (255, 255, 255),
            2
        )
        
        # Total
        cv2.putText(
            frame,
            "TOTAL:",
            (cart_x + 10, cart_y + cart_height - 30),
            self.font,
            0.7,
            (255, 255, 255),
            2
        )
        
        cv2.putText(
            frame,
            f"${total:.2f}",
            (cart_x + cart_width - 100, cart_y + cart_height - 30),
            self.font,
            0.8,
            (0, 255, 0),
            2
        )
        
        return frame
    
    def draw_status_bar(self, frame: np.ndarray, listening: bool = False,
                       fps: float = 0.0, detection_count: int = 0) -> np.ndarray:
        """
        Dibuja una barra de estado en la parte inferior
        
        Args:
            frame: Frame de entrada
            listening: Si el asistente está escuchando
            fps: Frames por segundo
            detection_count: Número de detecciones actuales
            
        Returns:
            Frame con barra de estado
        """
        h, w = frame.shape[:2]
        bar_height = 40
        
        # Dibujar fondo de la barra
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, h - bar_height),
            (w, h),
            (30, 30, 30),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Estado de escucha
        if listening:
            status_text = "🎤 ESCUCHANDO..."
            status_color = (0, 255, 0)
        else:
            status_text = "⏸️ Voz: Inactiva"
            status_color = (100, 100, 100)
        
        cv2.putText(
            frame,
            status_text,
            (10, h - 12),
            self.font,
            0.5,
            status_color,
            1
        )
        
        # FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            frame,
            fps_text,
            (w - 150, h - 12),
            self.font,
            0.5,
            (255, 255, 255),
            1
        )
        
        # Número de detecciones
        det_text = f"Productos: {detection_count}"
        cv2.putText(
            frame,
            det_text,
            (w // 2 - 50, h - 12),
            self.font,
            0.5,
            (255, 255, 0),
            1
        )
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja instrucciones de uso en el frame
        
        Args:
            frame: Frame de entrada
            
        Returns:
            Frame con instrucciones
        """
        instructions = [
            "Presiona 'ESC' para salir",
            "Di 'LISTO' para finalizar compra",
            "Presiona 'C' para limpiar carrito"
        ]
        
        y_offset = 30
        for instruction in instructions:
            cv2.putText(
                frame,
                instruction,
                (10, y_offset),
                self.font,
                0.4,
                (200, 200, 200),
                1
            )
            y_offset += 20
        
        return frame
    
    def show_frame(self, frame: np.ndarray):
        """
        Muestra el frame en una ventana
        
        Args:
            frame: Frame a mostrar
        """
        cv2.imshow(self.window_name, frame)
    
    def create_window(self):
        """Crea la ventana principal"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        logger.info(f"✅ Ventana creada: {self.window_name}")
    
    def destroy_window(self):
        """Destruye todas las ventanas"""
        cv2.destroyAllWindows()
        logger.info("🔒 Ventanas cerradas")
    
    def wait_key(self, delay: int = 1) -> int:
        """
        Espera por una tecla presionada
        
        Args:
            delay: Tiempo de espera en milisegundos
            
        Returns:
            Código de la tecla presionada
        """
        return cv2.waitKey(delay) & 0xFF


class ShoppingCart:
    """
    Clase para manejar el carrito de compras
    """
    
    def __init__(self):
        """Inicializa el carrito vacío"""
        self.items: Dict[str, Dict] = {}
        self.total: float = 0.0
    
    def update(self, productos: List[str], prices_map: Dict[str, float]):
        """
        Actualiza el carrito con los productos detectados
        
        Args:
            productos: Lista de nombres de productos detectados
            prices_map: Mapa de productos a precios
        """
        self.items.clear()
        self.total = 0.0
        
        # Contar productos
        from collections import Counter
        product_counts = Counter(productos)
        
        # Actualizar items
        for producto, cantidad in product_counts.items():
            precio = prices_map.get(producto, 0.0)
            subtotal = precio * cantidad
            
            self.items[producto] = {
                'cantidad': cantidad,
                'precio': precio,
                'subtotal': subtotal
            }
            
            self.total += subtotal
    
    def clear(self):
        """Limpia el carrito"""
        self.items.clear()
        self.total = 0.0
        logger.info("🛒 Carrito limpiado")
    
    def get_product_list(self) -> List[str]:
        """
        Obtiene lista de productos en el carrito
        
        Returns:
            Lista de nombres de productos
        """
        return list(self.items.keys())
    
    def get_total(self) -> float:
        """
        Obtiene el total del carrito
        
        Returns:
            Total en dólares
        """
        return self.total
    
    def is_empty(self) -> bool:
        """
        Verifica si el carrito está vacío
        
        Returns:
            True si está vacío, False en caso contrario
        """
        return len(self.items) == 0
