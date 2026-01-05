"""
Menú Principal y Panel de Administración
Sistema POS Bazar Gulpery
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
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import threading
from database import db_manager
import logging
from config import CAMERA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MenuPrincipal:
    """Menú principal del sistema POS"""
    
    def __init__(self):
        self.seleccion = None
        self.ventana_activa = True

    def _default_camera_ip(self) -> str:
        """Extrae una IP (si es posible) desde CAMERA_CONFIG['source']."""
        src = str(CAMERA_CONFIG.get('source', '') or '')
        if not src:
            return ''
        if src.startswith('http://') or src.startswith('https://'):
            # http(s)://IP:PORT/...
            try:
                rest = src.split('://', 1)[1]
                hostport = rest.split('/', 1)[0]
                host = hostport.split(':', 1)[0]
                return host
            except Exception:
                return ''
        # Si está como IP:PORT o IP
        return src.split('/', 1)[0].split(':', 1)[0]

    def _prompt_camera_source(self, window_name: str) -> str | None:
        """Solicita URL/IP de cámara dentro de la misma ventana OpenCV.

        Returns:
            str: Texto ingresado (o default si Enter vacío)
            None: si el usuario cancela con ESC
        """
        default_ip = self._default_camera_ip()
        typed = ""

        while True:
            frame = np.zeros((700, 900, 3), dtype=np.uint8)

            # Fondo
            for i in range(700):
                color = int(15 + (i / 700) * 25)
                cv2.line(frame, (0, i), (900, i), (color, color, color + 8), 1)

            # Banner
            cv2.rectangle(frame, (0, 0), (900, 110), (40, 40, 60), -1)
            cv2.rectangle(frame, (0, 105), (900, 115), (0, 255, 150), -1)
            cv2.putText(frame, "CONFIGURAR CAMARA", (230, 70),
                        cv2.FONT_HERSHEY_DUPLEX, 1.6, (100, 255, 200), 3, cv2.LINE_AA)

            # Caja
            cv2.rectangle(frame, (120, 210), (780, 500), (55, 55, 55), -1)
            cv2.rectangle(frame, (120, 210), (780, 500), (120, 120, 120), 2)

            cv2.putText(frame, "Ingrese SOLO la IP de la camara:", (150, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)

            # Input box
            cv2.rectangle(frame, (150, 300), (750, 360), (30, 30, 30), -1)
            cv2.rectangle(frame, (150, 300), (750, 360), (0, 255, 150), 2)

            display_text = typed if typed else f"(Enter = default: {default_ip})"
            color = (255, 255, 255) if typed else (170, 170, 170)
            # Mostrar solo el final si es muy largo
            if len(display_text) > 60:
                display_text = "..." + display_text[-57:]
            cv2.putText(frame, display_text, (165, 340),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

            # Ayuda
            cv2.putText(frame, "Ej: 172.25.235.192   (se usara http://IP:8080/video)", (150, 410),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.putText(frame, "ENTER = continuar   ESC = volver", (285, 460),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (150, 255, 150), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(50) & 0xFF

            if key == 27:  # ESC
                return None
            if key in (10, 13):  # ENTER
                return typed.strip() if typed.strip() else str(default_ip)
            if key in (8, 127):  # BACKSPACE
                typed = typed[:-1]
                continue

            # Caracteres imprimibles
            if 32 <= key <= 126:
                if len(typed) < 120:
                    typed += chr(key)
        
    def mostrar(self):
        """Muestra el menú principal usando OpenCV"""
        # Crear ventana
        window_name = "Bazar Gulpery - Menu Principal"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 900, 700)
        
        while self.ventana_activa and self.seleccion is None:
            # Crear frame con degradado
            frame = np.zeros((700, 900, 3), dtype=np.uint8)
            
            # Fondo degradado sutil
            for i in range(700):
                color = int(20 + (i / 700) * 30)
                cv2.line(frame, (0, i), (900, i), (color, color, color+10), 1)
            
            # Banner superior decorativo
            cv2.rectangle(frame, (0, 0), (900, 120), (40, 40, 60), -1)
            cv2.rectangle(frame, (0, 115), (900, 125), (0, 255, 150), -1)
            
            # Título principal con sombra
            cv2.putText(frame, "BAZAR GULPERY", (240, 65),
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, (30, 30, 30), 5, cv2.LINE_AA)
            cv2.putText(frame, "BAZAR GULPERY", (235, 60),
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, (100, 255, 200), 4, cv2.LINE_AA)
            
            # Subtítulo
            cv2.putText(frame, "Sistema POS con Inteligencia Artificial", (225, 105),
                       cv2.FONT_HERSHEY_DUPLEX, 0.9, (180, 180, 180), 2, cv2.LINE_AA)
            
            # Botones centrados
            btn_width = 320
            btn_height = 140
            spacing = 50
            center_x = 450  # Centro de la ventana (900/2)
            start_x = center_x - btn_width - spacing // 2
            btn_y = 270
            
            # Botón ADMIN (Izquierdo)
            admin_x = start_x
            # Sombra
            cv2.rectangle(frame, (admin_x + 8, btn_y + 8),
                         (admin_x + btn_width + 8, btn_y + btn_height + 8),
                         (20, 20, 20), -1)
            # Fondo principal
            cv2.rectangle(frame, (admin_x, btn_y),
                         (admin_x + btn_width, btn_y + btn_height),
                         (45, 50, 180), -1)
            # Borde brillante
            cv2.rectangle(frame, (admin_x, btn_y),
                         (admin_x + btn_width, btn_y + btn_height),
                         (120, 130, 255), 4)
            # Borde interno
            cv2.rectangle(frame, (admin_x + 8, btn_y + 8),
                         (admin_x + btn_width - 8, btn_y + btn_height - 8),
                         (80, 90, 220), 2)
            
            # Texto ADMIN
            cv2.putText(frame, "ADMIN", (admin_x + 70, btn_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Panel de Control", (admin_x + 55, btn_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Presione  A", (admin_x + 70, btn_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2, cv2.LINE_AA)
            
            # Botón INICIAR (Derecho)
            iniciar_x = start_x + btn_width + spacing
            # Sombra
            cv2.rectangle(frame, (iniciar_x + 8, btn_y + 8),
                         (iniciar_x + btn_width + 8, btn_y + btn_height + 8),
                         (20, 20, 20), -1)
            # Fondo principal
            cv2.rectangle(frame, (iniciar_x, btn_y),
                         (iniciar_x + btn_width, btn_y + btn_height),
                         (40, 180, 50), -1)
            # Borde brillante
            cv2.rectangle(frame, (iniciar_x, btn_y),
                         (iniciar_x + btn_width, btn_y + btn_height),
                         (100, 255, 100), 4)
            # Borde interno
            cv2.rectangle(frame, (iniciar_x + 8, btn_y + 8),
                         (iniciar_x + btn_width - 8, btn_y + btn_height - 8),
                         (70, 220, 80), 2)
            
            # Texto INICIAR
            cv2.putText(frame, "INICIAR", (iniciar_x + 65, btn_y + 65),
                       cv2.FONT_HERSHEY_DUPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Sistema POS", (iniciar_x + 75, btn_y + 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, "Presione  I", (iniciar_x + 80, btn_y + 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 255, 150), 2, cv2.LINE_AA)
            
            # Decoración inferior
            cv2.line(frame, (150, 530), (750, 530), (0, 255, 150), 2)
            
            # Iconos decorativos
            cv2.circle(frame, (100, 300), 15, (0, 255, 150), -1)
            cv2.circle(frame, (800, 300), 15, (0, 255, 150), -1)
            
            # Instrucciones de salida
            cv2.rectangle(frame, (300, 570), (600, 630), (60, 60, 60), -1)
            cv2.rectangle(frame, (300, 570), (600, 630), (100, 100, 100), 2)
            cv2.putText(frame, "ESC - Salir del sistema", (335, 607),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Footer
            cv2.putText(frame, "v2.0 - Sistema Inteligente de Punto de Venta", (280, 680),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)
            
            cv2.imshow(window_name, frame)
            
            # Detectar teclas
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('a') or key == ord('A'):
                self.seleccion = 'admin'
                logger.info("🔧 Accediendo a panel de administración...")
            elif key == ord('i') or key == ord('I'):
                camera_text = self._prompt_camera_source(window_name)
                if camera_text is None:
                    continue
                self.seleccion = ('iniciar', camera_text)
                logger.info("🚀 Iniciando sistema POS...")
            elif key == 27:  # ESC
                self.seleccion = 'salir'
                logger.info("👋 Saliendo del sistema...")
        
        cv2.destroyWindow(window_name)
        return self.seleccion


class PanelAdmin:
    """Panel de administración para modificar precios"""
    
    def __init__(self):
        self.root = None
        self.productos = []
        
    def cargar_productos(self):
        """Carga productos desde la base de datos"""
        try:
            productos_data = db_manager.get_all_products()
            if not productos_data:
                logger.warning("⚠️ No hay productos en la base de datos")
                return False
            
            self.productos = [
                {
                    'nombre': p.get('nombre_producto', ''),
                    'precio': float(p.get('precio', 0)),
                    'stock': int(p.get('stock', 0))
                }
                for p in productos_data
            ]
            logger.info(f"✅ {len(self.productos)} productos cargados")
            return True
        except Exception as e:
            logger.error(f"❌ Error al cargar productos: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def actualizar_precio(self, nombre_producto, nuevo_precio):
        """Actualiza el precio de un producto"""
        connection = None
        try:
            connection = db_manager.get_connection()
            cursor = connection.cursor()
            cursor.execute("""
                UPDATE inventario 
                SET precio = %s 
                WHERE nombre_producto = %s
            """, (nuevo_precio, nombre_producto))
            connection.commit()
            cursor.close()
            logger.info(f"✅ Precio actualizado: {nombre_producto} = ${nuevo_precio:.2f}")
            return True
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"❌ Error al actualizar precio: {e}")
            return False
        finally:
            if connection:
                db_manager.return_connection(connection)
    
    def mostrar(self):
        """Muestra el panel de administración usando Tkinter"""
        self.root = tk.Tk()
        self.root.title("Panel de Administración - Bazar Gulpery")
        self.root.geometry("700x500")
        self.root.configure(bg='#2c3e50')
        
        # Cargar productos
        if not self.cargar_productos():
            messagebox.showerror("Error", "No se pudieron cargar los productos")
            self.root.destroy()
            return
        
        # Título
        titulo = tk.Label(
            self.root,
            text="ADMINISTRACIÓN DE PRECIOS",
            font=("Arial", 18, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        titulo.pack(pady=20)
        
        # Frame para la tabla
        frame_tabla = tk.Frame(self.root, bg='#2c3e50')
        frame_tabla.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(frame_tabla)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Treeview (tabla)
        columnas = ('Producto', 'Precio Actual', 'Stock')
        self.tree = ttk.Treeview(
            frame_tabla,
            columns=columnas,
            show='headings',
            yscrollcommand=scrollbar.set,
            height=15
        )
        
        # Configurar columnas
        self.tree.heading('Producto', text='Producto')
        self.tree.heading('Precio Actual', text='Precio Actual ($)')
        self.tree.heading('Stock', text='Stock')
        
        self.tree.column('Producto', width=350)
        self.tree.column('Precio Actual', width=150, anchor='center')
        self.tree.column('Stock', width=100, anchor='center')
        
        # Insertar datos
        for p in self.productos:
            self.tree.insert('', tk.END, values=(p['nombre'], f"${p['precio']:.2f}", p['stock']))
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)
        
        # Frame para botones
        frame_botones = tk.Frame(self.root, bg='#2c3e50')
        frame_botones.pack(pady=15)
        
        # Botón modificar precio
        btn_modificar = tk.Button(
            frame_botones,
            text="Modificar Precio",
            command=self.modificar_precio_dialog,
            font=("Arial", 12, "bold"),
            bg='#3498db',
            fg='white',
            width=15,
            height=2
        )
        btn_modificar.pack(side=tk.LEFT, padx=10)
        
        # Botón cerrar
        btn_cerrar = tk.Button(
            frame_botones,
            text="Cerrar",
            command=self.root.destroy,
            font=("Arial", 12, "bold"),
            bg='#e74c3c',
            fg='white',
            width=15,
            height=2
        )
        btn_cerrar.pack(side=tk.LEFT, padx=10)
        
        self.root.mainloop()
    
    def modificar_precio_dialog(self):
        """Diálogo para modificar el precio de un producto"""
        seleccion = self.tree.selection()
        if not seleccion:
            messagebox.showwarning("Advertencia", "Selecciona un producto primero")
            return
        
        item = self.tree.item(seleccion[0])
        nombre_producto = item['values'][0]
        precio_actual = float(item['values'][1].replace('$', ''))
        
        # Crear ventana de diálogo
        dialog = tk.Toplevel(self.root)
        dialog.title("Modificar Precio")
        dialog.geometry("400x250")
        dialog.configure(bg='#34495e')
        
        # Centrar ventana
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Contenido
        tk.Label(
            dialog,
            text=f"Producto: {nombre_producto}",
            font=("Arial", 12, "bold"),
            bg='#34495e',
            fg='white'
        ).pack(pady=15)
        
        tk.Label(
            dialog,
            text=f"Precio actual: ${precio_actual:.2f}",
            font=("Arial", 11),
            bg='#34495e',
            fg='#ecf0f1'
        ).pack(pady=5)
        
        tk.Label(
            dialog,
            text="Nuevo precio:",
            font=("Arial", 11, "bold"),
            bg='#34495e',
            fg='white'
        ).pack(pady=10)
        
        # Entry para nuevo precio
        entry_precio = tk.Entry(dialog, font=("Arial", 14), width=15, justify='center')
        entry_precio.pack(pady=5)
        entry_precio.insert(0, f"{precio_actual:.2f}")
        entry_precio.focus()
        entry_precio.select_range(0, tk.END)
        
        def guardar():
            try:
                nuevo_precio = float(entry_precio.get())
                if nuevo_precio <= 0:
                    messagebox.showerror("Error", "El precio debe ser mayor a 0")
                    return
                
                if self.actualizar_precio(nombre_producto, nuevo_precio):
                    # Actualizar tabla
                    self.tree.item(seleccion[0], values=(nombre_producto, f"${nuevo_precio:.2f}", item['values'][2]))
                    messagebox.showinfo("Éxito", f"Precio actualizado a ${nuevo_precio:.2f}")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "No se pudo actualizar el precio")
            except ValueError:
                messagebox.showerror("Error", "Ingresa un precio válido")
        
        # Botones
        frame_btn = tk.Frame(dialog, bg='#34495e')
        frame_btn.pack(pady=15)
        
        tk.Button(
            frame_btn,
            text="Guardar",
            command=guardar,
            font=("Arial", 11, "bold"),
            bg='#27ae60',
            fg='white',
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            frame_btn,
            text="Cancelar",
            command=dialog.destroy,
            font=("Arial", 11, "bold"),
            bg='#95a5a6',
            fg='white',
            width=10
        ).pack(side=tk.LEFT, padx=5)
        
        # Enter para guardar
        entry_precio.bind('<Return>', lambda e: guardar())
