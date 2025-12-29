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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MenuPrincipal:
    """Menú principal del sistema POS"""
    
    def __init__(self):
        self.seleccion = None
        self.ventana_activa = True
        
    def mostrar(self):
        """Muestra el menú principal usando OpenCV"""
        # Crear ventana
        window_name = "Bazar Gulpery - Menu Principal"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        while self.ventana_activa and self.seleccion is None:
            # Crear frame negro
            frame = np.zeros((600, 800, 3), dtype=np.uint8)
            
            # Título
            cv2.putText(frame, "BAZAR GULPERY", (180, 100),
                       cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Sistema POS con IA", (220, 150),
                       cv2.FONT_HERSHEY_DUPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA)
            
            # Línea decorativa
            cv2.line(frame, (100, 180), (700, 180), (0, 255, 0), 2)
            
            # Botón ADMIN
            admin_rect = (150, 250, 300, 120)
            cv2.rectangle(frame, (admin_rect[0], admin_rect[1]),
                         (admin_rect[0] + admin_rect[2], admin_rect[1] + admin_rect[3]),
                         (50, 50, 200), -1)
            cv2.rectangle(frame, (admin_rect[0], admin_rect[1]),
                         (admin_rect[0] + admin_rect[2], admin_rect[1] + admin_rect[3]),
                         (100, 100, 255), 3)
            cv2.putText(frame, "ADMIN", (220, 325),
                       cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Presiona 'A'", (190, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Botón INICIAR
            iniciar_rect = (500, 250, 300, 120)
            cv2.rectangle(frame, (iniciar_rect[0], iniciar_rect[1]),
                         (iniciar_rect[0] + iniciar_rect[2], iniciar_rect[1] + iniciar_rect[3]),
                         (50, 200, 50), -1)
            cv2.rectangle(frame, (iniciar_rect[0], iniciar_rect[1]),
                         (iniciar_rect[0] + iniciar_rect[2], iniciar_rect[1] + iniciar_rect[3]),
                         (100, 255, 100), 3)
            cv2.putText(frame, "INICIAR", (550, 325),
                       cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "Presiona 'I'", (545, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            
            # Instrucciones
            cv2.putText(frame, "Presiona ESC para salir", (270, 500),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)
            
            cv2.imshow(window_name, frame)
            
            # Detectar teclas
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('a') or key == ord('A'):
                self.seleccion = 'admin'
                logger.info("🔧 Accediendo a panel de administración...")
            elif key == ord('i') or key == ord('I'):
                self.seleccion = 'iniciar'
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
