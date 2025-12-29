"""
Módulo de Gestión de Base de Datos - PostgreSQL
Maneja todas las operaciones CRUD para inventario y ventas
"""

import psycopg2  # type: ignore
from psycopg2 import pool  # type: ignore
from psycopg2.extras import RealDictCursor  # type: ignore
from datetime import datetime
from typing import List, Dict, Optional
import logging
from config import DATABASE_CONFIG

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Gestor centralizado de conexiones a PostgreSQL con pool de conexiones
    """
    
    def __init__(self):
        """Inicializa el pool de conexiones"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 10,  # Mínimo y máximo de conexiones
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port'],
                user=DATABASE_CONFIG['user'],
                password=DATABASE_CONFIG['password'],
                database=DATABASE_CONFIG['database']
            )
            logger.info("✅ Pool de conexiones creado exitosamente")
        except Exception as e:
            logger.error(f"❌ Error al crear pool de conexiones: {e}")
            raise
    
    def get_connection(self):
        """Obtiene una conexión del pool, reabre si está cerrado"""
        if self.connection_pool is None or self.connection_pool.closed:
            logger.warning("⚠️ Pool cerrado, reabriendo...")
            self._create_pool()
        return self.connection_pool.getconn()
    
    def return_connection(self, connection):
        """Devuelve una conexión al pool"""
        self.connection_pool.putconn(connection)
    
    def close_all_connections(self):
        """Cierra todas las conexiones del pool"""
        if self.connection_pool and not self.connection_pool.closed:
            self.connection_pool.closeall()
            logger.info("🔒 Todas las conexiones cerradas")
    
    # ==================== OPERACIONES DE INVENTARIO ====================
    
    def get_product_by_name(self, nombre_producto: str) -> Optional[Dict]:
        """
        Obtiene información de un producto por su nombre
        
        Args:
            nombre_producto: Nombre del producto
            
        Returns:
            Diccionario con datos del producto o None si no existe
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT id, nombre_producto, precio, stock 
                FROM inventario 
                WHERE LOWER(nombre_producto) = LOWER(%s)
            """
            cursor.execute(query, (nombre_producto,))
            result = cursor.fetchone()
            
            cursor.close()
            return dict(result) if result else None
            
        except Exception as e:
            logger.error(f"❌ Error al obtener producto '{nombre_producto}': {e}")
            return None
        finally:
            if connection:
                self.return_connection(connection)
    
    def get_all_products(self) -> List[Dict]:
        """
        Obtiene todos los productos del inventario
        
        Returns:
            Lista de diccionarios con todos los productos
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            query = "SELECT id, nombre_producto, precio, stock FROM inventario ORDER BY nombre_producto"
            cursor.execute(query)
            results = cursor.fetchall()
            
            cursor.close()
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"❌ Error al obtener todos los productos: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)
    
    def update_stock(self, nombre_producto: str, cantidad: int) -> bool:
        """
        Actualiza el stock de un producto
        
        Args:
            nombre_producto: Nombre del producto
            cantidad: Nueva cantidad en stock
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                UPDATE inventario 
                SET stock = %s 
                WHERE LOWER(nombre_producto) = LOWER(%s)
            """
            cursor.execute(query, (cantidad, nombre_producto))
            connection.commit()
            
            cursor.close()
            logger.info(f"✅ Stock actualizado para '{nombre_producto}': {cantidad}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al actualizar stock: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)
    
    def add_product(self, nombre_producto: str, precio: float, stock: int = 0) -> bool:
        """
        Agrega un nuevo producto al inventario
        
        Args:
            nombre_producto: Nombre del producto
            precio: Precio del producto
            stock: Cantidad inicial en stock
            
        Returns:
            True si se agregó correctamente, False en caso contrario
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                INSERT INTO inventario (nombre_producto, precio, stock) 
                VALUES (%s, %s, %s)
            """
            cursor.execute(query, (nombre_producto, precio, stock))
            connection.commit()
            
            cursor.close()
            logger.info(f"✅ Producto agregado: '{nombre_producto}' - ${precio}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al agregar producto: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)
    
    # ==================== OPERACIONES DE VENTAS ====================
    
    def register_sale(self, productos: List[str], total: float) -> bool:
        """
        Registra una venta en el historial
        
        Args:
            productos: Lista de nombres de productos vendidos
            total: Total de la venta
            
        Returns:
            True si se registró correctamente, False en caso contrario
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                INSERT INTO historial_ventas (fecha, total_pago, productos) 
                VALUES (%s, %s, %s)
            """
            productos_str = ', '.join(productos)
            cursor.execute(query, (datetime.now(), total, productos_str))
            connection.commit()
            
            cursor.close()
            logger.info(f"✅ Venta registrada: ${total:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error al registrar venta: {e}")
            if connection:
                connection.rollback()
            return False
        finally:
            if connection:
                self.return_connection(connection)
    
    def get_sales_history(self, limit: int = 50) -> List[Dict]:
        """
        Obtiene el historial de ventas
        
        Args:
            limit: Número máximo de registros a obtener
            
        Returns:
            Lista de diccionarios con el historial de ventas
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            query = """
                SELECT id, fecha, total_pago, productos 
                FROM historial_ventas 
                ORDER BY fecha DESC 
                LIMIT %s
            """
            cursor.execute(query, (limit,))
            results = cursor.fetchall()
            
            cursor.close()
            return [dict(row) for row in results]
            
        except Exception as e:
            logger.error(f"❌ Error al obtener historial de ventas: {e}")
            return []
        finally:
            if connection:
                self.return_connection(connection)
    
    def get_daily_sales_total(self) -> float:
        """
        Obtiene el total de ventas del día actual
        
        Returns:
            Total de ventas del día
        """
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT COALESCE(SUM(total_pago), 0) 
                FROM historial_ventas 
                WHERE DATE(fecha) = CURRENT_DATE
            """
            cursor.execute(query)
            total = cursor.fetchone()[0]
            
            cursor.close()
            return float(total)
            
        except Exception as e:
            logger.error(f"❌ Error al obtener total de ventas del día: {e}")
            return 0.0
        finally:
            if connection:
                self.return_connection(connection)


# Instancia singleton del gestor de base de datos
db_manager = DatabaseManager()
