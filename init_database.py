"""
Script de Inicialización de Base de Datos
Crea las tablas necesarias y carga datos iniciales de productos

Ejecutar este script antes de usar el sistema POS por primera vez:
    python init_database.py
"""

import torch
# Monkey-patch torch.load para usar weights_only=False (compatibilidad PyTorch 2.6+)
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import psycopg2  # type: ignore
from psycopg2 import sql  # type: ignore
import logging
from config import DATABASE_CONFIG, PRODUCT_CLASSES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """
    Inicializador de base de datos para el sistema POS
    """
    
    def __init__(self):
        """Inicializa el inicializador de BD"""
        self.config = DATABASE_CONFIG
        self.connection = None
    
    def connect(self):
        """Establece conexión con PostgreSQL"""
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database='postgres'  # Conectar a BD por defecto primero
            )
            self.connection.autocommit = True
            logger.info("✅ Conectado a PostgreSQL")
        except Exception as e:
            logger.error(f"❌ Error al conectar a PostgreSQL: {e}")
            raise
    
    def create_database(self):
        """Crea la base de datos si no existe"""
        try:
            cursor = self.connection.cursor()
            
            # Verificar si la BD existe
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.config['database'],)
            )
            exists = cursor.fetchone()
            
            if not exists:
                cursor.execute(
                    sql.SQL("CREATE DATABASE {}").format(
                        sql.Identifier(self.config['database'])
                    )
                )
                logger.info(f"✅ Base de datos '{self.config['database']}' creada")
            else:
                logger.info(f"ℹ️ Base de datos '{self.config['database']}' ya existe")
            
            cursor.close()
            
            # Reconectar a la nueva base de datos
            self.connection.close()
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            self.connection.autocommit = False
            logger.info(f"✅ Conectado a base de datos '{self.config['database']}'")
            
        except Exception as e:
            logger.error(f"❌ Error al crear base de datos: {e}")
            raise
    
    def create_tables(self):
        """Crea las tablas necesarias"""
        try:
            cursor = self.connection.cursor()
            
            # Tabla de inventario
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS inventario (
                    id SERIAL PRIMARY KEY,
                    nombre_producto VARCHAR(100) UNIQUE NOT NULL,
                    precio DECIMAL(10, 2) NOT NULL,
                    stock INTEGER NOT NULL DEFAULT 0,
                    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    fecha_actualizacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("✅ Tabla 'inventario' creada/verificada")
            
            # Tabla de historial de ventas
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS historial_ventas (
                    id SERIAL PRIMARY KEY,
                    fecha TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    total_pago DECIMAL(10, 2) NOT NULL,
                    productos TEXT NOT NULL,
                    metodo_pago VARCHAR(50) DEFAULT 'efectivo'
                )
            """)
            logger.info("✅ Tabla 'historial_ventas' creada/verificada")
            
            # Índices para mejorar rendimiento
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_inventario_nombre 
                ON inventario(nombre_producto)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ventas_fecha 
                ON historial_ventas(fecha)
            """)
            
            self.connection.commit()
            logger.info("✅ Tablas e índices creados exitosamente")
            
            cursor.close()
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"❌ Error al crear tablas: {e}")
            raise
    
    def insert_initial_products(self):
        """Inserta productos iniciales con precios"""
        try:
            cursor = self.connection.cursor()
            
            # Precios iniciales para los productos
            initial_products = {
                'Borrador de ballena': 0.50,
                'Borrador de sirena': 0.50,
                'Esfero Negro': 0.35,
                'Flash Kingston 4GB': 8.50,
                'Flash Verbatim 16Gb': 12.00,
                'Pasador Cabello Minimouse': 1.25,
                'Resaltador': 0.75,
                'cartera': 15.00,
                'perfume': 25.00
            }
            
            inserted_count = 0
            
            for producto, precio in initial_products.items():
                try:
                    cursor.execute("""
                        INSERT INTO inventario (nombre_producto, precio, stock)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (nombre_producto) DO NOTHING
                    """, (producto, precio, 100))  # Stock inicial de 100 unidades
                    
                    if cursor.rowcount > 0:
                        inserted_count += 1
                        logger.info(f"  ✓ {producto}: ${precio:.2f}")
                    
                except Exception as e:
                    logger.warning(f"  ⚠️ Producto '{producto}' ya existe o error: {e}")
            
            self.connection.commit()
            logger.info(f"✅ {inserted_count} productos nuevos insertados")
            
            # Mostrar resumen de productos
            cursor.execute("SELECT COUNT(*) FROM inventario")
            total_productos = cursor.fetchone()[0]
            logger.info(f"📦 Total de productos en inventario: {total_productos}")
            
            cursor.close()
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"❌ Error al insertar productos: {e}")
            raise
    
    def show_database_info(self):
        """Muestra información de la base de datos"""
        try:
            cursor = self.connection.cursor()
            
            logger.info("\n" + "=" * 60)
            logger.info("📊 INFORMACIÓN DE LA BASE DE DATOS")
            logger.info("=" * 60)
            
            # Productos en inventario
            cursor.execute("""
                SELECT nombre_producto, precio, stock 
                FROM inventario 
                ORDER BY nombre_producto
            """)
            productos = cursor.fetchall()
            
            logger.info(f"\n🛍️ PRODUCTOS EN INVENTARIO ({len(productos)}):")
            logger.info("-" * 60)
            for nombre, precio, stock in productos:
                logger.info(f"  • {nombre:30s} ${precio:6.2f}  Stock: {stock}")
            
            # Ventas registradas
            cursor.execute("SELECT COUNT(*), COALESCE(SUM(total_pago), 0) FROM historial_ventas")
            num_ventas, total_ventas = cursor.fetchone()
            
            logger.info(f"\n💰 HISTORIAL DE VENTAS:")
            logger.info("-" * 60)
            logger.info(f"  • Total de ventas registradas: {num_ventas}")
            logger.info(f"  • Monto total: ${total_ventas:.2f}")
            
            logger.info("=" * 60 + "\n")
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"❌ Error al obtener información: {e}")
    
    def close(self):
        """Cierra la conexión"""
        if self.connection:
            self.connection.close()
            logger.info("🔒 Conexión cerrada")
    
    def initialize(self):
        """Ejecuta todo el proceso de inicialización"""
        logger.info("=" * 60)
        logger.info("🚀 INICIALIZACIÓN DE BASE DE DATOS")
        logger.info("=" * 60 + "\n")
        
        try:
            # 1. Conectar a PostgreSQL
            logger.info("📡 Paso 1: Conectando a PostgreSQL...")
            self.connect()
            
            # 2. Crear base de datos
            logger.info("\n📦 Paso 2: Creando base de datos...")
            self.create_database()
            
            # 3. Crear tablas
            logger.info("\n🏗️ Paso 3: Creando tablas...")
            self.create_tables()
            
            # 4. Insertar productos iniciales
            logger.info("\n🛍️ Paso 4: Insertando productos iniciales...")
            self.insert_initial_products()
            
            # 5. Mostrar información
            self.show_database_info()
            
            logger.info("✅ Inicialización completada exitosamente\n")
            
        except Exception as e:
            logger.error(f"\n❌ Error durante la inicialización: {e}\n")
            raise
        finally:
            self.close()


def main():
    """Función principal"""
    try:
        # Crear inicializador
        initializer = DatabaseInitializer()
        
        # Ejecutar inicialización
        initializer.initialize()
        
        print("\n" + "=" * 60)
        print("✅ BASE DE DATOS LISTA PARA USAR")
        print("=" * 60)
        print("\nAhora puedes ejecutar el sistema POS:")
        print("    python app.py")
        print("\n")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ ERROR EN LA INICIALIZACIÓN")
        print("=" * 60)
        print(f"\n{str(e)}\n")
        print("Verifica que:")
        print("  1. PostgreSQL esté instalado y corriendo")
        print("  2. Las credenciales en config.py sean correctas")
        print("  3. El usuario tenga permisos para crear bases de datos\n")


if __name__ == "__main__":
    main()
