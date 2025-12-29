# 🎓 MEJORES PRÁCTICAS Y GUÍA AVANZADA

## 📚 Tabla de Contenidos

1. [Configuración Óptima](#configuración-óptima)
2. [Optimización de Rendimiento](#optimización-de-rendimiento)
3. [Seguridad](#seguridad)
4. [Mantenimiento](#mantenimiento)
5. [Resolución de Problemas](#resolución-de-problemas)
6. [Personalización Avanzada](#personalización-avanzada)

---

## ⚙️ Configuración Óptima

### 1. Base de Datos

**Configuración Recomendada para PostgreSQL:**

```ini
# postgresql.conf
max_connections = 100
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
```

**Índices adicionales para mejor rendimiento:**

```sql
-- Índice para búsquedas por precio
CREATE INDEX idx_inventario_precio ON inventario(precio);

-- Índice compuesto para reportes
CREATE INDEX idx_ventas_fecha_total ON historial_ventas(fecha, total_pago);

-- Índice para búsquedas de texto
CREATE INDEX idx_inventario_nombre_trgm ON inventario 
USING gin(nombre_producto gin_trgm_ops);
```

### 2. Modelo YOLO

**Ajustar umbrales según ambiente:**

```python
# Para ambientes con buena iluminación
MODEL_CONFIG = {
    'confidence_threshold': 0.5,  # Más estricto
    'iou_threshold': 0.45
}

# Para ambientes con poca luz
MODEL_CONFIG = {
    'confidence_threshold': 0.35,  # Más permisivo
    'iou_threshold': 0.50
}
```

### 3. Cámara

**Resoluciones recomendadas:**

- **Alta precisión:** 1920x1080 (puede reducir FPS)
- **Balance:** 1280x720 (recomendado)
- **Rápido:** 640x480 (mayor FPS, menor precisión)

**Configurar en [config.py](config.py):**

```python
CAMERA_CONFIG = {
    'source': 'http://192.168.100.11:8080/video',
    'width': 1280,
    'height': 720,
    'fps': 30
}
```

---

## 🚀 Optimización de Rendimiento

### 1. Usar GPU para YOLO

Instalar PyTorch con soporte CUDA:

```bash
# Para NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verificar GPU en código:

```python
from ultralytics import YOLO
import torch

print(f"GPU disponible: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Modelo usará GPU automáticamente si está disponible
model = YOLO('best.pt')
```

### 2. Caché de Precios

El sistema ya implementa caché de precios en memoria. Para sistemas grandes:

```python
# En database.py, agregar caché con TTL
from functools import lru_cache
import time

@lru_cache(maxsize=128)
def get_product_price_cached(nombre_producto: str, cache_time: int):
    """Caché con TTL de 5 minutos"""
    return db_manager.get_product_by_name(nombre_producto)

# Usar con timestamp cada 5 minutos
cache_time = int(time.time() / 300)
precio = get_product_price_cached("Esfero Negro", cache_time)
```

### 3. Procesamiento Asíncrono

Para sistemas multi-cámara, usar threading:

```python
import threading
from queue import Queue

def process_camera(camera_id, frame_queue):
    cap = VideoCapture(f"http://camera_{camera_id}/video")
    while True:
        ret, frame = cap.read()
        if ret:
            frame_queue.put((camera_id, frame))

# Crear hilos para múltiples cámaras
queues = [Queue() for _ in range(3)]
threads = [threading.Thread(target=process_camera, args=(i, queues[i])) 
           for i in range(3)]
```

### 4. Optimizar Detecciones

Reducir tamaño de imagen para inferencia:

```python
# En detector.py
def detect(self, frame: np.ndarray):
    # Redimensionar antes de inferencia
    resized = cv2.resize(frame, (640, 640))
    results = self.model(resized, ...)
    
    # Escalar detecciones al tamaño original
    scale_x = frame.shape[1] / 640
    scale_y = frame.shape[0] / 640
    # ... ajustar bboxes
```

---

## 🔒 Seguridad

### 1. Variables de Entorno

**Crear archivo `.env`:**

```env
# .env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=gulpery_secure_2025
DB_NAME=bazar_gulpery

CAMERA_URL=http://192.168.100.11:8080/video

ROBOFLOW_API_KEY=your_api_key_here
```

**Actualizar [config.py](config.py):**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'bazar_gulpery')
}
```

### 2. Conexión SSL a PostgreSQL

```python
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'user': 'postgres',
    'password': os.getenv('DB_PASSWORD'),
    'database': 'bazar_gulpery',
    'sslmode': 'require',  # Requerir SSL
    'sslcert': '/path/to/client-cert.pem',
    'sslkey': '/path/to/client-key.pem',
    'sslrootcert': '/path/to/ca-cert.pem'
}
```

### 3. Autenticación de Usuarios

Agregar tabla de usuarios:

```sql
CREATE TABLE usuarios (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    rol VARCHAR(20) NOT NULL DEFAULT 'cajero',
    activo BOOLEAN DEFAULT TRUE,
    fecha_creacion TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 4. Logs de Auditoría

```sql
CREATE TABLE auditoria (
    id SERIAL PRIMARY KEY,
    usuario_id INTEGER REFERENCES usuarios(id),
    accion VARCHAR(50) NOT NULL,
    tabla_afectada VARCHAR(50),
    registro_id INTEGER,
    detalles JSONB,
    fecha TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔧 Mantenimiento

### 1. Respaldo de Base de Datos

**Backup diario automático:**

```bash
# backup.sh
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -U postgres bazar_gulpery > backup_$DATE.sql
gzip backup_$DATE.sql

# Mantener solo últimos 30 días
find . -name "backup_*.sql.gz" -mtime +30 -delete
```

**Configurar tarea programada en Windows:**

```powershell
# backup.ps1
$date = Get-Date -Format "yyyyMMdd_HHmmss"
$file = "backup_$date.sql"
pg_dump -U postgres bazar_gulpery > $file
Compress-Archive -Path $file -DestinationPath "$file.zip"
Remove-Item $file
```

### 2. Limpieza de Logs

```python
# En utils.py
def clean_old_logs(days=30):
    """Elimina logs antiguos"""
    import glob
    import time
    
    cutoff = time.time() - (days * 86400)
    
    for log_file in glob.glob("logs/*.log"):
        if os.path.getmtime(log_file) < cutoff:
            os.remove(log_file)
            logger.info(f"Log eliminado: {log_file}")
```

### 3. Monitoreo de Sistema

```python
# monitor.py
import psutil
import logging

def check_system_health():
    """Verifica salud del sistema"""
    cpu = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('/').percent
    
    if cpu > 80:
        logging.warning(f"CPU alta: {cpu}%")
    if memory > 85:
        logging.warning(f"Memoria alta: {memory}%")
    if disk > 90:
        logging.warning(f"Disco casi lleno: {disk}%")
```

### 4. Actualización de Precios

```python
# update_prices.py
import pandas as pd
from database import db_manager

def update_prices_from_csv(csv_file):
    """Actualiza precios desde CSV"""
    df = pd.read_csv(csv_file)
    
    for _, row in df.iterrows():
        # Aquí necesitas implementar update_price en database.py
        success = db_manager.update_product_price(
            row['nombre_producto'],
            row['nuevo_precio']
        )
        
        if success:
            print(f"✓ {row['nombre_producto']}: ${row['nuevo_precio']}")
```

---

## 🔍 Resolución de Problemas

### 1. Detecciones Inestables

**Problema:** Productos parpadean en pantalla

**Solución:**
```python
# En config.py, aumentar caché
self.cache_size = 10  # De 5 a 10 frames
```

### 2. FPS Bajo

**Diagnóstico:**
```python
import time

start = time.time()
results = model(frame)
inference_time = time.time() - start
print(f"Tiempo de inferencia: {inference_time*1000:.2f}ms")
```

**Soluciones:**
- Reducir resolución de entrada
- Usar modelo más pequeño (yolov8n vs yolov8s)
- Habilitar GPU
- Reducir confidence_threshold

### 3. Memoria Alta

**Verificar uso:**
```python
import tracemalloc

tracemalloc.start()
# ... código
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')

for stat in top_stats[:10]:
    print(stat)
```

**Soluciones:**
- Limpiar caché periódicamente
- Reducir pool de conexiones BD
- Liberar frames no usados

### 4. Voz No Reconoce

**Verificar:**
```bash
# Test de micrófono
python -m speech_recognition
```

**Ajustar sensibilidad:**
```python
# En voice_assistant.py
recognizer.energy_threshold = 4000  # Ajustar según ruido
recognizer.dynamic_energy_threshold = True
```

---

## 🎨 Personalización Avanzada

### 1. Tema de Colores

```python
# En config.py
UI_CONFIG = {
    # Tema Oscuro
    'bbox_color': (0, 255, 0),        # Verde
    'text_color': (255, 255, 255),     # Blanco
    'cart_bg_color': (30, 30, 30),     # Gris muy oscuro
    
    # Tema Claro
    # 'bbox_color': (0, 100, 200),
    # 'text_color': (0, 0, 0),
    # 'cart_bg_color': (240, 240, 240),
}
```

### 2. Múltiples Idiomas

```python
# languages.py
MESSAGES = {
    'es': {
        'sale_complete': 'Compra finalizada',
        'total': 'El total es',
        'thank_you': 'Gracias por su compra'
    },
    'en': {
        'sale_complete': 'Purchase complete',
        'total': 'The total is',
        'thank_you': 'Thank you for your purchase'
    }
}
```

### 3. Notificaciones

```python
# notifications.py
import smtplib
from email.mime.text import MIMEText

def send_low_stock_alert(producto, stock):
    """Envía alerta de stock bajo"""
    if stock < 10:
        msg = MIMEText(f"Stock bajo: {producto} - {stock} unidades")
        msg['Subject'] = '⚠️ Alerta de Stock Bajo'
        msg['From'] = 'sistema@bazargulpery.com'
        msg['To'] = 'admin@bazargulpery.com'
        
        # Enviar email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login('user', 'password')
            server.send_message(msg)
```

### 4. Dashboard Web

```python
# app_web.py (Flask)
from flask import Flask, render_template, jsonify
from database import db_manager

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/sales/today')
def sales_today():
    total = db_manager.get_daily_sales_total()
    return jsonify({'total': total})

@app.route('/api/products')
def products():
    productos = db_manager.get_all_products()
    return jsonify(productos)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📊 Métricas y Analytics

### 1. KPIs del Sistema

```python
def get_system_kpis():
    """Obtiene KPIs del sistema"""
    return {
        'total_ventas_dia': db_manager.get_daily_sales_total(),
        'productos_mas_vendidos': get_top_products(),
        'promedio_venta': get_average_sale(),
        'fps_promedio': calculate_avg_fps(),
        'precision_deteccion': get_detection_accuracy()
    }
```

### 2. Reportes Automáticos

```python
# report_generator.py
from datetime import datetime, timedelta
import pandas as pd

def generate_daily_report():
    """Genera reporte diario en Excel"""
    ventas = db_manager.get_sales_history(limit=1000)
    df = pd.DataFrame(ventas)
    
    # Filtrar último día
    today = datetime.now().date()
    df['fecha'] = pd.to_datetime(df['fecha']).dt.date
    df_today = df[df['fecha'] == today]
    
    # Guardar en Excel
    filename = f"reporte_{today}.xlsx"
    df_today.to_excel(filename, index=False)
    print(f"Reporte generado: {filename}")
```

---

## 🎯 Checklist de Producción

Antes de desplegar en producción:

- [ ] Cambiar todas las contraseñas por defecto
- [ ] Configurar variables de entorno
- [ ] Habilitar SSL para PostgreSQL
- [ ] Configurar backups automáticos
- [ ] Implementar sistema de logs
- [ ] Configurar alertas de stock
- [ ] Probar en condiciones reales
- [ ] Documentar configuración específica
- [ ] Capacitar al personal
- [ ] Establecer plan de mantenimiento

---

**¡Sistema listo para producción! 🚀**
