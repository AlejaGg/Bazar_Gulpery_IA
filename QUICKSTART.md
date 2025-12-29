# 🚀 Guía de Inicio Rápido - Sistema POS con IA

## ⏱️ Configuración en 5 Minutos

### 1️⃣ Instalar Dependencias (2 min)

```powershell
# Activar entorno virtual (opcional pero recomendado)
python -m venv venv
.\venv\Scripts\activate

# Instalar todas las dependencias
pip install -r requirements.txt
```

### 2️⃣ Configurar PostgreSQL (1 min)

Asegúrate de que PostgreSQL esté corriendo y ejecuta:

```powershell
python init_database.py
```

Esto creará:
- ✅ Base de datos `bazar_gulpery`
- ✅ Tabla `inventario` con 9 productos y precios
- ✅ Tabla `historial_ventas`

### 3️⃣ Verificar Configuración (1 min)

Edita [config.py](config.py) si necesitas cambiar:

```python
# Contraseña de PostgreSQL
DATABASE_CONFIG = {
    'password': 'gulpery',  # 👈 Cámbialo aquí
    ...
}

# URL de la cámara
CAMERA_CONFIG = {
    'source': 'http://192.168.100.11:8080/video',  # O usa 0 para webcam
    ...
}
```

### 4️⃣ Ejecutar Diagnóstico (Opcional - 1 min)

```powershell
python utils.py
```

Selecciona opción `1` para ejecutar diagnóstico completo.

### 5️⃣ ¡Iniciar el Sistema! (0 min)

```powershell
python app.py
```

---

## 🎮 Cómo Usar

### Controles del Teclado

| Tecla | Función |
|-------|---------|
| `ESC` | 🚪 Salir |
| `C` | 🗑️ Limpiar carrito |
| `V` | 🎤 Toggle voz |

### Comandos de Voz

1. **Di "LISTO"** para finalizar compra
2. El sistema detectará productos en el frame
3. Calculará el total automáticamente
4. Anunciará: *"He detectado [productos]. El total es $X.XX dólares"*
5. Guardará la venta en la base de datos

---

## 🎯 Flujo de Trabajo Típico

```
1. 📹 Abrir app.py → Se inicia la cámara
2. 🛍️ Colocar productos frente a la cámara
3. 👁️ Ver detecciones en tiempo real
4. 🛒 Revisar carrito en pantalla
5. 🎤 Decir "LISTO"
6. 💳 Sistema procesa y anuncia total
7. ✅ Venta guardada en BD
```

---

## 🐛 Soluciones Rápidas

### ❌ Error: "No module named 'psycopg2'"
```powershell
pip install psycopg2-binary
```

### ❌ Error: "Can't open camera"
- Verifica que la URL de la cámara sea correcta
- Para webcam local, cambia `source` a `0` en config.py

### ❌ Error: "Connection to database failed"
```powershell
# Verificar que PostgreSQL esté corriendo
# En Windows Services, busca "PostgreSQL"
# O en PowerShell:
Get-Service postgresql*
```

### ❌ Error en reconocimiento de voz
- Requiere conexión a Internet (usa Google API)
- Verifica permisos de micrófono en Windows

---

## 📁 Estructura de Archivos Principal

```
d:\cnn_proyect\
│
├── 🎯 app.py              ← INICIA AQUÍ
├── ⚙️ config.py           ← Configuración
├── 🔧 init_database.py    ← Ejecutar primero
├── 🛠️ utils.py            ← Diagnóstico
│
├── 🗄️ database.py         ← Módulo de BD
├── 👁️ detector.py         ← Detección YOLO
├── 🎤 voice_assistant.py  ← Asistente de voz
├── 🖥️ ui.py               ← Interfaz visual
│
├── 🎯 best.pt             ← Modelo entrenado
└── 📦 requirements.txt    ← Dependencias
```

---

## 📊 Verificar Ventas

### Consultar en PostgreSQL

```sql
-- Conectarse a la base de datos
psql -U postgres -d bazar_gulpery

-- Ver todas las ventas
SELECT * FROM historial_ventas ORDER BY fecha DESC LIMIT 10;

-- Total de ventas de hoy
SELECT SUM(total_pago) FROM historial_ventas 
WHERE DATE(fecha) = CURRENT_DATE;
```

### Usar script de utilidades

```powershell
python utils.py
# Seleccionar opción 6: "Mostrar reporte de ventas"
```

---

## 🎓 Próximos Pasos

1. ✅ Sistema funcionando
2. 📸 Probar con productos reales
3. 🎯 Ajustar umbrales de confianza si es necesario
4. 📊 Revisar métricas de ventas
5. 🔧 Personalizar según necesidades

---

## 🆘 ¿Necesitas Ayuda?

1. **Ejecuta diagnóstico:** `python utils.py` → Opción 1
2. **Revisa logs:** El sistema muestra mensajes detallados
3. **Verifica configuración:** Revisa [config.py](config.py)
4. **Consulta README:** [README.md](README.md) tiene información completa

---

## ✨ ¡Listo para Usar!

Tu sistema está configurado. Solo ejecuta:

```powershell
python app.py
```

**¡Disfruta tu Sistema POS con IA! 🎉**
