# 🚀 Guía de Inicio - Sistema POS con IA

## ✅ Requisitos Previos

Antes de ejecutar la aplicación, asegúrate de tener:

- ✔️ **PostgreSQL** corriendo (puerto 5432)
- ✔️ **Python 3.8+** instalado
- ✔️ **IP Webcam** instalado en tu móvil (opcional si usas webcam)
- ✔️ Móvil y PC en la **misma red Wi-Fi**

---

## 📋 Pasos para Ejecutar

### 1️⃣ Abrir Terminal en el Proyecto

```powershell
cd d:\cnn_proyect
```

### 2️⃣ Activar Entorno Virtual

```powershell
.\venv\Scripts\activate
```

Deberías ver `(venv)` al inicio de tu línea de comandos.

### 3️⃣ Verificar Configuración (Primera vez)

Abre `config.py` y verifica:

**Contraseña de PostgreSQL:**
```python
DATABASE_CONFIG = {
    'password': 'gulpery2025',  # 👈 Debe coincidir con tu PostgreSQL
    ...
}
```

**URL de la Cámara:**
```python
CAMERA_CONFIG = {
    'source': 'http://192.168.100.11:8080/video',  # 👈 IP de tu móvil
    ...
}
```

### 4️⃣ Inicializar Base de Datos (Solo primera vez)

```powershell
python init_database.py
```

Esto crea la base de datos `bazar_gulpery` con los productos y precios.

### 5️⃣ Preparar IP Webcam

**En tu móvil:**

1. Abre la app **IP Webcam**
2. Ve hasta el final y toca **"Iniciar servidor"**
3. Anota la URL que aparece (ejemplo: `http://192.168.100.11:8080/video`)
4. Si es diferente a la del `config.py`, actualízala

### 6️⃣ Ejecutar la Aplicación

```powershell
python app.py
```

Se abrirá el **Menú Principal** con dos opciones:

- **Presiona `A`** → Panel de Administración (modificar precios)
- **Presiona `I`** → Iniciar Sistema POS

---

## 🎮 Uso del Sistema

### Controles del Sistema POS

| Tecla | Acción |
|-------|--------|
| **ESC** | Salir del sistema |
| **C** | Limpiar carrito |
| **V** | Activar/desactivar voz |

### Flujo de Trabajo

1. **Coloca productos** frente a la cámara
2. El sistema **detecta automáticamente** y los agrega al carrito
3. **Di "LISTO"** para finalizar la compra
4. El sistema **calcula el total** y lo anuncia por voz
5. La venta se **guarda automáticamente** en la base de datos

---

## 🛠️ Solución de Problemas

### ❌ Error: "No module found"
```powershell
pip install -r requirements.txt
```

### ❌ Error: "No se puede conectar a PostgreSQL"
- Verifica que PostgreSQL esté corriendo
- Comprueba la contraseña en `config.py`
- Ejecuta `python init_database.py`

### ⚠️ "Esperando conexión con cámara"
- Verifica que IP Webcam esté activa
- Comprueba que estés en la misma red Wi-Fi
- Actualiza la IP en `config.py` si cambió

### 🔇 La voz no funciona
- Verifica que tengas micrófono conectado
- Presiona `V` para activar/desactivar voz
- Requiere conexión a internet para reconocimiento de voz

---

## 📱 Configuración Alternativa - Webcam USB

Si no tienes IP Webcam, puedes usar una webcam USB:

En `config.py`, cambia:
```python
CAMERA_CONFIG = {
    'source': 0,  # 0 = webcam predeterminada, 1 = segunda cámara
    ...
}
```

---

## 🎯 Panel de Administración

Para modificar precios:

1. En el menú principal, presiona **`A`**
2. Selecciona un producto de la tabla
3. Haz clic en **"Modificar Precio"**
4. Ingresa el nuevo precio
5. Haz clic en **"Guardar"**

---

## 📊 Entrenar Nuevo Modelo (Opcional)

Si quieres mejorar la detección:

```powershell
python train_model.py
```

El nuevo modelo se guardará en `runs/detect/train/weights/best.pt`

Cópialo a la raíz del proyecto para reemplazar el modelo actual.

---

## 🚪 Salir del Sistema

- Presiona **ESC** en cualquier pantalla
- O cierra la ventana directamente

---

## 📞 Resumen Rápido

```powershell
# 1. Activar entorno
.\venv\Scripts\activate

# 2. Iniciar IP Webcam en móvil

# 3. Ejecutar
python app.py

# 4. Presionar 'I' para iniciar

# 5. ¡Listo! Empieza a detectar productos
```

---

**¡Disfruta del Sistema POS con IA! 🎉**
