# Sistema de Punto de Venta con Visión por Computadora
## Documentación Técnica Completa

**Autor:** Arquitecto de Soluciones de IA  
**Institución:** Bazar Gulpery  
**Fecha:** Diciembre 2025  
**Versión:** 2.0

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Introducción](#2-introducción)
3. [Marco Teórico](#3-marco-teórico)
4. [Desarrollo del Sistema](#4-desarrollo-del-sistema)
5. [Resultados y Evaluación](#5-resultados-y-evaluación)
6. [Conclusiones](#6-conclusiones)
7. [Anexos](#7-anexos)

---

## 1. Resumen Ejecutivo

El presente documento describe el desarrollo e implementación de un **Sistema de Punto de Venta Automatizado** basado en técnicas de **Visión por Computadora** y **Aprendizaje Profundo** para el Bazar Gulpery. El sistema utiliza el modelo **YOLOv11** para la detección automática de productos en tiempo real, integrado con una base de datos **PostgreSQL** para la gestión de inventario y precios, y un asistente de voz interactivo para mejorar la experiencia del usuario.

### Objetivos Alcanzados

- ✅ Detección automática de 9 clases de productos con precisión superior al 90%
- ✅ Sistema de punto de venta completamente funcional con interfaz visual
- ✅ Integración con base de datos para gestión de inventario
- ✅ Asistente de voz para operación manos libres
- ✅ Procesamiento en tiempo real (>20 FPS)

### Tecnologías Clave

- **Framework de Deep Learning:** PyTorch 2.6+
- **Modelo de Detección:** YOLOv11
- **Procesamiento de Imágenes:** OpenCV 4.8
- **Base de Datos:** PostgreSQL 15
- **Lenguaje:** Python 3.11+

---

## 2. Introducción

### 2.1 Contexto del Problema

Los sistemas de punto de venta tradicionales requieren escaneo manual de códigos de barras o entrada manual de productos, lo que resulta en:

- ⏱️ Tiempos de espera prolongados
- ❌ Errores humanos en la entrada de datos
- 💰 Costos operativos elevados
- 📉 Experiencia de usuario subóptima

### 2.2 Propuesta de Solución

Implementar un sistema inteligente que:

1. **Detecta automáticamente** productos mediante cámara
2. **Identifica y clasifica** productos en tiempo real
3. **Calcula precios** automáticamente desde base de datos
4. **Permite confirmación** mediante comandos de voz
5. **Registra ventas** para análisis posterior

### 2.3 Alcance del Proyecto

**Incluye:**
- Detección de 9 clases de productos del inventario
- Sistema POS con interfaz gráfica
- Gestión de base de datos PostgreSQL
- Asistente de voz en español
- Sistema de logging y monitoreo

**No Incluye:**
- Integración con sistemas de pago
- Aplicación móvil
- Múltiples cámaras simultáneas

### 2.4 Justificación

La automatización mediante visión por computadora ofrece:

- **Velocidad:** Detección instantánea vs. escaneo manual
- **Precisión:** >90% de exactitud en detección
- **Escalabilidad:** Fácil adición de nuevos productos
- **Análisis:** Datos estructurados para business intelligence
- **Innovación:** Diferenciación competitiva

---

## 3. Marco Teórico

### 3.1 Visión por Computadora

La visión por computadora es un campo de la inteligencia artificial que permite a las computadoras "ver" y comprender imágenes digitales. En este proyecto, se utiliza para:

- **Detección de objetos:** Identificar productos en el frame
- **Clasificación:** Asignar categoría a cada producto
- **Localización:** Determinar posición mediante bounding boxes

### 3.2 Redes Neuronales Convolucionales (CNN)

Las CNN son arquitecturas de deep learning especializadas en procesamiento de imágenes. Características clave:

#### Capas Convolucionales
- **Función:** Extracción de características locales
- **Operación:** Convolución con kernels aprendibles
- **Output:** Feature maps de diferentes niveles

#### Pooling
- **Función:** Reducción de dimensionalidad
- **Tipos:** Max pooling, Average pooling
- **Beneficio:** Invarianza a pequeñas traslaciones

#### Capas Fully Connected
- **Función:** Clasificación final
- **Operación:** Combinación lineal + activación
- **Output:** Probabilidades por clase

### 3.3 YOLO (You Only Look Once)

YOLO es una familia de arquitecturas para detección de objetos en tiempo real.

#### Principio de Funcionamiento

1. **Imagen dividida en grid:** NxN celdas
2. **Predicción por celda:** 
   - Bounding boxes
   - Confianza de detección
   - Probabilidades de clase
3. **Non-Maximum Suppression:** Elimina detecciones duplicadas

#### YOLOv11 - Mejoras

- **Arquitectura mejorada:** C2f modules
- **Mayor precisión:** mAP superior a versiones anteriores
- **Eficiencia:** Menos parámetros, mismo rendimiento
- **Velocidad:** >100 FPS en GPU moderna

### 3.4 Transfer Learning

Técnica utilizada en el entrenamiento:

1. **Modelo Pre-entrenado:** COCO dataset (80 clases, 330K imágenes)
2. **Fine-tuning:** Ajuste con dataset específico de productos
3. **Ventajas:**
   - Menor cantidad de datos necesarios
   - Convergencia más rápida
   - Mejor generalización

### 3.5 Métricas de Evaluación

#### Precisión (Precision)
```
Precision = TP / (TP + FP)
```
Proporción de detecciones correctas entre todas las detecciones.

#### Recall (Sensibilidad)
```
Recall = TP / (TP + FN)
```
Proporción de objetos correctamente detectados.

#### mAP (mean Average Precision)
```
mAP = (1/N) × Σ AP(clase_i)
```
Promedio de Average Precision sobre todas las clases.

#### IoU (Intersection over Union)
```
IoU = Área_intersección / Área_unión
```
Métrica de solapamiento entre predicción y ground truth.

---

## 4. Desarrollo del Sistema Basado en Visión por Computadora

### 4.1 Descripción del Problema y Conjunto de Datos

#### 4.1.1 Contexto General del Problema

El Bazar Gulpery comercializa productos de papelería y accesorios que requieren identificación rápida y precisa en el punto de venta. Los productos presentan características visuales distintivas pero pueden ser confundidos entre categorías similares (e.g., diferentes tipos de flash USB).

**Desafíos específicos:**
- Variabilidad en iluminación del ambiente
- Diferentes orientaciones de productos
- Oclusiones parciales
- Productos visualmente similares
- Variación en distancia de cámara

#### 4.1.2 Objetivo del Sistema

Desarrollar un sistema de detección automática capaz de:

1. **Identificar** correctamente 9 clases de productos
2. **Localizar** productos en el espacio mediante bounding boxes
3. **Procesar** en tiempo real (>20 FPS)
4. **Mantener** precisión >90% en condiciones reales
5. **Integrarse** con sistema POS existente

#### 4.1.3 Alcance de la Solución Propuesta

**Componentes del Sistema:**

1. **Módulo de Detección**
   - Inferencia con YOLOv11
   - Filtrado de confianza
   - Estabilización temporal

2. **Módulo de Base de Datos**
   - Gestión de inventario
   - Consulta de precios
   - Registro de ventas

3. **Módulo de Interfaz**
   - Visualización en tiempo real
   - Carrito de compras
   - Estadísticas del sistema

4. **Módulo de Voz**
   - Reconocimiento de comandos
   - Síntesis de texto a voz
   - Confirmación de operaciones

#### 4.1.4 Herramientas Tecnológicas y Librerías Utilizadas

##### Librerías para Manipulación y Análisis de Datos

**NumPy 1.24+**
- Operaciones vectoriales eficientes
- Manipulación de arrays multidimensionales
- Operaciones matemáticas en imágenes

```python
import numpy as np

# Normalización de imágenes
imagen_normalizada = imagen.astype(np.float32) / 255.0

# Operaciones sobre bounding boxes
iou = calculate_iou(bbox1, bbox2)
```

**Pandas (opcional)**
- Análisis de métricas de entrenamiento
- Gestión de logs estructurados
- Generación de reportes

##### Frameworks de Aprendizaje Profundo

**PyTorch 2.6+**
- Backend de computación tensorial
- Soporte para GPU (CUDA)
- Autograd para backpropagation

```python
import torch

# Configuración de dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
```

**Ultralytics YOLOv11**
- Framework de alto nivel para YOLO
- API simplificada de entrenamiento
- Herramientas de evaluación integradas

```python
from ultralytics import YOLO

# Cargar modelo
model = YOLO('yolo11n.pt')

# Entrenar
results = model.train(
    data='dataset/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

##### Herramientas de Procesamiento de Imágenes

**OpenCV 4.8**
- Captura de video en tiempo real
- Transformaciones de imagen
- Dibujo de anotaciones
- Conversión de espacios de color

```python
import cv2

# Captura de video
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# Conversión RGB a BGR
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Dibujo de bounding box
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
```

**Pillow (PIL)**
- Carga y guardado de imágenes
- Transformaciones básicas
- Compatibilidad con formatos diversos

##### Librerías para Evaluación y Visualización

**Matplotlib**
- Generación de gráficos de métricas
- Visualización de curvas de entrenamiento
- Plots de distribución de datos

```python
import matplotlib.pyplot as plt

# Gráfico de pérdida
plt.plot(epochs, train_loss, label='Train')
plt.plot(epochs, val_loss, label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_loss.png')
```

**Seaborn**
- Visualizaciones estadísticas
- Matrices de confusión
- Gráficos de distribución

```python
import seaborn as sns

# Matriz de confusión
sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
```

##### Herramientas para el Desarrollo de la Interfaz Gráfica

**OpenCV (Highgui)**
- Ventanas de visualización
- Manejo de eventos de teclado
- Renderizado de interfaz

```python
# Crear ventana
cv2.namedWindow('POS System', cv2.WINDOW_NORMAL)

# Mostrar frame
cv2.imshow('POS System', annotated_frame)

# Esperar tecla
key = cv2.waitKey(1) & 0xFF
```

**pyttsx3**
- Síntesis de texto a voz
- Soporte multiidioma
- Control de velocidad y volumen

```python
import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('voice', 'spanish')
engine.say("Producto detectado")
engine.runAndWait()
```

**SpeechRecognition**
- Reconocimiento de voz
- Integración con Google Speech API
- Soporte para español

```python
import speech_recognition as sr

recognizer = sr.Recognizer()
with sr.Microphone() as source:
    audio = recognizer.listen(source)
    text = recognizer.recognize_google(audio, language='es-ES')
```

##### Base de Datos

**PostgreSQL 15**
- Sistema de gestión de base de datos relacional
- Soporte para transacciones ACID
- Rendimiento optimizado para consultas

**psycopg2**
- Adaptador PostgreSQL para Python
- Pool de conexiones
- Ejecución de queries parametrizadas

```python
import psycopg2
from psycopg2 import pool

# Pool de conexiones
db_pool = pool.SimpleConnectionPool(1, 10, **db_config)

# Query
conn = db_pool.getconn()
cursor = conn.cursor()
cursor.execute("SELECT precio FROM inventario WHERE nombre = %s", (producto,))
precio = cursor.fetchone()[0]
```

#### 4.1.5 Organización y Estructura del Conjunto de Datos

##### Estructura de Almacenamiento

El dataset sigue la estructura estándar de YOLO:

```
dataset/
├── data.yaml              # Configuración del dataset
├── train/                 # Conjunto de entrenamiento (80%)
│   ├── images/           # Imágenes JPG/PNG
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── labels/           # Anotaciones TXT
│       ├── img_001.txt
│       ├── img_002.txt
│       └── ...
├── valid/                # Conjunto de validación (15%)
│   ├── images/
│   └── labels/
└── test/                 # Conjunto de prueba (5%)
    ├── images/
    └── labels/
```

**Archivo data.yaml:**
```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 9  # Número de clases
names: ['Borrador de ballena', 'Borrador de sirena', 'Esfero Negro', 
        'Flash Kingston 4GB', 'Flash Verbatim 16Gb', 
        'Pasador Cabello Minimouse', 'Resaltador', 'cartera', 'perfume']
```

##### Definición de Clases

| ID | Clase | Descripción | Características Visuales |
|----|-------|-------------|--------------------------|
| 0 | Borrador de ballena | Borrador con forma de ballena | Azul, forma distintiva |
| 1 | Borrador de sirena | Borrador con forma de sirena | Colores variados, forma de sirena |
| 2 | Esfero Negro | Bolígrafo de tinta negra | Cilíndrico, negro |
| 3 | Flash Kingston 4GB | USB Kingston 4GB | Logo Kingston, negro/rojo |
| 4 | Flash Verbatim 16Gb | USB Verbatim 16GB | Logo Verbatim, gris/azul |
| 5 | Pasador Cabello Minimouse | Accesorio de cabello Minnie | Forma orejas, rojo/blanco |
| 6 | Resaltador | Marcador fluorescente | Colores brillantes |
| 7 | Cartera | Billetera/monedero | Rectangular, varios colores |
| 8 | Perfume | Frasco de perfume | Forma de botella |

##### Conjunto de Datos Originales y Preprocesados

**Datos Originales:**
- **Fuente:** Capturas con cámara IP del local
- **Cantidad:** ~1000 imágenes
- **Resolución:** Variable (1920x1080 a 640x480)
- **Formato:** JPEG, PNG

**Datos Preprocesados:**
- **Redimensionamiento:** 640x640 píxeles
- **Normalización:** [0, 1] rango
- **Formato:** JPEG optimizado
- **Anotaciones:** Formato YOLO (normalizado)

##### Formato de Anotaciones YOLO

Cada archivo `.txt` contiene una línea por objeto:

```
<class_id> <x_center> <y_center> <width> <height>
```

Donde:
- `class_id`: ID de la clase (0-8)
- `x_center, y_center`: Centro del bbox (normalizado 0-1)
- `width, height`: Dimensiones del bbox (normalizado 0-1)

**Ejemplo (img_001.txt):**
```
2 0.456 0.378 0.123 0.089  # Esfero Negro
6 0.678 0.512 0.098 0.145  # Resaltador
```

#### 4.1.6 Características Técnicas de las Imágenes

##### Formatos y Compatibilidad

**Formatos Soportados:**
- JPEG (.jpg, .jpeg) - Principal
- PNG (.png) - Con transparencia
- BMP (.bmp) - Sin compresión

**Compatibilidad:**
- OpenCV: Todos los formatos
- PyTorch: Via transforms.ToTensor()
- YOLO: JPEG/PNG recomendado

##### Resolución Estándar

**Input del Modelo:**
- **Dimensiones:** 640x640 píxeles
- **Aspect Ratio:** 1:1 (cuadrado)
- **Padding:** Letterbox para mantener proporción

**Pipeline de Redimensionamiento:**
```python
def letterbox_resize(image, target_size=640):
    """
    Redimensiona imagen manteniendo aspect ratio
    Agrega padding gris si es necesario
    """
    h, w = image.shape[:2]
    scale = min(target_size / h, target_size / w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Padding
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
```

##### Espacio de Color

**Entrada Original:**
- **Formato:** BGR (OpenCV default)
- **Canales:** 3 (Blue, Green, Red)
- **Rango:** 0-255 (uint8)

**Conversión para YOLO:**
```python
# OpenCV captura en BGR
frame_bgr = cv2.imread('image.jpg')

# YOLO espera RGB
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

# Normalización a [0, 1]
frame_normalized = frame_rgb.astype(np.float32) / 255.0
```

**Espacios de Color Alternativos (Aumentación):**
- **HSV:** Para aumentación de color
- **Grayscale:** Para análisis de forma
- **LAB:** Para normalización de iluminación

##### Normalización de Valores

**Normalización Min-Max:**
```python
# Pixel values: [0, 255] -> [0, 1]
normalized = image.astype(np.float32) / 255.0
```

**Normalización Z-Score (opcional):**
```python
# Mean and std from ImageNet (transfer learning)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

normalized = (image - mean) / std
```

**Impacto:**
- Acelera convergencia del entrenamiento
- Estabiliza gradientes
- Mejora generalización

#### 4.1.7 Metodología de Preprocesamiento

##### Normalización de Imágenes

**Paso 1: Carga de Imagen**
```python
import cv2
import numpy as np

def load_image(image_path):
    """Carga imagen desde disco"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"No se pudo cargar: {image_path}")
    return image
```

**Paso 2: Conversión de Espacio de Color**
```python
def convert_color_space(image):
    """BGR -> RGB"""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
```

**Paso 3: Normalización de Intensidad**
```python
def normalize_intensity(image):
    """Normaliza valores de píxel a [0, 1]"""
    return image.astype(np.float32) / 255.0
```

##### Conversión de Espacios de Color

**Para Aumentación de Dataset:**

```python
def augment_color_space(image):
    """
    Aumenta variabilidad mediante transformaciones de color
    """
    # Conversión a HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Variación de Hue (±15°)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-15, 15)) % 180
    
    # Variación de Saturation (±30%)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    
    # Variación de Value (±30%)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * np.random.uniform(0.7, 1.3), 0, 255)
    
    # Reconversión a RGB
    augmented = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return augmented
```

##### Segmentación de Regiones de Interés

**Detección de Área de Interés:**

```python
def extract_roi(image, bbox):
    """
    Extrae región de interés de la imagen
    
    Args:
        image: Imagen completa
        bbox: (x1, y1, x2, y2) coordenadas del bounding box
    
    Returns:
        Región de interés recortada
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Validar límites
    h, w = image.shape[:2]
    x1 = max(0, min(x1, w))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h))
    y2 = max(0, min(y2, h))
    
    # Extraer ROI
    roi = image[y1:y2, x1:x2]
    return roi
```

##### Redimensionamiento Uniforme

**Letterbox Resize (Mantiene Aspect Ratio):**

```python
def letterbox_resize(image, target_size=640, color=(114, 114, 114)):
    """
    Redimensiona imagen a tamaño target manteniendo aspect ratio
    Agrega padding para completar dimensiones
    
    Args:
        image: Imagen original
        target_size: Tamaño objetivo (cuadrado)
        color: Color del padding
    
    Returns:
        Imagen redimensionada con padding
    """
    h, w = image.shape[:2]
    
    # Calcular escala
    scale = min(target_size / h, target_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    
    # Redimensionar
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Crear canvas
    canvas = np.full((target_size, target_size, 3), color, dtype=np.uint8)
    
    # Calcular offset para centrar
    y_offset = (target_size - new_h) // 2
    x_offset = (target_size - new_w) // 2
    
    # Colocar imagen en canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas, scale, (x_offset, y_offset)
```

**Ajuste de Coordenadas de Bounding Boxes:**

```python
def adjust_bbox_coordinates(bbox, scale, offset):
    """
    Ajusta coordenadas de bbox después de letterbox resize
    
    Args:
        bbox: (x1, y1, x2, y2) coordenadas originales
        scale: Factor de escala aplicado
        offset: (x_offset, y_offset) padding agregado
    
    Returns:
        Coordenadas ajustadas
    """
    x1, y1, x2, y2 = bbox
    x_off, y_off = offset
    
    # Escalar y desplazar
    x1_new = int(x1 * scale + x_off)
    y1_new = int(y1 * scale + y_off)
    x2_new = int(x2 * scale + x_off)
    y2_new = int(y2 * scale + y_off)
    
    return (x1_new, y1_new, x2_new, y2_new)
```

#### 4.1.8 Estrategia de División y Balanceamiento de Datos

##### Conjunto de Entrenamiento

**Distribución:**
- **Porcentaje:** 80% del dataset total
- **Cantidad:** ~800 imágenes
- **Propósito:** Entrenamiento del modelo

**Características:**
- Mayor variabilidad de condiciones
- Incluye todas las clases balanceadas
- Anotaciones validadas manualmente

**División por Clase:**
```
Borrador de ballena:    120 imágenes
Borrador de sirena:     115 imágenes
Esfero Negro:           108 imágenes
Flash Kingston 4GB:      95 imágenes
Flash Verbatim 16Gb:    102 imágenes
Pasador Minimouse:       88 imágenes
Resaltador:             125 imágenes
Cartera:                110 imágenes
Perfume:                 98 imágenes
```

##### Conjunto de Validación

**Distribución:**
- **Porcentaje:** 15% del dataset total
- **Cantidad:** ~150 imágenes
- **Propósito:** Validación durante entrenamiento

**Uso:**
- Monitoreo de sobreajuste
- Selección de hiperparámetros
- Early stopping

**División por Clase:**
```
Borrador de ballena:     25 imágenes
Borrador de sirena:      24 imágenes
Esfero Negro:            22 imágenes
Flash Kingston 4GB:      20 imágenes
Flash Verbatim 16Gb:     21 imágenes
Pasador Minimouse:       18 imágenes
Resaltador:              26 imágenes
Cartera:                 23 imágenes
Perfume:                 20 imágenes
```

##### Conjunto de Prueba

**Distribución:**
- **Porcentaje:** 5% del dataset total
- **Cantidad:** ~50 imágenes
- **Propósito:** Evaluación final del modelo

**Características:**
- Imágenes nunca vistas por el modelo
- Condiciones realistas del ambiente
- Evaluación imparcial del rendimiento

**División por Clase:**
```
Borrador de ballena:     12 imágenes
Borrador de sirena:      11 imágenes
Esfero Negro:            10 imágenes
Flash Kingston 4GB:       9 imágenes
Flash Verbatim 16Gb:     10 imágenes
Pasador Minimouse:        8 imágenes
Resaltador:              13 imágenes
Cartera:                 11 imágenes
Perfume:                  9 imágenes
```

##### Técnicas de Aumento de Datos

**Transformaciones Geométricas:**

```python
# Rotación aleatoria
rotation_angle = random.uniform(-15, 15)
rotated = rotate_image(image, rotation_angle)

# Flip horizontal
if random.random() > 0.5:
    flipped = cv2.flip(image, 1)

# Traslación
tx, ty = random.randint(-50, 50), random.randint(-50, 50)
translated = translate_image(image, tx, ty)

# Escala
scale_factor = random.uniform(0.8, 1.2)
scaled = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
```

**Transformaciones de Color:**

```python
# Brillo
brightness = random.uniform(0.7, 1.3)
bright_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

# Contraste
contrast = random.uniform(0.8, 1.2)
contrasted = cv2.convertScaleAbs(image, alpha=contrast, beta=128*(1-contrast))

# Saturación (HSV)
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.7, 1.3)
saturated = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
```

**Transformaciones de Ruido:**

```python
# Ruido Gaussiano
noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
noisy_image = cv2.add(image, noise)

# Desenfoque
blurred = cv2.GaussianBlur(image, (5, 5), 0)
```

**Configuración de Aumentación en YOLO:**

```python
# training_config.yaml
augmentation:
  hsv_h: 0.015      # Hue augmentation
  hsv_s: 0.7        # Saturation augmentation
  hsv_v: 0.4        # Value augmentation
  degrees: 0.0      # Rotation (+/- deg)
  translate: 0.1    # Translation (+/- fraction)
  scale: 0.5        # Scale (+/- gain)
  shear: 0.0        # Shear (+/- deg)
  perspective: 0.0  # Perspective (+/- fraction)
  flipud: 0.0       # Flip up-down (probability)
  fliplr: 0.5       # Flip left-right (probability)
  mosaic: 1.0       # Mosaic augmentation (probability)
  mixup: 0.0        # Mixup augmentation (probability)
```

**Beneficios del Aumento de Datos:**
- ✅ Mayor tamaño efectivo del dataset
- ✅ Mejor generalización del modelo
- ✅ Reducción de sobreajuste
- ✅ Robustez a variaciones de iluminación
- ✅ Invarianza a transformaciones geométricas

---

*Continúa en [02_ARQUITECTURA_Y_ENTRENAMIENTO.md](./02_ARQUITECTURA_Y_ENTRENAMIENTO.md)*
