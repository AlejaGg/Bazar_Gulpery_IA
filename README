# 🛍️ Sistema de Punto de Venta con IA - Bazar Gulpery

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-orange.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema automatizado de punto de venta que utiliza **YOLOv11** para detección automática de productos, **PostgreSQL** para gestión de inventario, y un **Asistente de Voz** para interacción natural.

---

## 🌟 Características Principales

### 🎯 Detección Inteligente de Productos
- **Modelo:** YOLOv11 entrenado con 9 clases de productos
- **Precisión:** >92% mAP@0.5
- **Rendimiento:** >25 FPS en tiempo real
- **Visualización:** Bounding boxes con nombres y precios en vivo

### 💾 Gestión de Inventario
- **Base de Datos:** PostgreSQL con pool de conexiones
- **Tablas:** `inventario` y `historial_ventas`
- **Funciones:** CRUD completo, registro automático de transacciones

### 🎤 Asistente de Voz Interactivo
- **Reconocimiento:** Speech Recognition con Google API
- **Síntesis:** pyttsx3 para texto-a-voz en español
- **Comando:** Di "LISTO" para confirmar compra automáticamente

### 📊 Interfaz Visual Completa
- **Carrito dinámico:** Actualización en tiempo real
- **Precios automáticos:** Desde base de datos
- **Estadísticas:** FPS, conteo de productos, métricas del sistema

---

## 📦 Productos Detectables

El sistema puede identificar automáticamente los siguientes productos:

| ID | Producto | Descripción |
|----|----------|-------------|
| 0 | 🐋 Borrador de ballena | Borrador con forma de ballena |
| 1 | 🧜 Borrador de sirena | Borrador con forma de sirena |
| 2 | 🖊️ Esfero Negro | Bolígrafo de tinta negra |
| 3 | 💾 Flash Kingston 4GB | Memoria USB Kingston |
| 4 | 💾 Flash Verbatim 16Gb | Memoria USB Verbatim |
| 5 | 🎀 Pasador Cabello Minimouse | Accesorio de cabello |
| 6 | ✨ Resaltador | Marcador fluorescente |
| 7 | 👛 Cartera | Billetera/monedero |
| 8 | 🌸 Perfume | Frasco de perfume |

---

## 🏗️ Arquitectura del Proyecto

### Estructura de Directorios

```
cnn_proyect/
├── 📁 src/                          # Código fuente refactorizado
│   ├── config.py                    # Configuración centralizada
│   ├── 📁 core/                     # Módulos principales
│   │   ├── detector.py              # Detector YOLO optimizado
│   │   ├── database_manager.py     # Gestión de base de datos
│   │   └── pos_system.py           # Sistema POS principal
│   ├── 📁 utils/                    # Utilidades
│   │   ├── logger.py               # Sistema de logging
│   │   └── video_capture.py        # Captura de video optimizada
│   └── 📁 models/                   # Definiciones de modelos
│
├── 📁 documentacion/                # Documentación técnica completa
│   ├── 01_SISTEMA_COMPLETO.md      # Sistema y dataset
│   ├── 02_ARQUITECTURA_Y_ENTRENAMIENTO.md
│   ├── 03_RESULTADOS_Y_EVALUACION.md
│   └── 📁 imagenes/                 # Visualizaciones generadas
│       ├── 01_arquitectura_sistema.png
│       ├── 02_arquitectura_yolo.png
│       ├── 03_metricas_entrenamiento.png
│       ├── 04_matriz_confusion.png
│       ├── 05_distribucion_dataset.png
│       └── 06_rendimiento_por_clase.png
│
├── 📁 dataset/                      # Dataset de productos
│   ├── data.yaml                   # Configuración YOLO
│   ├── train/                      # 80% - Entrenamiento
│   ├── valid/                      # 15% - Validación
│   └── test/                       # 5% - Prueba
│
├── 📁 scripts/                      # Scripts auxiliares
│   ├── generate_documentation_images.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── best.pt                         # Modelo entrenado
├── requirements.txt                # Dependencias
└── README.md                       # Este archivo
```

### Componentes del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE PRESENTACIÓN                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │ UI Renderer  │  │Voice Assistant│  │  Menu System │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                 CAPA DE LÓGICA DE NEGOCIO                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  POS System  │  │Shopping Cart │  │Price Manager │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│               CAPA DE PROCESAMIENTO DE IA                    │
│  ┌──────────────┐  ┌──────────────────────────────────┐    │
│  │Product       │  │   Image Processing Pipeline      │    │
│  │Detector      │  │   (YOLOv11)                      │    │
│  └──────────────┘  └──────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            ↓↑
┌─────────────────────────────────────────────────────────────┐
│                      CAPA DE DATOS                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PostgreSQL  │  │Video Capture │  │Model Weights │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Inicio Rápido

### Prerrequisitos

- **Python:** 3.11 o superior
- **PostgreSQL:** 15 o superior
- **Cámara:** Webcam o cámara IP
- **GPU (Opcional):** Para mayor rendimiento

### Instalación

1. **Clonar el repositorio:**
```bash
git clone https://github.com/tu-usuario/cnn_proyect.git
cd cnn_proyect
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar base de datos:**
```bash
# Iniciar PostgreSQL
# Crear base de datos
createdb bazar_gulpery

# Inicializar tablas
python init_database.py
```

5. **Configurar variables de entorno:**
```bash
# Crear archivo .env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=tu_password
DB_NAME=bazar_gulpery
CAMERA_SOURCE=0  # 0 para webcam o URL de cámara IP
USE_GPU=true
```

### Entrenamiento del Modelo (Opcional)

Si necesitas reentrenar el modelo:

```bash
python scripts/train_model.py
```

**Configuración de entrenamiento:**
- **Épocas:** 100
- **Batch Size:** 16
- **Tamaño de imagen:** 640x640
- **Optimizador:** AdamW
- **Learning Rate:** 0.01

### Ejecución del Sistema

```bash
python app.py
```

**Controles:**
- **Q:** Salir
- **LISTO (voz):** Finalizar compra
- **ESC:** Cancelar operación

---

## 📊 Rendimiento del Sistema

### Métricas del Modelo

| Métrica | Valor | Descripción |
|---------|-------|-------------|
| **mAP@0.5** | 92.3% | Precisión de detección a IoU 0.5 |
| **mAP@0.5:0.95** | 84.7% | Precisión promedio a diferentes IoU |
| **Precisión** | 91.6% | Detecciones correctas |
| **Recall** | 89.1% | Objetos detectados |
| **F1-Score** | 90.3% | Balance P-R |
| **FPS** | >25 | Frames por segundo |
| **Latencia** | <40ms | Tiempo de inferencia |

### Rendimiento por Clase

![Rendimiento por Clase](documentacion/imagenes/06_rendimiento_por_clase.png)

| Clase | Precisión | Recall | F1-Score |
|-------|-----------|--------|----------|
| Resaltador | 93% | 91% | 92.0% |
| Borrador ballena | 94% | 91% | 92.5% |
| Flash Verbatim | 91% | 88% | 89.5% |
| Cartera | 90% | 87% | 88.5% |
| Perfume | 88% | 86% | 87.0% |

---

## 🔧 Configuración Avanzada

### Ajustar Umbrales de Detección

En `src/config.py`:

```python
MODEL_CONFIG = {
    'confidence_threshold': 0.5,  # Umbral de confianza
    'iou_threshold': 0.45,         # Umbral IoU para NMS
}
```

**Guía de umbrales:**
- **0.3:** Mayor recall, más falsos positivos
- **0.5:** Balance óptimo (recomendado)
- **0.7:** Mayor precisión, menos falsos positivos

### Configurar Aumentación de Datos

```python
TRAINING_CONFIG = {
    'augmentation': {
        'hsv_h': 0.015,    # Variación de Hue
        'hsv_s': 0.7,      # Variación de Saturación
        'hsv_v': 0.4,      # Variación de Valor
        'fliplr': 0.5,     # Flip horizontal
        'mosaic': 1.0,     # Mosaic augmentation
    }
}
```

### Optimización para Producción

**Exportar a ONNX:**
```bash
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"
```

**Cuantización INT8:**
```bash
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx', int8=True)"
```

---

## 📖 Documentación Completa

La documentación técnica detallada está disponible en la carpeta `documentacion/`:

1. **[01_SISTEMA_COMPLETO.md](documentacion/01_SISTEMA_COMPLETO.md)**
   - Descripción del problema y conjunto de datos
   - Herramientas tecnológicas utilizadas
   - Organización del dataset
   - Metodología de preprocesamiento

2. **[02_ARQUITECTURA_Y_ENTRENAMIENTO.md](documentacion/02_ARQUITECTURA_Y_ENTRENAMIENTO.md)**
   - Diseño de arquitectura del modelo
   - Configuración de entrenamiento
   - Proceso de validación
   - Técnicas de regularización

3. **[03_RESULTADOS_Y_EVALUACION.md](documentacion/03_RESULTADOS_Y_EVALUACION.md)**
   - Métricas de rendimiento
   - Análisis de errores
   - Evaluación de robustez
   - Conclusiones y trabajo futuro

---

## 🔬 Generar Visualizaciones

Para regenerar las visualizaciones de la documentación:

```bash
python scripts/generate_documentation_images.py
```

Esto generará:
- ✅ Diagrama de arquitectura del sistema
- ✅ Arquitectura de YOLOv11
- ✅ Métricas de entrenamiento
- ✅ Matriz de confusión
- ✅ Distribución del dataset
- ✅ Rendimiento por clase

---

## 🐛 Solución de Problemas

### Error: No se puede conectar a la base de datos

```bash
# Verificar que PostgreSQL está corriendo
pg_isready

# Verificar credenciales en .env
cat .env
```

### Error: No se encuentra la cámara

```bash
# Listar dispositivos de video
python -c "import cv2; print([cv2.VideoCapture(i).isOpened() for i in range(5)])"

# Cambiar source en config.py
CAMERA_CONFIG['source'] = 0  # Para webcam
```

### Error: Modelo no encontrado

```bash
# Descargar modelo pre-entrenado
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11n.pt

# O entrenar desde cero
python scripts/train_model.py
```

### Bajo rendimiento (FPS)

```bash
# Usar GPU si está disponible
USE_GPU=true python app.py

# Reducir resolución de entrada
MODEL_CONFIG['img_size'] = 416  # En lugar de 640

# Exportar a ONNX
python -c "from ultralytics import YOLO; YOLO('best.pt').export(format='onnx')"
```

---

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## 👥 Autores

- **Arquitecto de Soluciones de IA** - Desarrollo principal
- **Bazar Gulpery** - Casos de uso y testing

---

## 🙏 Agradecimientos

- [Ultralytics](https://ultralytics.com/) por YOLOv11
- [PyTorch](https://pytorch.org/) por el framework de deep learning
- [OpenCV](https://opencv.org/) por herramientas de visión por computadora
- [Roboflow](https://roboflow.com/) por herramientas de anotación

---

## 📧 Contacto

Para preguntas o sugerencias:
- **Email:** contacto@bazargulpery.com
- **Website:** https://bazargulpery.com
- **GitHub Issues:** https://github.com/tu-usuario/cnn_proyect/issues

---

## 📈 Roadmap

- [x] Sistema básico de detección
- [x] Integración con base de datos
- [x] Asistente de voz
- [x] Documentación completa
- [ ] Dashboard web
- [ ] App móvil
- [ ] Multi-cámara
- [ ] Análisis predictivo
- [ ] Integración con ERP

---

## 🔗 Enlaces Útiles

- [Documentación de YOLOv11](https://docs.ultralytics.com/)
- [Tutorial de PyTorch](https://pytorch.org/tutorials/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

---

**⭐ Si este proyecto te resulta útil, considera darle una estrella en GitHub!**

---

*Última actualización: Diciembre 30, 2025*
