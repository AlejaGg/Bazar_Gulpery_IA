# Desarrollo del Sistema - Parte 2
## Arquitectura del Modelo y Entrenamiento

---

## 4.2 Diseño de la Arquitectura del Modelo

### 4.2.1 Estructura General del Modelo

#### Definición de la Entrada

**Especificaciones de Entrada:**
- **Dimensiones:** 640 × 640 × 3
- **Formato:** Tensor RGB normalizado
- **Rango:** [0, 1] (float32)
- **Batch Size:** 16 imágenes

```python
# Preparación de entrada
input_tensor = torch.from_numpy(image).float()
input_tensor = input_tensor.permute(2, 0, 1)  # HWC -> CHW
input_tensor = input_tensor.unsqueeze(0)       # Batch dimension
input_tensor = input_tensor / 255.0            # Normalización
input_tensor = input_tensor.to(device)         # GPU
```

**Pipeline de Entrada:**
```
Imagen Raw (BGR) 
    ↓
Conversión RGB
    ↓
Letterbox Resize (640x640)
    ↓
Normalización [0,1]
    ↓
Conversión a Tensor
    ↓
Modelo YOLO
```

#### Capas de Extracción de Características

**Backbone: CSPDarknet**

YOLOv11 utiliza una arquitectura backbone mejorada basada en CSPDarknet con módulos C2f:

**Capa 1: Stem**
```
Input: 640×640×3
    ↓
Conv2d(3, 32, kernel=6, stride=2, padding=2)  # Focus
    ↓
BatchNorm2d + SiLU
    ↓
Output: 320×320×32
```

**Capa 2: Stage 1**
```
Input: 320×320×32
    ↓
Conv2d(32, 64, kernel=3, stride=2, padding=1)
    ↓
C2f(64, 64, n=1)
    ↓
Output: 160×160×64
```

**Capa 3: Stage 2**
```
Input: 160×160×64
    ↓
Conv2d(64, 128, kernel=3, stride=2, padding=1)
    ↓
C2f(128, 128, n=2)
    ↓
Output: 80×80×128
```

**Capa 4: Stage 3**
```
Input: 80×80×128
    ↓
Conv2d(128, 256, kernel=3, stride=2, padding=1)
    ↓
C2f(256, 256, n=2)
    ↓
Output: 40×40×256
```

**Capa 5: Stage 4**
```
Input: 40×40×256
    ↓
Conv2d(256, 512, kernel=3, stride=2, padding=1)
    ↓
C2f(512, 512, n=1)
    ↓
Output: 20×20×512
```

**Módulo C2f (CSP Bottleneck with 2 Convolutions):**

```python
class C2f(nn.Module):
    """CSP Bottleneck con 2 convoluciones"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, e=1.0) 
            for _ in range(n)
        )
    
    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
```

#### Capas de Regularización

**Batch Normalization:**
- Normaliza activaciones por batch
- Reduce Internal Covariate Shift
- Permite learning rates más altos

```python
class BatchNorm2d(nn.Module):
    """
    Normalización por batch
    """
    def forward(self, x):
        # x: (N, C, H, W)
        mean = x.mean([0, 2, 3])
        var = x.var([0, 2, 3])
        x_norm = (x - mean) / sqrt(var + eps)
        return gamma * x_norm + beta
```

**Dropout (No usado en YOLO):**
- YOLO usa otras técnicas de regularización
- Mosaic augmentation
- DropBlock en algunas variantes

**SiLU Activation (Swish):**
```python
def silu(x):
    """Sigmoid Linear Unit"""
    return x * torch.sigmoid(x)
```

#### Capas de Clasificación

**Detection Head:**

YOLOv11 utiliza un head de detección desacoplado:

```python
class DetectionHead(nn.Module):
    def __init__(self, nc=9):  # nc: número de clases
        super().__init__()
        self.nc = nc
        
        # Rama de clasificación
        self.cls_conv = nn.Sequential(
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            nn.Conv2d(256, nc, 1)
        )
        
        # Rama de regresión (bbox)
        self.reg_conv = nn.Sequential(
            Conv(256, 256, 3, 1),
            Conv(256, 256, 3, 1),
            nn.Conv2d(256, 4, 1)
        )
        
        # Rama de objectness
        self.obj_conv = nn.Sequential(
            Conv(256, 256, 3, 1),
            nn.Conv2d(256, 1, 1)
        )
    
    def forward(self, x):
        cls = self.cls_conv(x)  # (N, nc, H, W)
        reg = self.reg_conv(x)  # (N, 4, H, W)
        obj = self.obj_conv(x)  # (N, 1, H, W)
        
        return cls, reg, obj
```

### 4.2.2 Configuración de la Arquitectura

#### Número de Capas y Filtros

**Resumen de Arquitectura:**

| Stage | Tipo | Input Size | Output Size | Filtros | Parámetros |
|-------|------|------------|-------------|---------|------------|
| Input | - | 640×640×3 | 640×640×3 | - | 0 |
| Stem | Conv+BN+SiLU | 640×640×3 | 320×320×32 | 32 | 1.7K |
| Stage1 | Conv+C2f | 320×320×32 | 160×160×64 | 64 | 21K |
| Stage2 | Conv+C2f | 160×160×64 | 80×80×128 | 128 | 86K |
| Stage3 | Conv+C2f | 80×80×128 | 40×40×256 | 256 | 344K |
| Stage4 | Conv+C2f | 40×40×256 | 20×20×512 | 512 | 1.3M |
| SPPF | Pooling | 20×20×512 | 20×20×512 | 512 | 656K |
| Neck | FPN+PAN | Multi-scale | Multi-scale | Variable | 1.8M |
| Head | Detection | Multi-scale | Detections | 9 classes | 520K |
| **Total** | | | | | **~4.8M** |

**Distribución de Parámetros:**
- Backbone: 60%
- Neck: 30%
- Head: 10%

#### Funciones de Activación

**SiLU (Swish) - Principal:**
```python
f(x) = x * σ(x)
```

**Características:**
- Suave y diferenciable
- No acotada superiormente
- Auto-gating mechanism
- Mejor que ReLU para deep networks

**Comparación de Activaciones:**

| Activación | Ecuación | Rango | Ventajas |
|------------|----------|-------|----------|
| ReLU | max(0, x) | [0, ∞) | Simple, rápida |
| Leaky ReLU | max(0.01x, x) | (-∞, ∞) | Evita dying ReLU |
| SiLU | x·σ(x) | (-∞, ∞) | Suave, mejor gradiente |
| Mish | x·tanh(ln(1+eˣ)) | (-∞, ∞) | Más suave que SiLU |

**Por qué SiLU en YOLO:**
- Gradientes más suaves
- Mejor flujo de información
- Convergencia más rápida
- Menor riesgo de dying neurons

#### Técnicas de Regularización

**1. Data Augmentation:**
```python
augmentation_config = {
    'mosaic': 1.0,           # Combina 4 imágenes
    'mixup': 0.0,            # Mezcla de imágenes
    'copy_paste': 0.0,       # Copia objetos entre imágenes
    'hsv_h': 0.015,          # Variación Hue
    'hsv_s': 0.7,            # Variación Saturation
    'hsv_v': 0.4,            # Variación Value
    'degrees': 0.0,          # Rotación
    'translate': 0.1,        # Traslación
    'scale': 0.5,            # Escala
    'fliplr': 0.5,           # Flip horizontal
}
```

**2. Batch Normalization:**
- Normaliza por mini-batch
- Reduce covariate shift
- Actúa como regularizador

**3. Weight Decay (L2 Regularization):**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005  # L2 regularization
)
```

**4. Gradient Clipping:**
```python
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm=10.0
)
```

**5. Early Stopping:**
```python
patience = 50  # Épocas sin mejora antes de detener
best_fitness = 0.0
epochs_no_improve = 0

for epoch in epochs:
    val_fitness = validate(model)
    
    if val_fitness > best_fitness:
        best_fitness = val_fitness
        epochs_no_improve = 0
        save_checkpoint(model)
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break
```

### 4.2.3 Configuración de Entrenamiento y Optimización

#### Optimizador

**AdamW (Adam con Weight Decay):**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.01,                    # Learning rate inicial
    betas=(0.937, 0.999),       # Coeficientes de momentum
    weight_decay=0.0005,        # L2 regularization
    eps=1e-8                    # Estabilidad numérica
)
```

**Ecuaciones de Actualización:**

```
m_t = β₁ · m_{t-1} + (1 - β₁) · ∇θ
v_t = β₂ · v_{t-1} + (1 - β₂) · (∇θ)²
m̂_t = m_t / (1 - β₁ᵗ)
v̂_t = v_t / (1 - β₂ᵗ)
θ_t = θ_{t-1} - η · (m̂_t / (√v̂_t + ε) + λ · θ_{t-1})
```

**Ventajas de AdamW:**
- Adaptive learning rates por parámetro
- Momentum para acelerar convergencia
- Weight decay desacoplado (mejor regularización)
- Convergencia rápida y estable

#### Función de Pérdida

**Pérdida Compuesta de YOLO:**

```python
total_loss = λ₁·L_cls + λ₂·L_box + λ₃·L_obj
```

**1. Classification Loss (Cross-Entropy):**

```python
def classification_loss(pred_cls, target_cls, num_classes):
    """
    Binary Cross-Entropy para cada clase
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    
    # One-hot encoding del target
    target_onehot = F.one_hot(target_cls, num_classes)
    
    # Calcular pérdida
    loss = bce_loss(pred_cls, target_onehot.float())
    return loss
```

**2. Bounding Box Loss (CIoU):**

```python
def ciou_loss(pred_box, target_box):
    """
    Complete IoU Loss
    Considera: overlap, distancia de centros, aspect ratio
    """
    # IoU
    iou = calculate_iou(pred_box, target_box)
    
    # Distancia de centros
    c_x = (pred_box.center - target_box.center).pow(2).sum()
    
    # Diagonal del rectángulo envolvente
    c = calculate_diagonal(pred_box, target_box)
    
    # Penalty por distancia
    distance_penalty = c_x / c
    
    # Penalty por aspect ratio
    v = (4 / π²) * (
        torch.atan(target_box.w / target_box.h) - 
        torch.atan(pred_box.w / pred_box.h)
    ).pow(2)
    alpha = v / (1 - iou + v)
    aspect_penalty = alpha * v
    
    # CIoU Loss
    loss = 1 - iou + distance_penalty + aspect_penalty
    return loss.mean()
```

**3. Objectness Loss (Binary Cross-Entropy):**

```python
def objectness_loss(pred_obj, target_obj):
    """
    Pérdida de confianza de objeto
    """
    bce_loss = nn.BCEWithLogitsLoss(reduction='mean')
    return bce_loss(pred_obj, target_obj)
```

**Pesos de Pérdida:**
```python
loss_weights = {
    'cls': 0.5,   # Classification
    'box': 0.05,  # Bounding box
    'obj': 1.0,   # Objectness
}

total_loss = (
    loss_weights['cls'] * cls_loss +
    loss_weights['box'] * box_loss +
    loss_weights['obj'] * obj_loss
)
```

#### Parámetros de Entrenamiento

**Configuración Completa:**

```python
training_config = {
    # Optimización
    'optimizer': 'AdamW',
    'learning_rate': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    
    # Learning Rate Schedule
    'scheduler': 'cosine',
    'warmup_epochs': 3,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'final_lr_ratio': 0.01,
    
    # Entrenamiento
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    
    # Regularización
    'augmentation': True,
    'mosaic': 1.0,
    'mixup': 0.0,
    
    # Early Stopping
    'patience': 50,
    
    # Hardware
    'device': 'cuda',
    'workers': 8,
    'amp': True,  # Automatic Mixed Precision
}
```

**Learning Rate Scheduler (Cosine Annealing):**

```python
def cosine_lr_schedule(epoch, total_epochs, lr_init, lr_final):
    """
    Cosine annealing learning rate
    """
    return lr_final + (lr_init - lr_final) * (
        1 + np.cos(np.pi * epoch / total_epochs)
    ) / 2
```

**Gráfico de Learning Rate:**
```
lr
│     ╱‾‾‾╲
│    ╱     ╲
│   ╱       ╲
│  ╱         ╲___
│ ╱               ╲___
└─────────────────────→ epoch
  warmup  cosine decay
```

**Warmup Schedule:**

```python
def warmup_schedule(epoch, warmup_epochs, lr_init):
    """
    Linear warmup para primeras épocas
    """
    if epoch < warmup_epochs:
        return lr_init * (epoch / warmup_epochs)
    return lr_init
```

---

## 4.3 Proceso de Entrenamiento y Validación

### 4.3.1 Configuración del Proceso de Entrenamiento

#### Parámetros del Entrenamiento

**Script de Entrenamiento:**

```python
from ultralytics import YOLO

# Cargar modelo base pre-entrenado
model = YOLO('yolo11n.pt')

# Configuración de entrenamiento
results = model.train(
    # Dataset
    data='dataset/data.yaml',
    
    # Hiperparámetros
    epochs=100,
    batch=16,
    imgsz=640,
    
    # Optimización
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    
    # Augmentation
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.0,
    
    # Regularización
    patience=50,
    
    # Hardware
    device='cuda:0',
    workers=8,
    project='runs/detect',
    name='bazar_gulpery',
    exist_ok=False,
    
    # Logging
    verbose=True,
    save=True,
    save_period=10,
    plots=True,
    
    # Advanced
    amp=True,  # Automatic Mixed Precision
    cache=False,
    rect=False,
)
```

**Estructura de Directorios de Salida:**

```
runs/detect/bazar_gulpery/
├── weights/
│   ├── best.pt              # Mejor modelo (val mAP)
│   ├── last.pt              # Último checkpoint
│   └── epoch{N}.pt          # Checkpoints periódicos
├── results.csv              # Métricas por época
├── confusion_matrix.png     # Matriz de confusión
├── F1_curve.png            # Curva F1-Score
├── P_curve.png             # Curva Precision
├── R_curve.png             # Curva Recall
├── PR_curve.png            # Curva Precision-Recall
└── train_batch*.jpg        # Batches de entrenamiento
```

#### Métricas de Evaluación

**Métricas Principales:**

1. **Precision (P):**
```
P = TP / (TP + FP)
```
Proporción de detecciones correctas.

2. **Recall (R):**
```
R = TP / (TP + FN)
```
Proporción de objetos detectados.

3. **F1-Score:**
```
F1 = 2 × (P × R) / (P + R)
```
Media armónica de Precision y Recall.

4. **mAP@0.5 (mean Average Precision):**
```
mAP@0.5 = (1/N) × Σ AP_i(IoU=0.5)
```
Promedio de AP para IoU ≥ 0.5.

5. **mAP@0.5:0.95:**
```
mAP@0.5:0.95 = (1/10) × Σ mAP(IoU=0.5+i×0.05)
```
Promedio de mAP para IoU desde 0.5 hasta 0.95.

**Registro de Métricas:**

```python
class MetricsLogger:
    def __init__(self):
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'map50': [],
            'map50_95': [],
            'fitness': [],
        }
    
    def log(self, epoch, metrics_dict):
        """Registra métricas de una época"""
        self.metrics['epoch'].append(epoch)
        for key, value in metrics_dict.items():
            if key in self.metrics:
                self.metrics[key].append(value)
    
    def save(self, filepath):
        """Guarda métricas en CSV"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(filepath, index=False)
    
    def plot(self, output_dir):
        """Genera gráficos de métricas"""
        # Plot de pérdida
        plt.figure(figsize=(10, 6))
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], label='Train')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], label='Val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(f'{output_dir}/loss_curve.png')
        plt.close()
```

### 4.3.2 Estrategia de Validación de Datos

#### División Estratificada

**Estratificación por Clase:**

```python
def stratified_split(dataset, train_ratio=0.8, val_ratio=0.15, test_ratio=0.05):
    """
    Divide dataset manteniendo proporción de clases
    """
    from sklearn.model_selection import train_test_split
    
    # Extraer clases
    images, labels = zip(*dataset)
    classes = [extract_main_class(label) for label in labels]
    
    # Primera división: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, classes,
        train_size=train_ratio,
        stratify=classes,
        random_state=42
    )
    
    # Segunda división: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        train_size=val_size,
        stratify=y_temp,
        random_state=42
    )
    
    return {
        'train': (X_train, y_train),
        'val': (X_val, y_val),
        'test': (X_test, y_test)
    }
```

**Verificación de Balance:**

```python
def verify_class_balance(dataset_split):
    """
    Verifica que todas las clases estén balanceadas
    """
    for split_name, (images, labels) in dataset_split.items():
        class_counts = Counter(labels)
        print(f"\n{split_name.upper()} Set:")
        print(f"Total images: {len(images)}")
        print("\nClass distribution:")
        for class_name, count in sorted(class_counts.items()):
            percentage = (count / len(images)) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")
```

#### Uso del Conjunto de Validación

**Validación durante Entrenamiento:**

```python
def train_with_validation(model, train_loader, val_loader, epochs):
    """
    Entrena con validación periódica
    """
    best_val_map = 0.0
    
    for epoch in range(epochs):
        # Entrenamiento
        model.train()
        train_loss = train_one_epoch(model, train_loader)
        
        # Validación
        model.eval()
        with torch.no_grad():
            val_metrics = validate(model, val_loader)
        
        # Logging
        print(f"Epoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val mAP@0.5: {val_metrics['map50']:.4f}")
        print(f"  Val mAP@0.5:0.95: {val_metrics['map50_95']:.4f}")
        
        # Guardar mejor modelo
        if val_metrics['map50'] > best_val_map:
            best_val_map = val_metrics['map50']
            torch.save(model.state_dict(), 'best.pt')
            print(f"  ✅ New best model saved!")
```

**Evaluación en Conjunto de Validación:**

```python
def validate(model, val_loader):
    """
    Evalúa modelo en conjunto de validación
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            predictions = model(images)
            
            all_predictions.extend(predictions)
            all_targets.extend(targets)
    
    # Calcular métricas
    metrics = calculate_metrics(all_predictions, all_targets)
    
    return metrics
```

### 4.3.3 Técnicas de Regularización y Aumento de Datos

#### Regularización del Modelo

**Weight Decay (L2):**

```python
# En el optimizador
optimizer = AdamW(
    model.parameters(),
    lr=0.01,
    weight_decay=0.0005  # Penaliza pesos grandes
)

# Ecuación de actualización
θ_new = θ_old - lr * (gradient + weight_decay * θ_old)
```

**Dropout (si se usara):**

```python
class DetectorWithDropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.backbone = Backbone()
        self.dropout = nn.Dropout(dropout_rate)
        self.head = DetectionHead()
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)  # Durante entrenamiento
        output = self.head(features)
        return output
```

**Batch Normalization:**

```python
class ConvBnAct(nn.Module):
    """Convolución + BatchNorm + Activación"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)  # Normaliza y regula
        x = self.act(x)
        return x
```

#### Aumento Artificial del Conjunto de Datos

**Mosaic Augmentation:**

```python
def mosaic_augmentation(images, labels):
    """
    Combina 4 imágenes en una sola
    Mejora detección de objetos pequeños
    """
    # Seleccionar 4 imágenes
    indices = random.sample(range(len(images)), 4)
    selected_images = [images[i] for i in indices]
    selected_labels = [labels[i] for i in indices]
    
    # Crear canvas 2x2
    h, w = 640, 640
    mosaic = np.full((h, w, 3), 114, dtype=np.uint8)
    
    # Dividir en 4 cuadrantes
    splits = [
        (0, h//2, 0, w//2),      # Top-left
        (0, h//2, w//2, w),      # Top-right
        (h//2, h, 0, w//2),      # Bottom-left
        (h//2, h, w//2, w),      # Bottom-right
    ]
    
    new_labels = []
    for img, lbl, (y1, y2, x1, x2) in zip(selected_images, selected_labels, splits):
        # Redimensionar y colocar
        resized = cv2.resize(img, (x2-x1, y2-y1))
        mosaic[y1:y2, x1:x2] = resized
        
        # Ajustar labels
        adjusted_labels = adjust_bbox(lbl, (x1, y1), (x2-x1, y2-y1))
        new_labels.extend(adjusted_labels)
    
    return mosaic, new_labels
```

**Random Affine Transformations:**

```python
def random_affine(image, labels, degrees=10, translate=0.1, scale=0.1):
    """
    Aplica transformaciones afines aleatorias
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # Rotación
    angle = random.uniform(-degrees, degrees)
    
    # Escala
    scale_factor = random.uniform(1 - scale, 1 + scale)
    
    # Traslación
    tx = random.uniform(-translate, translate) * w
    ty = random.uniform(-translate, translate) * h
    
    # Matriz de transformación
    M = cv2.getRotationMatrix2D(center, angle, scale_factor)
    M[0, 2] += tx
    M[1, 2] += ty
    
    # Aplicar transformación
    transformed_image = cv2.warpAffine(image, M, (w, h), borderValue=(114, 114, 114))
    
    # Transformar bounding boxes
    transformed_labels = transform_bboxes(labels, M)
    
    return transformed_image, transformed_labels
```

**HSV Augmentation:**

```python
def hsv_augmentation(image, h_gain=0.015, s_gain=0.7, v_gain=0.4):
    """
    Variación de color en espacio HSV
    """
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1
    
    # Convertir a HSV
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    
    # Aplicar variaciones
    hue = (hue * r[0]) % 180
    sat = np.clip(sat * r[1], 0, 255)
    val = np.clip(val * r[2], 0, 255)
    
    # Reconvertir a RGB
    augmented = cv2.merge((hue, sat, val)).astype(np.uint8)
    augmented = cv2.cvtColor(augmented, cv2.COLOR_HSV2RGB)
    
    return augmented
```

**Pipeline Completo de Augmentation:**

```python
class AugmentationPipeline:
    def __init__(self, config):
        self.config = config
    
    def __call__(self, image, labels):
        # 1. Mosaic (probabilidad)
        if random.random() < self.config['mosaic']:
            image, labels = mosaic_augmentation([image], [labels])
        
        # 2. HSV
        if random.random() < 0.9:
            image = hsv_augmentation(
                image,
                self.config['hsv_h'],
                self.config['hsv_s'],
                self.config['hsv_v']
            )
        
        # 3. Affine
        if random.random() < 0.9:
            image, labels = random_affine(
                image, labels,
                self.config['degrees'],
                self.config['translate'],
                self.config['scale']
            )
        
        # 4. Flip horizontal
        if random.random() < self.config['fliplr']:
            image = np.fliplr(image)
            labels = flip_bboxes_horizontal(labels)
        
        return image, labels
```

### 4.3.4 Monitoreo del Entrenamiento

#### Control del Sobreajuste

**Early Stopping:**

```python
class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered after {self.counter} epochs")
        else:
            self.best_score = val_score
            self.counter = 0
        
        return self.early_stop
```

**Monitoreo de Train vs Val Loss:**

```python
def plot_train_val_loss(train_losses, val_losses):
    """
    Visualiza divergencia entre train y val loss
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Detectar overfitting
    gap = np.array(val_losses) - np.array(train_losses)
    if np.mean(gap[-10:]) > 0.1:
        plt.text(
            len(train_losses)//2, max(val_losses),
            '⚠️ Possible Overfitting Detected',
            ha='center', color='red', fontsize=12
        )
    
    plt.savefig('train_val_loss.png', dpi=300)
    plt.close()
```

#### Ajuste Dinámico de Parámetros

**Learning Rate Scheduling:**

```python
class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_init, lr_final):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_init = lr_init
        self.lr_final = lr_final
    
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_init * (epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_final + (self.lr_init - self.lr_final) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        
        # Actualizar learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

**Adaptive Augmentation:**

```python
class AdaptiveAugmentation:
    def __init__(self, initial_strength=1.0):
        self.strength = initial_strength
    
    def adjust(self, epoch, val_metrics):
        """
        Ajusta fuerza de augmentation según rendimiento
        """
        if epoch > 20:
            # Si hay overfitting, aumentar augmentation
            if val_metrics['val_loss'] > val_metrics['train_loss'] * 1.2:
                self.strength = min(self.strength * 1.1, 2.0)
                print(f"Increasing augmentation strength to {self.strength:.2f}")
            # Si underfitting, reducir augmentation
            elif val_metrics['map50'] < 0.5:
                self.strength = max(self.strength * 0.9, 0.5)
                print(f"Decreasing augmentation strength to {self.strength:.2f}")
```

#### Registro de Métricas

**TensorBoard Integration:**

```python
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    def __init__(self, log_dir='runs/experiment'):
        self.writer = SummaryWriter(log_dir)
    
    def log_scalars(self, metrics_dict, step):
        """Registra métricas escalares"""
        for key, value in metrics_dict.items():
            self.writer.add_scalar(key, value, step)
    
    def log_images(self, images, step):
        """Registra imágenes de ejemplo"""
        self.writer.add_images('predictions', images, step)
    
    def log_hyperparameters(self, hparams, metrics):
        """Registra hiperparámetros y resultados finales"""
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        self.writer.close()

# Uso
logger = TrainingLogger()

for epoch in range(epochs):
    train_metrics = train_one_epoch(model, train_loader)
    val_metrics = validate(model, val_loader)
    
    # Log
    logger.log_scalars({
        'train/loss': train_metrics['loss'],
        'val/loss': val_metrics['loss'],
        'val/mAP@0.5': val_metrics['map50'],
        'val/mAP@0.5:0.95': val_metrics['map50_95'],
    }, epoch)
```

---

*Continúa en [03_RESULTADOS_Y_EVALUACION.md](./03_RESULTADOS_Y_EVALUACION.md)*
