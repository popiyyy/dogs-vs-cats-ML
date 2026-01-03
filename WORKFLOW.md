# 🐱🐶 Dogs vs Cats Classification - Workflow

## 📊 Dataset Overview

| Aspek | Detail |
|-------|--------|
| **Dataset** | Dogs vs Cats Image Classification |
| **Total Gambar** | ~10,000 images |
| **Kelas** | 2 kelas (Binary Classification) |
| **Distribusi** | Cat: ~5,000 \| Dog: ~5,000 |
| **Format** | JPG |
| **Struktur** | `datashet/cat/` dan `datashet/dog/` |

---

## 🔄 Workflow Steps

### Step 1: Data Exploration & Visualization
- [ ] Load dan visualisasi sample gambar dari setiap kelas
- [ ] Cek distribusi jumlah gambar per kelas
- [ ] Analisis ukuran/resolusi gambar
- [ ] Identifikasi gambar yang corrupt atau bermasalah

### Step 2: Data Preprocessing
- [ ] Resize gambar ke ukuran seragam (misal: 150x150 atau 224x224)
- [ ] Normalisasi pixel values (0-1 atau -1 to 1)
- [ ] Convert ke format array yang sesuai

### Step 3: Data Augmentation
- [ ] Horizontal/Vertical Flip
- [ ] Rotation (0-40°)
- [ ] Zoom (0.1-0.2)
- [ ] Width/Height Shift
- [ ] Shear transformation

### Step 4: Data Splitting
- [ ] Split dataset menjadi:
  - **Train set**: 70-80%
  - **Validation set**: 10-15%
  - **Test set**: 10-15%

### Step 5: Model Building
Pilih salah satu approach:

#### Option A: Build CNN from Scratch
- [ ] Input Layer
- [ ] Conv2D + MaxPooling layers (3-5 blocks)
- [ ] Flatten
- [ ] Dense layers
- [ ] Output layer (sigmoid untuk binary)

#### Option B: Transfer Learning (Recommended)
- [ ] Load pretrained model (VGG16 / ResNet50 / MobileNetV2)
- [ ] Freeze base layers
- [ ] Add custom classification head
- [ ] Fine-tune if needed

### Step 6: Model Compilation
- [ ] Optimizer: Adam (lr=0.001)
- [ ] Loss: Binary Crossentropy
- [ ] Metrics: Accuracy

### Step 7: Training
- [ ] Set callbacks:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau
- [ ] Train model
- [ ] Monitor training/validation loss & accuracy

### Step 8: Evaluation
- [ ] Evaluate on test set
- [ ] Generate metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- [ ] Visualisasi:
  - Confusion Matrix
  - Training History (Loss & Accuracy curves)
  - Sample predictions

### Step 9: Model Optimization (Optional)
- [ ] Hyperparameter tuning
- [ ] Try different architectures
- [ ] Ensemble methods

### Step 10: Save & Deploy (Optional)
- [ ] Save model (.h5 atau SavedModel format)
- [ ] Create inference script
- [ ] Deploy sebagai API (Flask/FastAPI)

---

## 📦 Required Libraries

### Already Installed ✅
| Library | Version | Kegunaan |
|---------|---------|----------|
| numpy | 2.4.0 | Operasi array/matrix |
| pandas | 2.3.3 | Data manipulation |
| matplotlib | 3.10.8 | Visualisasi dasar |
| pillow | 12.0.0 | Image processing |

### Newly Installed ✅
| Library | Kegunaan |
|---------|----------|
| tensorflow | Deep learning framework |
| keras | High-level API untuk neural networks |

### Recommended to Install
```bash
pip install scikit-learn seaborn
```

| Library | Kegunaan |
|---------|----------|
| scikit-learn | Train/test split, metrics, confusion matrix |
| seaborn | Advanced visualization |

---

## 📁 Project Structure (Recommended)

```
dogs-vs-cats-ML/
├── datashet/
│   ├── cat/          # 5000 cat images
│   └── dog/          # 5000 dog images
├── main/
│   └── main.ipynb    # Main notebook
├── models/           # Saved models
├── logs/             # Training logs
├── WORKFLOW.md       # This file
└── requirements.txt  # Dependencies
```

---

## 🚀 Quick Start Code

```python
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Parameters
IMG_SIZE = 150
BATCH_SIZE = 32
EPOCHS = 20

# Data Generator with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Load Data
train_generator = train_datagen.flow_from_directory(
    'datashet',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'datashet',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Build Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)
```

---

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| Training Accuracy | > 90% |
| Validation Accuracy | > 85% |
| Test Accuracy | > 80% |

---

*Last Updated: 2026-01-01*
