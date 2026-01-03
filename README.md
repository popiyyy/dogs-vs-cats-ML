# 🐱🐶 Dogs vs Cats Classification

Image classification model menggunakan Convolutional Neural Network (CNN) untuk membedakan gambar kucing dan anjing.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📋 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Usage](#-usage)
- [Results](#-expected-results)

---

## 🎯 Overview

Proyek ini membangun model klasifikasi gambar untuk membedakan **kucing** dan **anjing** menggunakan:

- **Deep Learning** dengan TensorFlow/Keras
- **CNN (Convolutional Neural Network)** architecture
- **Data Augmentation** untuk meningkatkan generalisasi
- **Transfer Learning** (optional) dengan pretrained models

| Aspek | Detail |
|-------|--------|
| **Task** | Binary Image Classification |
| **Classes** | Cat, Dog |
| **Framework** | TensorFlow / Keras |

---

## 📊 Dataset

Dataset menggunakan **Dogs vs Cats** dari Kaggle:

| Info | Detail |
|------|--------|
| **Source** | [Kaggle - Dogs vs Cats](https://www.kaggle.com/datasets/hassanaitnacer/dogs-vs-cats) |
| **Total Images** | ~10,000 |
| **Distribution** | Cat: ~5,000 \| Dog: ~5,000 |
| **Format** | JPG |

### 📥 Download Dataset

#### Prasyarat
1. Buat akun [Kaggle](https://www.kaggle.com/)
2. Download API key dari [Account Settings](https://www.kaggle.com/settings) → **Create New Token**
3. Simpan file `kaggle.json` ke:
   - **Windows**: `C:\Users\<username>\.kaggle\kaggle.json`
   - **Linux/Mac**: `~/.kaggle/kaggle.json`

#### Install Kaggle CLI
```bash
pip install kaggle
```

#### Download & Extract Dataset
```bash
# Download dataset
kaggle datasets download hassanaitnacer/dogs-vs-cats

# Extract ke folder datashet
unzip dogs-vs-cats.zip -d datashet/

# Hapus file zip (optional)
rm dogs-vs-cats.zip
```

> **⚠️ Note**: Pastikan struktur folder setelah extract seperti berikut:
> ```
> datashet/
> ├── cat/    # berisi gambar kucing
> └── dog/    # berisi gambar anjing
> ```

---

## ⚙️ Installation

### 1. Clone Repository
```bash
git clone https://github.com/<username>/dogs-vs-cats-ML.git
cd dogs-vs-cats-ML
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Dataset
Ikuti instruksi di bagian [Download Dataset](#-download-dataset)

---

## 📁 Project Structure

```
dogs-vs-cats-ML/
├── datashet/               # Dataset images (not in git)
│   ├── cat/                # ~5000 cat images
│   └── dog/                # ~5000 dog images
├── main/
│   └── main.ipynb          # Main training notebook
├── models/                 # Saved trained models
├── logs/                   # Training logs (TensorBoard)
├── .gitignore              # Git ignore rules
├── WORKFLOW.md             # Detailed workflow guide
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## 🚀 Usage

### Training Model

1. Buka Jupyter Notebook:
```bash
jupyter notebook main/main.ipynb
```

2. Atau jalankan via terminal:
```bash
python main/main.py
```

### Monitor Training (TensorBoard)
```bash
tensorboard --logdir=logs
```

### Inference (Prediksi)
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('models/cat_dog_classifier.h5')

# Load dan preprocess image
img = image.load_img('path/to/image.jpg', target_size=(150, 150))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
result = 'Dog' if prediction[0] > 0.5 else 'Cat'
print(f'Prediction: {result}')
```

---

## 📈 Expected Results

| Metric | Target |
|--------|--------|
| Training Accuracy | > 90% |
| Validation Accuracy | > 85% |
| Test Accuracy | > 80% |

---

## 🛠️ Technologies Used

- **Python** 3.10+
- **TensorFlow** 2.x
- **Keras** - High-level neural networks API
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation
- **Matplotlib** - Visualization
- **Pillow** - Image processing

---

