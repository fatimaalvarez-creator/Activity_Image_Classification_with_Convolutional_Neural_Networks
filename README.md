# 🛰️ Activity: Image Classification with Convolutional Neural Networks

> **Course activity:** Build and compare two image classification approaches on satellite images of Mexican biomes — a Multilayer Perceptron using color histogram features and a Convolutional Neural Network learning directly from pixels — both evaluated with cross-validation.

---

## 📋 Overview

This repository contains three exercises that walk through the full image classification workflow — from manual feature engineering (color histograms) to end-to-end deep learning (CNN), finishing with a head-to-head comparison of both approaches.

All neural networks are implemented in **Keras (TensorFlow)** as required, with **scikit-learn** handling preprocessing, cross-validation, and evaluation metrics. **OpenCV** is used for image loading and preprocessing.

---

## 🗂️ Repository Structure

```
📦 ACTIVITY_IMAGE_CLASSIFICATION_WITH_CONVOLUTIONAL_NEURAL_NETWORKS
├── 📁 Biomas/
│   ├── 📁 Agua/                          ← Satellite images of water bodies
│   ├── 📁 Bosque/                        ← Satellite images of forests
│   ├── 📁 Ciudad/                        ← Satellite images of cities
│   ├── 📁 Cultivo/                       ← Satellite images of farmland
│   ├── 📁 Desierto/                      ← Satellite images of deserts
│   └── 📁 Montaña/                       ← Satellite images of mountains
├── 📁 Exercise1/
│   ├── 📄 Exercise1.py                   ← MLP + Color Histograms
│   └── 🖼️ Exercise1.png                  ← Per-fold accuracy & confusion matrix
├── 📁 Exercise2/
│   ├── 📄 Exercise2.py                   ← CNN classifier (Keras)
│   └── 🖼️ Exercise2.png                  ← Per-fold accuracy, learning curves, confusion matrix
├── 📁 Exercise3/
│   ├── 📄 Exercise3.py                   ← MLP vs CNN comparison
│   └── 🖼️ Exercise3.png                  ← Side-by-side metrics, box plots, radar chart
├── 📓 Image_Classification_Biomes.ipynb  ← Unified notebook (all three exercises)
└── 📄 README.md
```

---

## 📚 Exercises Summary

### Exercise 1 — MLP with Color Histograms
**Dataset:** `Biomas/` — Satellite images of 6 Mexican biomes  
**Features:** 32-bin normalized histogram per RGB channel → **96-D feature vector**  
**Architecture:** `Input(96) → Dense(256, ReLU) → Dense(128, ReLU) → Dense(6, Softmax)`  
**Optimizer:** Adam | **Early Stopping:** patience=15 | **Evaluation:** 5-fold stratified CV

| Fold | Train Accuracy | Validation Accuracy |
|------|----------------|----------------------|
| 1 | ~92% | ~78% |
| 2 | ~93% | ~80% |
| 3 | ~92% | ~79% |
| 4 | ~93% | ~81% |
| 5 | ~92% | ~78% |
| **Mean ± SD** | **~92%** | **~79% ± 1.5%** |

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Agua | High | High | High |
| Bosque | Moderate | Moderate | Moderate |
| Ciudad | Moderate | Moderate | Moderate |
| Cultivo | Moderate | Moderate | Moderate |
| Desierto | High | High | High |
| Montaña | Moderate | Moderate | Moderate |

**Key finding:** Color histograms provide a strong baseline thanks to the visually distinctive palettes of biomes like Water (blue) and Desert (warm tones). Confusion concentrates on biomes that share dominant colors — Forest vs. Crop (both green) and Mountain vs. City (both gray/brown) — exposing the main limitation of histogram features: **complete loss of spatial information**.

> *Note: replace the placeholder ranges above with your actual experimental results.*

---

### Exercise 2 — Convolutional Neural Network

> ⚠️ Implemented in **Keras (TensorFlow)** as required.

**Dataset:** Same biome images, used as raw RGB pixels (48×48×3) for CPU feasibility.

**Architecture:**
```
Input(48, 48, 3)
  → Conv2D(16, 3×3, ReLU) → MaxPool(2×2) → Dropout(0.25)
  → Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Dropout(0.25)
  → Flatten → Dense(64, ReLU) → Dropout(0.4)
  → Dense(6, Softmax)
```

**Optimizer:** Adam (lr=1e-3) | **Loss:** Sparse Categorical Crossentropy  
**Epochs:** 15 (with EarlyStopping patience=4) | **Batch size:** 64  
**Evaluation:** 5-fold stratified CV

| Fold | Validation Accuracy | F1 Macro |
|------|----------------------|----------|
| 1 | ~87% | ~86% |
| 2 | ~85% | ~84% |
| 3 | ~88% | ~87% |
| 4 | ~86% | ~85% |
| 5 | ~87% | ~86% |
| **Mean ± SD** | **~87% ± 1%** | **~86% ± 1%** |

| Class | Precision | Recall | F1-score |
|-------|-----------|--------|----------|
| Agua | High | High | High |
| Bosque | High | High | High |
| Ciudad | High | High | High |
| Cultivo | High | Moderate | High |
| Desierto | High | High | High |
| Montaña | Moderate | High | Moderate |

**Key finding:** The CNN learns hierarchical features — bordes, texturas, and regional structure — that the histogram-based MLP cannot capture. The biggest gains over the MLP appear precisely where colors overlap: Forest vs. Crop (distinguished by texture) and City vs. Mountain (distinguished by geometric vs. organic shapes).

> *Note: replace the placeholder ranges above with your actual experimental results.*

---

### Exercise 3 — Head-to-Head Comparison

**Goal:** Compare both models using the same cross-validation protocol and held-out test set.

| Metric | MLP + Histograms | CNN |
|--------|------------------|-----|
| CV Accuracy (mean ± SD) | ~79% ± 1.5% | **~87% ± 1.0%** |
| CV F1 Macro (mean ± SD) | ~78% ± 1.5% | **~86% ± 1.0%** |
| Test Accuracy | ~80% | **~87%** |
| Test F1 Macro | ~79% | **~86%** |
| Test Precision (Macro) | ~80% | **~87%** |
| Test Recall (Macro) | ~79% | **~86%** |
| CV Training Time | **~10 s** | ~600 s |

**Comparison summary:**

| Aspect | MLP + Histograms | CNN |
|--------|------------------|-----|
| Feature engineering | Manual (histograms) | Learned end-to-end |
| Information used | **Global** color distribution | Local color + **spatial** structure |
| Translation invariance | N/A | Yes (via MaxPooling) |
| Training speed | Very fast | Slower (GPU recommended) |
| Best on biomes with… | Distinctive colors (Water, Desert) | Distinctive textures/shapes (Forest, City) |

**Key finding:** The CNN consistently outperforms the histogram-based MLP, especially on biome pairs that share color palettes. The MLP remains valuable as a fast, interpretable baseline — and for biomes with a unique color signature, it can match the CNN. With a larger dataset and a deeper CNN at 64×64 input, the gap would widen further in favor of the CNN.

> *Note: replace the placeholder values above with your actual experimental results.*

---

## ⚙️ Setup & Requirements

### Install dependencies

```bash
pip install tensorflow scikit-learn opencv-python matplotlib numpy seaborn
```

### Dataset

Place the satellite images under a `Biomas/` folder, organized in one subfolder per biome:

```
Biomas/
├── Agua/      *.jpg
├── Bosque/    *.jpg
├── Ciudad/    *.jpg
├── Cultivo/   *.jpg
├── Desierto/  *.jpg
└── Montaña/   *.jpg
```

> **Windows users:** The scripts use a Unicode-safe image reader (`np.fromfile` + `cv2.imdecode`) so paths with non-ASCII characters like `Montaña` load correctly. Standard `cv2.imread()` would silently fail on those paths.

### Run the scripts

```bash
# Exercise 1 — MLP with Color Histograms
cd Exercise1
python Exercise1.py

# Exercise 2 — CNN (Keras)
cd ../Exercise2
python Exercise2.py

# Exercise 3 — Head-to-head comparison
cd ../Exercise3
python Exercise3.py
```

Each script saves a results figure (`ExerciseN.png`) in its own folder.

### Or run everything in one notebook

```bash
jupyter notebook Image_Classification_Biomes.ipynb
```

> Adjust `DATASET_DIR` in the Setup cell if your `Biomas/` folder is not at the expected relative path.

---

## ⚡ Performance Notes

- TensorFlow ≥ 2.11 **no longer supports native GPU on Windows**. The provided scripts are tuned for CPU (48×48 input, 2 conv blocks, 15 epochs) and finish in **~8–15 minutes** end-to-end.
- For higher accuracy with GPU access:
  - Increase `IMG_SIZE` to `(64, 64)` or `(96, 96)`
  - Add a third convolutional block (Conv2D 64 → Conv2D 128)
  - Raise `EPOCHS_CNN` to 25–40
  - Optionally add data augmentation (`RandomFlip`, `RandomRotation`)
- **Google Colab** provides free GPU access — recommended for full-scale experiments.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.11 | Programming language |
| NumPy | Numerical operations, histogram computation |
| OpenCV (`opencv-python`) | Image loading, resizing, color conversion |
| TensorFlow / Keras | CNN implementation (Exercise 2 & 3) |
| scikit-learn | MLP, cross-validation, metrics, preprocessing |
| Matplotlib / Seaborn | Results visualization |

---

## 📌 Key Concepts Covered

- **Manual feature engineering** with color histograms vs. **end-to-end representation learning** with CNNs
- **Unicode-safe image I/O** on Windows (`np.fromfile` + `cv2.imdecode`)
- **5-fold stratified cross-validation** to estimate generalization reliably
- **Convolutional layers, pooling, and dropout** as the standard building blocks of modern image classifiers
- **EarlyStopping & ReduceLROnPlateau** as practical training callbacks
- **Confusion matrix analysis** to diagnose which classes the model confuses
- **Trade-offs**: feature engineering speed and interpretability (MLP) vs. accuracy and spatial reasoning (CNN)
- **Multi-metric comparison** with grouped bar charts, box plots, and radar charts

---

*Implemented with Python 3.11 · NumPy · OpenCV · TensorFlow/Keras · scikit-learn · Matplotlib · Seaborn*
