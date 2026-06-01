"""
EJERCICIO 2 — Clasificador CNN para Biomas
Dataset: Biomas de México (6 clases)
Evaluación: Cross-validation (5-fold)

⚠ NOTA SOBRE TIEMPOS:
   TensorFlow ya NO soporta GPU nativa en Windows (TF ≥ 2.11).
   Este script corre en CPU. Tiempos esperados con ~4000 imgs:
     - Por fold (CV): 3–6 min
     - Total (5 folds + modelo final): 20–40 min
   Si lo encuentras lento:
     1) Usa Google Colab (GPU gratis): https://colab.research.google.com
     2) Reduce EPOCHS o N_SPLITS abajo

Ejecutar:
    cd Exercise2
    python Exercise2.py
"""

import os, sys, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    print(f"✅ TensorFlow {tf.__version__}")
except ImportError:
    print("❌ TensorFlow no instalado. pip install tensorflow")
    sys.exit(1)

# ──────────────────────────────────────────────
# 1. CONFIGURACIÓN (optimizada para CPU)
# ──────────────────────────────────────────────
DATASET_DIR   = "../Biomas"
IMG_SIZE      = (64, 64)
USE_GRAYSCALE = False
N_SPLITS      = 5
BATCH_SIZE    = 64                 # ↑ acelera CPU
EPOCHS        = 25                 # ↓ con early stopping, más que suficiente
RANDOM_STATE  = 42
N_CHANNELS    = 1 if USE_GRAYSCALE else 3

BIOMES = sorted([p.name for p in Path(DATASET_DIR).iterdir()
                 if p.is_dir()]) if Path(DATASET_DIR).exists() else []
N_CLASSES = len(BIOMES)


# ──────────────────────────────────────────────
# 2. LECTURA UNICODE-SAFE (clave para "Montaña" en Windows)
# ──────────────────────────────────────────────
def safe_imread(path):
    """cv2.imread no maneja Unicode en Windows. Esto sí."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ──────────────────────────────────────────────
# 3. CARGA DE IMÁGENES
# ──────────────────────────────────────────────
def load_images(root_dir):
    X, y = [], []
    root = Path(root_dir)
    failed = 0

    for biome in BIOMES:
        folder = root / biome
        if not folder.exists():
            continue
        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + \
               list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG")) + \
               list(folder.glob("*.PNG"))
        if not imgs:
            continue

        loaded = 0
        for img_path in imgs:
            img = safe_imread(img_path)
            if img is None:
                failed += 1
                continue
            img = cv2.resize(img, IMG_SIZE)
            if USE_GRAYSCALE:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            X.append(img.astype(np.float32) / 255.0)
            y.append(biome)
            loaded += 1
        print(f"  📂 {biome:12s}: {loaded}/{len(imgs)} cargadas")

    if failed > 0:
        print(f"\n  ⚠  {failed} archivos no pudieron leerse")
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# 4. ARQUITECTURA CNN
# ──────────────────────────────────────────────
def build_cnn(input_shape, n_classes):
    """3 bloques Conv-BN-ReLU-Pool-Dropout + Dense(256) + Softmax."""
    model = models.Sequential(name="BiomeCNN")
    model.add(keras.Input(shape=input_shape))      # Mejor que input_shape= en Conv2D

    model.add(layers.Conv2D(32, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ──────────────────────────────────────────────
# 5. CROSS-VALIDATION
# ──────────────────────────────────────────────
def run_cross_validation(X, y_enc):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    input_shape = X.shape[1:]

    fold_histories, fold_metrics = [], []

    print(f"\n⏳ Ejecutando {N_SPLITS}-fold cross-validation …")
    print(f"   (Cada fold puede tardar varios minutos en CPU)\n")
    print(f"{'Fold':>5} | {'Val Acc':>10} | {'Val Loss':>10} | {'Tiempo':>8}")
    print("-" * 45)

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        t0 = time.time()
        tf.keras.backend.clear_session()
        np.random.seed(RANDOM_STATE + fold)
        tf.random.set_seed(RANDOM_STATE + fold)

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]

        model = build_cnn(input_shape, N_CLASSES)
        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_accuracy",
                                        patience=6,
                                        restore_best_weights=True,
                                        verbose=0),
                callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.5, patience=3,
                                            verbose=0, min_lr=1e-6)
            ],
            verbose=0
        )
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        fold_histories.append(history.history)
        fold_metrics.append((val_loss, val_acc))
        dt = time.time() - t0
        print(f"  {fold+1:>3} | {val_acc*100:>9.2f}% | {val_loss:>10.4f} | "
              f"{dt:>6.1f}s")

    val_accs = [m[1] for m in fold_metrics]
    print(f"\n📈 Accuracy media (CV): "
          f"{np.mean(val_accs)*100:.2f}% ± {np.std(val_accs)*100:.2f}%")
    return fold_histories, fold_metrics


# ──────────────────────────────────────────────
# 6. MODELO FINAL
# ──────────────────────────────────────────────
def train_final_model(X, y_enc, class_names):
    from sklearn.model_selection import train_test_split
    tf.keras.backend.clear_session()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2,
        stratify=y_enc, random_state=RANDOM_STATE
    )
    model = build_cnn(X.shape[1:], N_CLASSES)
    model.summary()

    print("\n⏳ Entrenando modelo final …")
    history = model.fit(
        X_tr, y_tr, validation_split=0.1,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy", patience=8,
                                    restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                        patience=4, verbose=1, min_lr=1e-6),
            callbacks.ModelCheckpoint("best_cnn_biomes.keras",
                                      save_best_only=True, verbose=0)
        ],
        verbose=1
    )
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print("\n📋 CLASSIFICATION REPORT (CNN — split 80/20)")
    print(classification_report(y_te, y_pred, target_names=class_names))
    return history, y_te, y_pred, model


# ──────────────────────────────────────────────
# 7. VISUALIZACIÓN
# ──────────────────────────────────────────────
def plot_results(fold_histories, fold_metrics, history_final,
                 y_te, y_pred, class_names):
    dark_bg, panel_bg = "#0d1117", "#161b22"
    accent, orange, green = "#58a6ff", "#f78166", "#3fb950"
    text_col, grid_col = "#e6edf3", "#21262d"

    fig = plt.figure(figsize=(18, 11), facecolor=dark_bg)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

    # 1. Accuracy por fold
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(panel_bg)
    folds = np.arange(1, N_SPLITS + 1)
    val_accs = [m[1] * 100 for m in fold_metrics]
    ax1.bar(folds, val_accs, color=accent, alpha=0.85, zorder=3)
    ax1.axhline(np.mean(val_accs), color=green, lw=1.5, ls="--",
                label=f"Media: {np.mean(val_accs):.1f}%", zorder=4)
    ax1.set_xticks(folds); ax1.set_xticklabels([f"F{i}" for i in folds], color=text_col)
    ax1.set_ylabel("Val Accuracy (%)", color=text_col)
    ax1.set_ylim(0, 105)
    ax1.set_title("Accuracy por Fold (CV)", color=text_col, fontsize=10)
    ax1.tick_params(colors=text_col); ax1.spines[:].set_color(grid_col)
    ax1.grid(axis="y", color=grid_col, lw=0.7, zorder=0)
    ax1.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=8)

    # 2. Curva de aprendizaje
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(panel_bg)
    h = history_final.history
    ep = range(1, len(h["accuracy"]) + 1)
    ax2.plot(ep, [v*100 for v in h["accuracy"]], color=accent, lw=2, label="Train")
    ax2.plot(ep, [v*100 for v in h["val_accuracy"]], color=green, lw=2,
             ls="--", label="Val")
    ax2.set_xlabel("Épocas", color=text_col); ax2.set_ylabel("Accuracy (%)", color=text_col)
    ax2.set_title("Curva de Aprendizaje", color=text_col, fontsize=10)
    ax2.tick_params(colors=text_col); ax2.spines[:].set_color(grid_col)
    ax2.grid(color=grid_col, lw=0.7)
    ax2.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=8)

    # 3. Curva de pérdida
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(panel_bg)
    ax3.plot(ep, h["loss"], color=orange, lw=2, label="Train Loss")
    ax3.plot(ep, h["val_loss"], color="#ff7b72", lw=2, ls="--", label="Val Loss")
    ax3.set_xlabel("Épocas", color=text_col); ax3.set_ylabel("Loss", color=text_col)
    ax3.set_title("Curva de Pérdida", color=text_col, fontsize=10)
    ax3.tick_params(colors=text_col); ax3.spines[:].set_color(grid_col)
    ax3.grid(color=grid_col, lw=0.7)
    ax3.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=8)

    # 4. Matriz de confusión
    ax4 = fig.add_subplot(gs[1, :2]); ax4.set_facecolor(panel_bg)
    cm = confusion_matrix(y_te, y_pred)
    im = ax4.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax4).ax.yaxis.set_tick_params(color=text_col)
    ticks = np.arange(len(class_names))
    ax4.set_xticks(ticks); ax4.set_yticks(ticks)
    ax4.set_xticklabels(class_names, rotation=40, ha="right", color=text_col, fontsize=8)
    ax4.set_yticklabels(class_names, color=text_col, fontsize=8)
    ax4.set_xlabel("Predicho", color=text_col); ax4.set_ylabel("Real", color=text_col)
    ax4.set_title("Matriz de Confusión — CNN", color=text_col, fontsize=10)
    ax4.tick_params(colors=text_col); ax4.spines[:].set_color(grid_col)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > thresh else text_col, fontsize=9)

    # 5. Box plot CV
    ax5 = fig.add_subplot(gs[1, 2]); ax5.set_facecolor(panel_bg)
    all_cv = [m[1] * 100 for m in fold_metrics]
    bp = ax5.boxplot(all_cv, patch_artist=True,
                     medianprops=dict(color=green, lw=2),
                     boxprops=dict(facecolor=accent, alpha=0.5, color=accent),
                     whiskerprops=dict(color=text_col),
                     capprops=dict(color=text_col),
                     flierprops=dict(markerfacecolor=orange, markersize=6))
    ax5.scatter([1]*len(all_cv), all_cv, color=accent, zorder=5, s=40, alpha=0.9)
    ax5.set_xticks([1]); ax5.set_xticklabels(["CNN Biomas"], color=text_col)
    ax5.set_ylabel("Val Accuracy (%)", color=text_col)
    ax5.set_title("Distribución CV", color=text_col, fontsize=10)
    ax5.tick_params(colors=text_col); ax5.spines[:].set_color(grid_col)
    ax5.grid(axis="y", color=grid_col, lw=0.7)

    fig.suptitle("Ejercicio 2 — CNN para Clasificación de Biomas de México",
                 color=text_col, fontsize=13, y=1.01, fontweight="bold")
    plt.savefig("Exercise2.png", dpi=150,
                bbox_inches="tight", facecolor=dark_bg)
    plt.show()
    print("\n✅ Gráfica guardada: Exercise2.png")


# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  EJERCICIO 2 — CNN para Biomas de México")
    print(f"  Modo: {'Grises' if USE_GRAYSCALE else 'Color RGB'}")
    print(f"  Imagen: {IMG_SIZE[0]}×{IMG_SIZE[1]}×{N_CHANNELS}")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | Folds: {N_SPLITS}")
    print("=" * 55)

    if not Path(DATASET_DIR).exists():
        print(f"\n❌ No se encontró '{DATASET_DIR}'.")
        sys.exit(1)
    if not BIOMES:
        print(f"\n❌ Carpeta '{DATASET_DIR}' vacía.")
        sys.exit(1)

    print(f"\n📂 Dataset: {Path(DATASET_DIR).resolve()}")
    print(f"   Clases detectadas: {BIOMES}")

    print(f"\n📁 Cargando imágenes …")
    X, y = load_images(DATASET_DIR)
    print(f"\n✅ Dataset cargado: {X.shape}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    unique, counts = np.unique(y, return_counts=True)
    print("\n   Distribución:")
    for cls, cnt in zip(unique, counts):
        print(f"     {cls:12s}: {cnt}")

    fold_histories, fold_metrics = run_cross_validation(X, y_enc)
    history_final, y_te, y_pred, model = train_final_model(X, y_enc, class_names)
    plot_results(fold_histories, fold_metrics, history_final,
                 y_te, y_pred, class_names)
    print("\n✅ Modelo guardado: best_cnn_biomes.keras")