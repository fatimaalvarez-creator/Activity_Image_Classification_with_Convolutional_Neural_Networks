"""
EJERCICIO 3 — Comparación MLP vs CNN
Carga ambos modelos, los entrena y genera comparativa visual + análisis.

Estructura esperada del workspace:
    Activity_Image_Classification.../
    ├── Biomas/
    │   ├── Agua/  Bosque/  Ciudad/  Cultivo/  Desierto/  Montaña/
    └── Exercise3/
        └── ejercicio3_comparacion.py   ← este archivo

Ejecutar desde la carpeta Exercise3:
    cd Exercise3
    python ejercicio3_comparacion.py
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import cv2
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, accuracy_score
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    print("❌ TensorFlow no instalado. pip install tensorflow")
    sys.exit(1)

# ──────────────────────────────────────────────
# CONFIGURACIÓN COMPARTIDA
# ──────────────────────────────────────────────
DATASET_DIR  = "../Biomas"       # ← Carpeta hermana de Exercise3/
IMG_SIZE     = (64, 64)
N_BINS       = 32
N_SPLITS     = 5
BATCH_SIZE   = 32
EPOCHS_CNN   = 40
RANDOM_STATE = 42

# Detección automática de clases → tolerante a tildes
BIOMES = sorted([p.name for p in Path(DATASET_DIR).iterdir()
                 if p.is_dir()]) if Path(DATASET_DIR).exists() else []
N_CLASSES = len(BIOMES)


# ──────────────────────────────────────────────
# UTILIDADES DE CARGA
# ──────────────────────────────────────────────
def _glob_imgs(folder: Path):
    return (list(folder.glob("*.jpg")) + list(folder.glob("*.png")) +
            list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG")) +
            list(folder.glob("*.PNG")))


def load_histograms(root_dir):
    X, y = [], []
    root = Path(root_dir)
    for biome in BIOMES:
        folder = root / biome
        if not folder.exists():
            continue
        for img_path in _glob_imgs(folder):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
            hist = np.concatenate([
                np.histogram(img[:, :, c], bins=N_BINS,
                             range=(0, 256), density=True)[0]
                for c in range(3)
            ])
            X.append(hist)
            y.append(biome)
    return np.array(X), np.array(y)


def load_images_cnn(root_dir):
    X, y = [], []
    root = Path(root_dir)
    for biome in BIOMES:
        folder = root / biome
        if not folder.exists():
            continue
        for img_path in _glob_imgs(folder):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img = cv2.cvtColor(cv2.resize(img, IMG_SIZE), cv2.COLOR_BGR2RGB)
            X.append(img.astype(np.float32) / 255.0)
            y.append(biome)
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# MODELOS
# ──────────────────────────────────────────────
def mlp_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu", solver="adam",
            max_iter=500, learning_rate_init=1e-3,
            random_state=RANDOM_STATE,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, verbose=False
        ))
    ])


def build_cnn(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding="same",
                      input_shape=input_shape),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),

        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)), layers.Dropout(0.25),

        layers.Flatten(),
        layers.Dense(256), layers.BatchNormalization(),
        layers.Activation("relu"), layers.Dropout(0.5),
        layers.Dense(N_CLASSES, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ──────────────────────────────────────────────
# EVALUACIÓN CONJUNTA
# ──────────────────────────────────────────────
def evaluate_mlp(X, y_enc, class_names):
    from sklearn.model_selection import train_test_split
    cv  = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                          random_state=RANDOM_STATE)

    print("\n[MLP] Ejecutando cross-validation …")
    t0 = time.time()
    results = cross_validate(
        mlp_pipeline(), X, y_enc,
        cv=cv, scoring=["accuracy", "f1_macro"],
        return_train_score=True, n_jobs=-1
    )
    cv_time = time.time() - t0

    # Final split para métricas detalladas
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2,
        stratify=y_enc, random_state=RANDOM_STATE
    )
    pipe = mlp_pipeline()
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)

    metrics = {
        "name"          : "MLP + Histogramas",
        "cv_acc_mean"   : results["test_accuracy"].mean(),
        "cv_acc_std"    : results["test_accuracy"].std(),
        "cv_f1_mean"    : results["test_f1_macro"].mean(),
        "cv_f1_std"     : results["test_f1_macro"].std(),
        "cv_accs"       : results["test_accuracy"],
        "cv_time"       : cv_time,
        "y_te"          : y_te,
        "y_pred"        : y_pred,
        "test_acc"      : accuracy_score(y_te, y_pred),
        "test_f1"       : f1_score(y_te, y_pred, average="macro"),
        "test_precision": precision_score(y_te, y_pred, average="macro"),
        "test_recall"   : recall_score(y_te, y_pred, average="macro"),
        "report"        : classification_report(
                              y_te, y_pred, target_names=class_names)
    }
    return metrics


def evaluate_cnn(X, y_enc, class_names):
    from sklearn.model_selection import train_test_split
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    input_shape = X.shape[1:]

    print("\n[CNN] Ejecutando cross-validation …")
    t0 = time.time()
    fold_accs, fold_f1s = [], []

    for fold, (tr_idx, val_idx) in enumerate(cv.split(X, y_enc)):
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE + fold)

        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y_enc[tr_idx], y_enc[val_idx]

        model = build_cnn(input_shape)
        model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=EPOCHS_CNN, batch_size=BATCH_SIZE,
            callbacks=[
                callbacks.EarlyStopping(monitor="val_accuracy",
                                        patience=8,
                                        restore_best_weights=True,
                                        verbose=0),
                callbacks.ReduceLROnPlateau(monitor="val_loss",
                                            factor=0.5, patience=4,
                                            verbose=0, min_lr=1e-6)
            ],
            verbose=0
        )
        y_pred_v = np.argmax(model.predict(X_val, verbose=0), axis=1)
        fold_accs.append(accuracy_score(y_val, y_pred_v))
        fold_f1s.append(f1_score(y_val, y_pred_v, average="macro"))
        print(f"  Fold {fold+1}: acc={fold_accs[-1]*100:.1f}%  "
              f"f1={fold_f1s[-1]*100:.1f}%")

    cv_time = time.time() - t0

    # Final
    tf.keras.backend.clear_session()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_enc, test_size=0.2,
        stratify=y_enc, random_state=RANDOM_STATE
    )
    model = build_cnn(input_shape)
    model.fit(
        X_tr, y_tr, validation_split=0.1,
        epochs=EPOCHS_CNN, batch_size=BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(monitor="val_accuracy",
                                    patience=10,
                                    restore_best_weights=True,
                                    verbose=0)
        ],
        verbose=0
    )
    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    metrics = {
        "name"          : "CNN",
        "cv_acc_mean"   : np.mean(fold_accs),
        "cv_acc_std"    : np.std(fold_accs),
        "cv_f1_mean"    : np.mean(fold_f1s),
        "cv_f1_std"     : np.std(fold_f1s),
        "cv_accs"       : np.array(fold_accs),
        "cv_time"       : cv_time,
        "y_te"          : y_te,
        "y_pred"        : y_pred,
        "test_acc"      : accuracy_score(y_te, y_pred),
        "test_f1"       : f1_score(y_te, y_pred, average="macro"),
        "test_precision": precision_score(y_te, y_pred, average="macro"),
        "test_recall"   : recall_score(y_te, y_pred, average="macro"),
        "report"        : classification_report(
                              y_te, y_pred, target_names=class_names)
    }
    return metrics


# ──────────────────────────────────────────────
# REPORTE TEXTUAL
# ──────────────────────────────────────────────
def print_comparison(m_mlp, m_cnn):
    sep = "=" * 62

    print(f"\n{sep}")
    print("  COMPARATIVA FINAL: MLP vs CNN")
    print(sep)

    print(f"{'Métrica':<22} {'MLP':>15} {'CNN':>15}")
    print("-" * 55)

    rows = [
        ("CV Accuracy",
         f"{m_mlp['cv_acc_mean']*100:.2f}±{m_mlp['cv_acc_std']*100:.2f}%",
         f"{m_cnn['cv_acc_mean']*100:.2f}±{m_cnn['cv_acc_std']*100:.2f}%"),
        ("CV F1 Macro",
         f"{m_mlp['cv_f1_mean']*100:.2f}±{m_mlp['cv_f1_std']*100:.2f}%",
         f"{m_cnn['cv_f1_mean']*100:.2f}±{m_cnn['cv_f1_std']*100:.2f}%"),
        ("Test Accuracy",
         f"{m_mlp['test_acc']*100:.2f}%",
         f"{m_cnn['test_acc']*100:.2f}%"),
        ("Test F1 Macro",
         f"{m_mlp['test_f1']*100:.2f}%",
         f"{m_cnn['test_f1']*100:.2f}%"),
        ("Test Precision",
         f"{m_mlp['test_precision']*100:.2f}%",
         f"{m_cnn['test_precision']*100:.2f}%"),
        ("Test Recall",
         f"{m_mlp['test_recall']*100:.2f}%",
         f"{m_cnn['test_recall']*100:.2f}%"),
        ("Tiempo CV",
         f"{m_mlp['cv_time']:.1f} s",
         f"{m_cnn['cv_time']:.1f} s"),
    ]

    for label, v_mlp, v_cnn in rows:
        print(f"  {label:<20} {v_mlp:>15} {v_cnn:>15}")

    print(f"\n{sep}")
    diff_acc = (m_cnn['test_acc'] - m_mlp['test_acc']) * 100
    winner   = "CNN" if diff_acc > 0 else "MLP"
    print(f"  🏆  GANADOR: {winner} (diferencia en acc: "
          f"{abs(diff_acc):.2f}%)")
    print(sep)

    print("\n📝 ANÁLISIS\n")
    print("""  MLP + Histogramas de Color:
  • Extrae información GLOBAL de color de cada imagen.
  • Histogramas capturan la distribución cromática pero
    pierden la información espacial (¿dónde está el color?).
  • Rápido de entrenar y sin dependencia de GPU.
  • Limitación: dos imágenes con la misma distribución de
    colores pero diferente estructura parecen idénticas.
  • Adecuado como baseline robusto y explicable.

  CNN:
  • Aprende filtros locales que detectan texturas, bordes
    y patrones espaciales directamente sobre los píxeles.
  • Invariante a pequeñas traslaciones (MaxPooling).
  • Captura jerarquía de características:
      Cap. 1 → bordes y colores primarios
      Cap. 2 → texturas y patrones
      Cap. 3 → estructuras complejas (edificios, copas, agua)
  • Requiere más datos y tiempo de entrenamiento.
  • Generalmente supera al MLP en tareas de visión.

  ¿Por qué la CNN puede superar al MLP?
  Las imágenes satelitales tienen patrones espaciales muy
  distintivos: el Bosque tiene textura granular y verde,
  el Agua es homogénea y azul/oscura, la Ciudad tiene
  geometría rectilínea, el Desierto tiene tonos cálidos
  uniformes. La CNN captura estas texturas y formas;
  el histograma sólo ve los colores agregados.
""")

    print("\n📋 Classification Report — MLP")
    print(m_mlp['report'])
    print("📋 Classification Report — CNN")
    print(m_cnn['report'])


# ──────────────────────────────────────────────
# VISUALIZACIÓN COMPARATIVA
# ──────────────────────────────────────────────
def plot_comparison(m_mlp, m_cnn, class_names):
    dark_bg  = "#0d1117"
    panel_bg = "#161b22"
    blue     = "#58a6ff"
    orange   = "#f78166"
    green    = "#3fb950"
    purple   = "#bc8cff"
    text_col = "#e6edf3"
    grid_col = "#21262d"

    fig = plt.figure(figsize=(20, 13), facecolor=dark_bg)
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.45, wspace=0.38)

    # ── 1. Barras comparativas — métricas ───────────────
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor(panel_bg)

    metrics_labels = ["CV Accuracy", "CV F1 Macro",
                      "Test Accuracy", "Test F1", "Precision", "Recall"]
    mlp_vals = [
        m_mlp["cv_acc_mean"], m_mlp["cv_f1_mean"],
        m_mlp["test_acc"],    m_mlp["test_f1"],
        m_mlp["test_precision"], m_mlp["test_recall"]
    ]
    cnn_vals = [
        m_cnn["cv_acc_mean"], m_cnn["cv_f1_mean"],
        m_cnn["test_acc"],    m_cnn["test_f1"],
        m_cnn["test_precision"], m_cnn["test_recall"]
    ]

    x     = np.arange(len(metrics_labels))
    width = 0.35
    b1 = ax1.bar(x - width/2, [v*100 for v in mlp_vals],
                 width, color=blue,   alpha=0.85, label="MLP", zorder=3)
    b2 = ax1.bar(x + width/2, [v*100 for v in cnn_vals],
                 width, color=orange, alpha=0.85, label="CNN", zorder=3)

    for bar in list(b1) + list(b2):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}",
                 ha="center", va="bottom",
                 color=text_col, fontsize=7.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels, color=text_col, fontsize=9)
    ax1.set_ylabel("Valor (%)", color=text_col)
    ax1.set_ylim(0, 115)
    ax1.set_title("Comparación de Métricas: MLP vs CNN",
                  color=text_col, fontsize=11, pad=10)
    ax1.tick_params(colors=text_col)
    ax1.spines[:].set_color(grid_col)
    ax1.grid(axis="y", color=grid_col, lw=0.7, zorder=0)
    ax1.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=9)

    # ── 2. Box plot CV Accuracy ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(panel_bg)
    data   = [m_mlp["cv_accs"]*100, m_cnn["cv_accs"]*100]
    labels = ["MLP", "CNN"]
    colors = [blue, orange]
    bp = ax2.boxplot(data, patch_artist=True,
                     medianprops=dict(color=green, lw=2),
                     whiskerprops=dict(color=text_col),
                     capprops=dict(color=text_col),
                     flierprops=dict(markerfacecolor=purple, markersize=5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    for i, vals in enumerate(data):
        ax2.scatter([i+1]*len(vals), vals,
                    color=colors[i], zorder=5, s=45, alpha=0.9)

    ax2.set_xticklabels(labels, color=text_col, fontsize=10)
    ax2.set_ylabel("CV Accuracy (%)", color=text_col)
    ax2.set_title("Distribución CV (5-fold)", color=text_col, fontsize=10)
    ax2.tick_params(colors=text_col)
    ax2.spines[:].set_color(grid_col)
    ax2.grid(axis="y", color=grid_col, lw=0.7)

    # ── 3. Matrices de confusión: MLP ────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(panel_bg)
    cm_mlp = confusion_matrix(m_mlp["y_te"], m_mlp["y_pred"])
    im3    = ax3.imshow(cm_mlp, cmap="Blues", interpolation="nearest")
    plt.colorbar(im3, ax=ax3).ax.yaxis.set_tick_params(color=text_col)
    ticks = np.arange(len(class_names))
    ax3.set_xticks(ticks)
    ax3.set_yticks(ticks)
    ax3.set_xticklabels(class_names, rotation=40, ha="right",
                        color=text_col, fontsize=7)
    ax3.set_yticklabels(class_names, color=text_col, fontsize=7)
    ax3.set_title("Confusión — MLP", color=text_col, fontsize=10)
    ax3.tick_params(colors=text_col)
    ax3.spines[:].set_color(grid_col)
    thresh3 = cm_mlp.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax3.text(j, i, cm_mlp[i, j], ha="center", va="center",
                     fontsize=8,
                     color="white" if cm_mlp[i, j] > thresh3 else text_col)

    # ── 4. Matrices de confusión: CNN ────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(panel_bg)
    cm_cnn = confusion_matrix(m_cnn["y_te"], m_cnn["y_pred"])
    im4    = ax4.imshow(cm_cnn, cmap="Oranges", interpolation="nearest")
    plt.colorbar(im4, ax=ax4).ax.yaxis.set_tick_params(color=text_col)
    ax4.set_xticks(ticks)
    ax4.set_yticks(ticks)
    ax4.set_xticklabels(class_names, rotation=40, ha="right",
                        color=text_col, fontsize=7)
    ax4.set_yticklabels(class_names, color=text_col, fontsize=7)
    ax4.set_title("Confusión — CNN", color=text_col, fontsize=10)
    ax4.tick_params(colors=text_col)
    ax4.spines[:].set_color(grid_col)
    thresh4 = cm_cnn.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax4.text(j, i, cm_cnn[i, j], ha="center", va="center",
                     fontsize=8,
                     color="white" if cm_cnn[i, j] > thresh4 else text_col)

    # ── 5. Radar / spider chart ──────────────────────────
    ax5 = fig.add_subplot(gs[1, 2], polar=True)
    ax5.set_facecolor(panel_bg)

    radar_labels = ["Accuracy\n(CV)", "F1\n(CV)",
                    "Test Acc", "Test F1",
                    "Precision", "Recall"]
    N = len(radar_labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    mlp_r = [m_mlp["cv_acc_mean"], m_mlp["cv_f1_mean"],
              m_mlp["test_acc"],   m_mlp["test_f1"],
              m_mlp["test_precision"], m_mlp["test_recall"]]
    cnn_r = [m_cnn["cv_acc_mean"], m_cnn["cv_f1_mean"],
              m_cnn["test_acc"],   m_cnn["test_f1"],
              m_cnn["test_precision"], m_cnn["test_recall"]]
    mlp_r += mlp_r[:1]
    cnn_r += cnn_r[:1]

    ax5.plot(angles, mlp_r, color=blue,   lw=2, label="MLP")
    ax5.fill(angles, mlp_r, color=blue,   alpha=0.15)
    ax5.plot(angles, cnn_r, color=orange, lw=2, label="CNN")
    ax5.fill(angles, cnn_r, color=orange, alpha=0.15)

    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(radar_labels, color=text_col, fontsize=7.5)
    ax5.set_ylim(0, 1)
    ax5.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax5.set_yticklabels(["25%", "50%", "75%", "100%"],
                        color=text_col, fontsize=6)
    ax5.grid(color=grid_col, lw=0.8)
    ax5.spines["polar"].set_color(grid_col)
    ax5.set_facecolor(panel_bg)
    ax5.set_title("Radar — MLP vs CNN",
                  color=text_col, fontsize=10, pad=15)
    ax5.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15),
               facecolor=panel_bg, labelcolor=text_col, fontsize=8)

    fig.suptitle(
        "Ejercicio 3 — Comparación MLP vs CNN | Biomas de México",
        color=text_col, fontsize=13, y=1.01, fontweight="bold"
    )

    plt.savefig("Exercise3.png",
                dpi=150, bbox_inches="tight", facecolor=dark_bg)
    plt.show()
    print("\n✅ Gráfica guardada: Exercise3.png")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  EJERCICIO 3 — Comparación MLP vs CNN")
    print("=" * 60)

    if not Path(DATASET_DIR).exists():
        print(f"❌ Carpeta '{DATASET_DIR}' no encontrada.")
        print("   Ejecuta este script desde dentro de la carpeta Exercise3/")
        sys.exit(1)

    if not BIOMES:
        print(f"\n❌ La carpeta '{DATASET_DIR}' está vacía.")
        sys.exit(1)

    print(f"\n📂 Carpeta dataset: {Path(DATASET_DIR).resolve()}")
    print(f"   Clases detectadas: {BIOMES}")

    # — Cargar datos para MLP —
    print("\n📁 Cargando histogramas (MLP) …")
    X_hist, y = load_histograms(DATASET_DIR)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_
    print(f"   {X_hist.shape[0]} muestras, {X_hist.shape[1]} features")

    # — Cargar datos para CNN —
    print("\n📁 Cargando imágenes (CNN) …")
    X_img, _ = load_images_cnn(DATASET_DIR)
    print(f"   {X_img.shape}")

    # — Evaluar modelos —
    metrics_mlp = evaluate_mlp(X_hist, y_enc, class_names)
    metrics_cnn = evaluate_cnn(X_img,  y_enc, class_names)

    # — Comparativa textual —
    print_comparison(metrics_mlp, metrics_cnn)

    # — Visualización —
    plot_comparison(metrics_mlp, metrics_cnn, class_names)