"""
EJERCICIO 1 — Clasificador MLP con Histogramas de Color
Dataset: Biomas de México (6 clases)
Evaluación: Cross-validation (5-fold)

Estructura esperada del workspace:
    Activity_Image_Classification.../
    ├── Biomas/
    │   ├── Agua/  Bosque/  Ciudad/  Cultivo/  Desierto/  Montaña/
    └── Exercise1/
        └── ejercicio1_mlp_histogramas.py   ← este archivo

Ejecutar desde la carpeta Exercise1:
    cd Exercise1
    python ejercicio1_mlp_histogramas.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import cv2

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.pipeline import Pipeline

# ──────────────────────────────────────────────
# 1. CONFIGURACIÓN
# ──────────────────────────────────────────────
DATASET_DIR  = "../Biomas"       # ← Carpeta hermana de Exercise1/
IMG_SIZE     = (64, 64)
N_BINS       = 32
N_SPLITS     = 5
RANDOM_STATE = 42

# Detección automática de clases → tolerante a tildes y mayúsculas
BIOMES = sorted([p.name for p in Path(DATASET_DIR).iterdir()
                 if p.is_dir()]) if Path(DATASET_DIR).exists() else []


# ──────────────────────────────────────────────
# 2. EXTRACCIÓN DE CARACTERÍSTICAS
#    Histograma de color: 3 canales × N_BINS → vector de 3*N_BINS dimensiones
# ──────────────────────────────────────────────
def extract_color_histogram(img_path: str, n_bins: int = N_BINS) -> np.ndarray:
    """
    Carga una imagen RGB y calcula el histograma normalizado
    por canal (R, G, B), concatenando los tres vectores.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)

    hist_features = []
    for channel in range(3):                     # R, G, B
        hist, _ = np.histogram(
            img[:, :, channel],
            bins=n_bins,
            range=(0, 256),
            density=True                         # normalizado
        )
        hist_features.append(hist)

    return np.concatenate(hist_features)         # shape: (3*n_bins,)


def load_dataset(root_dir: str):
    """
    Recorre las subcarpetas del dataset y extrae histogramas + etiquetas.
    """
    X, y = [], []
    root = Path(root_dir)

    for biome in BIOMES:
        folder = root / biome
        if not folder.exists():
            print(f"  ⚠  Carpeta no encontrada: {folder}")
            continue

        imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + \
               list(folder.glob("*.jpeg")) + list(folder.glob("*.JPG")) + \
               list(folder.glob("*.PNG"))

        if not imgs:
            print(f"  ⚠  Sin imágenes en: {folder}")
            continue

        print(f"  📂 {biome}: {len(imgs)} imágenes")

        for img_path in imgs:
            try:
                feat = extract_color_histogram(str(img_path))
                X.append(feat)
                y.append(biome)
            except Exception as e:
                print(f"     Error en {img_path.name}: {e}")

    return np.array(X), np.array(y)


# ──────────────────────────────────────────────
# 3. MODELO — MLP
# ──────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    """
    Pipeline: Normalización → MLP.
    Arquitectura: dos capas ocultas [256, 128]
    """
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        max_iter=500,
        learning_rate_init=1e-3,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        verbose=False
    )
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp",    mlp)
    ])


# ──────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ──────────────────────────────────────────────
def run_cross_validation(X, y_encoded, class_names):
    """
    Ejecuta StratifiedKFold cross-validation y devuelve métricas.
    """
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    pipeline = build_pipeline()

    scoring = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    print(f"\n⏳ Ejecutando {N_SPLITS}-fold cross-validation …")

    results = cross_validate(
        pipeline, X, y_encoded,
        cv=cv, scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )

    print("\n📊 RESULTADOS POR FOLD")
    print(f"{'Fold':>5} | {'Train Acc':>10} | {'Val Acc':>10} | {'F1 Macro':>10}")
    print("-" * 45)
    for i in range(N_SPLITS):
        print(f"  {i+1:>3} | "
              f"{results['train_accuracy'][i]*100:>9.2f}% | "
              f"{results['test_accuracy'][i]*100:>9.2f}% | "
              f"{results['test_f1_macro'][i]*100:>9.2f}%")

    print("\n📈 RESUMEN")
    print(f"  Accuracy (val)  : {results['test_accuracy'].mean()*100:.2f}% "
          f"± {results['test_accuracy'].std()*100:.2f}%")
    print(f"  F1 Macro (val)  : {results['test_f1_macro'].mean()*100:.2f}% "
          f"± {results['test_f1_macro'].std()*100:.2f}%")
    print(f"  Precision Macro : {results['test_precision_macro'].mean()*100:.2f}%")
    print(f"  Recall Macro    : {results['test_recall_macro'].mean()*100:.2f}%")

    return results


# ──────────────────────────────────────────────
# 5. ENTRENAMIENTO FINAL + REPORTE
# ──────────────────────────────────────────────
def train_and_report(X, y_encoded, class_names):
    """
    Entrena con todo el conjunto y genera reporte + matriz de confusión.
    """
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_encoded, test_size=0.2,
        stratify=y_encoded, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    print("\n📋 CLASSIFICATION REPORT (train=80%, test=20%)")
    print(classification_report(y_te, y_pred, target_names=class_names))

    return y_te, y_pred, pipeline


# ──────────────────────────────────────────────
# 6. VISUALIZACIONES
# ──────────────────────────────────────────────
def plot_results(cv_results, y_te, y_pred, class_names):
    fig = plt.figure(figsize=(16, 6), facecolor="#0d1117")
    gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    dark_bg   = "#0d1117"
    panel_bg  = "#161b22"
    accent    = "#58a6ff"
    green     = "#3fb950"
    text_col  = "#e6edf3"
    grid_col  = "#21262d"

    # — Panel izquierdo: Accuracy por fold —
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(panel_bg)
    folds       = np.arange(1, N_SPLITS + 1)
    train_accs  = cv_results["train_accuracy"] * 100
    val_accs    = cv_results["test_accuracy"]  * 100

    ax1.bar(folds - 0.2, train_accs, width=0.35,
            color=accent, alpha=0.85, label="Train Accuracy")
    ax1.bar(folds + 0.2, val_accs,   width=0.35,
            color=green, alpha=0.85, label="Val Accuracy")
    ax1.axhline(val_accs.mean(), color=green, lw=1.5,
                ls="--", alpha=0.7, label=f"Media val: {val_accs.mean():.1f}%")

    ax1.set_xticks(folds)
    ax1.set_xticklabels([f"Fold {i}" for i in folds],
                        color=text_col, fontsize=9)
    ax1.set_ylabel("Accuracy (%)", color=text_col)
    ax1.set_ylim(0, 105)
    ax1.set_title("Accuracy por Fold — Cross-Validation",
                  color=text_col, fontsize=11, pad=12)
    ax1.tick_params(colors=text_col)
    ax1.spines[:].set_color(grid_col)
    ax1.yaxis.label.set_color(text_col)
    ax1.grid(axis="y", color=grid_col, lw=0.7)
    ax1.legend(facecolor=panel_bg, labelcolor=text_col, fontsize=8)

    # — Panel derecho: Matriz de confusión —
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(panel_bg)
    cm = confusion_matrix(y_te, y_pred)
    im = ax2.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im, ax=ax2).ax.yaxis.set_tick_params(color=text_col)

    ticks = np.arange(len(class_names))
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(class_names, rotation=40, ha="right",
                        color=text_col, fontsize=8)
    ax2.set_yticklabels(class_names, color=text_col, fontsize=8)
    ax2.set_xlabel("Predicho",  color=text_col)
    ax2.set_ylabel("Real",      color=text_col)
    ax2.set_title("Matriz de Confusión — MLP",
                  color=text_col, fontsize=11, pad=12)
    ax2.tick_params(colors=text_col)
    ax2.spines[:].set_color(grid_col)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax2.text(j, i, cm[i, j],
                     ha="center", va="center", fontsize=9,
                     color="white" if cm[i, j] > thresh else "#e6edf3")

    fig.suptitle("Ejercicio 1 — MLP + Histogramas de Color | Biomas de México",
                 color=text_col, fontsize=13, y=1.02, fontweight="bold")

    plt.savefig("Exercise1.png",
                dpi=150, bbox_inches="tight",
                facecolor=dark_bg)
    plt.show()
    print("\n✅ Gráfica guardada: Exercise1.png")


# ──────────────────────────────────────────────
# 7. MAIN
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  EJERCICIO 1 — MLP + Histogramas de Color")
    print("=" * 55)

    # — Verificar dataset —
    if not Path(DATASET_DIR).exists():
        print(f"\n❌ No se encontró la carpeta '{DATASET_DIR}'.")
        print("   Estructura esperada:")
        print(f"     ../Biomas/<NombreBioma>/*.jpg")
        print("   Ejecuta este script desde dentro de la carpeta Exercise1/")
        exit(1)

    if not BIOMES:
        print(f"\n❌ La carpeta '{DATASET_DIR}' está vacía.")
        exit(1)

    print(f"\n📂 Carpeta dataset: {Path(DATASET_DIR).resolve()}")
    print(f"   Clases detectadas: {BIOMES}")

    # — Cargar datos —
    print(f"\n📁 Cargando imágenes …")
    X, y = load_dataset(DATASET_DIR)
    print(f"\n✅ Dataset cargado: {X.shape[0]} muestras, "
          f"{X.shape[1]} features (histograma {N_BINS} bins × 3 canales)")

    # — Encodear etiquetas —
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_
    print(f"   Clases: {list(class_names)}")

    # Distribución
    unique, counts = np.unique(y, return_counts=True)
    print("\n   Distribución de clases:")
    for cls, cnt in zip(unique, counts):
        print(f"     {cls:12s}: {cnt} imágenes")

    # — Cross-validation —
    cv_results = run_cross_validation(X, y_enc, class_names)

    # — Entrenamiento final y reporte —
    y_te, y_pred, pipeline = train_and_report(X, y_enc, class_names)

    # — Visualización —
    plot_results(cv_results, y_te, y_pred, class_names)