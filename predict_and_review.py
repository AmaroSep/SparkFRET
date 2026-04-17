"""
Corre el modelo en imagenes nuevas, guarda predicciones como _seg.npy
para revision y correccion manual en la GUI de Cellpose.

Uso: D:/Cellpose/venv/Scripts/python D:/Cellpose/predict_and_review.py
"""
from pathlib import Path
import numpy as np
import tifffile
from cellpose import models, io

# ── Configuracion ─────────────────────────────────────────────────────────────
MODEL_PATH   = "D:/Cellpose/models/models/sparkle_fret_v1"
SOURCE_DIR   = Path(r"C:\Users\aamar\OneDrive - Baylor College of Medicine\CLR_Labs\Book\LabBook\Results\Cytation\260413_041955_Nur samples F9 and sark\260413_043012_Plate 1")
TRAIN_DIR    = Path("D:/Cellpose/training_data/raw")
N_IMAGES     = 10   # cuantas imagenes nuevas procesar

# ── Busca imagenes que NO esten ya en el dataset ──────────────────────────────
already_in_dataset = {f.name.replace("_seg.npy", ".tif")
                      for f in TRAIN_DIR.glob("*_seg.npy")}

candidates = [f for f in sorted(SOURCE_DIR.glob("**/*.tif"))
              if f.name not in already_in_dataset]

print(f"Imagenes disponibles nuevas: {len(candidates)}")
print(f"Seleccionando las primeras {N_IMAGES}...\n")

selected = candidates[:N_IMAGES]

# ── Carga modelo ──────────────────────────────────────────────────────────────
print("Cargando modelo entrenado...")
model = models.CellposeModel(gpu=True, pretrained_model=MODEL_PATH)

# ── Inferencia y guardado ─────────────────────────────────────────────────────
print(f"Procesando {len(selected)} imagenes...\n")

for i, src_path in enumerate(selected, 1):
    dst_img  = TRAIN_DIR / src_path.name
    dst_seg  = TRAIN_DIR / src_path.name.replace(".tif", "_seg.npy")

    # Copia imagen al training_dir si no existe
    if not dst_img.exists():
        import shutil
        shutil.copy2(src_path, dst_img)

    # Carga y predice
    img = tifffile.imread(str(dst_img))
    masks, flows, styles = model.eval(
        img,
        diameter=0,
        cellprob_threshold=0.0,
        flow_threshold=0.4,
        normalize={"normalize": True, "percentile": [1, 99.9]},
    )

    n_detected = int(masks.max())

    # Guarda en formato _seg.npy compatible con GUI de Cellpose
    seg_data = {
        "masks": masks,
        "flows": flows,
        "filename": str(dst_img),
        "img": img,
    }
    np.save(str(dst_seg), seg_data, allow_pickle=True)

    print(f"  [{i:2d}/{len(selected)}] {src_path.name}")
    print(f"           Sparkles detectados: {n_detected}  -> guardado en training/raw/")

print("\nLISTO. Ahora en Cellpose GUI:")
print("  1. Abre cada imagen de D:/Cellpose/training_data/raw/")
print("  2. Las mascaras predichas se cargan automaticamente")
print("  3. Corrige errores:")
print("       - Borra mascaras malas:  Ctrl + clic en la mascara")
print("       - Agrega sparkles faltantes: right-click + arrastra")
print("  4. Guarda correccion: Ctrl+S")
print("  5. Cuando termines todas -> python D:/Cellpose/train_sparkle.py")
