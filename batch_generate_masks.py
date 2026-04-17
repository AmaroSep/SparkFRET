"""
Genera mascaras TIF de sparkles para todas las imagenes.
CellProfiler carga estas mascaras como objetos pre-segmentados.

Uso: D:/Cellpose/venv/Scripts/python D:/Cellpose/batch_generate_masks.py
"""
import warnings
warnings.filterwarnings("ignore")
import sys, time
from pathlib import Path
import numpy as np
import tifffile

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

# ── GUI ───────────────────────────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

messagebox.showinfo(
    "Cellpose - Generar Mascaras",
    "Selecciona el modelo entrenado, la carpeta de imagenes y donde guardar las mascaras."
)

MODEL_PATH = filedialog.askopenfilename(
    title="Selecciona el modelo entrenado",
    initialdir="D:/Cellpose/models/models",
    filetypes=[("Modelo Cellpose", "*"), ("Todos", "*.*")]
)
if not MODEL_PATH:
    print("Cancelado.")
    sys.exit(0)

SOURCE_DIR = filedialog.askdirectory(
    title="Carpeta con imagenes originales (.tif)",
    initialdir=r"C:\Users\aamar\OneDrive - Baylor College of Medicine\CLR_Labs"
)
if not SOURCE_DIR:
    print("Cancelado.")
    sys.exit(0)
SOURCE_DIR = Path(SOURCE_DIR)

OUT_DIR = filedialog.askdirectory(
    title="Carpeta donde guardar las mascaras",
    initialdir="D:/Cellpose/results"
)
if not OUT_DIR:
    print("Cancelado.")
    sys.exit(0)
OUT_DIR = Path(OUT_DIR)
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNEL_FILTER = simpledialog.askstring(
    "Filtro de canal",
    "Texto que debe contener el nombre de imagen\n(deja vacio para usar todas):",
    initialvalue="CFP YFP FRET"
)

CELLPROB = simpledialog.askfloat(
    "Cell probability threshold",
    "Umbral de probabilidad de celda\n(menor = detecta mas, mayor = mas estricto):",
    initialvalue=0.0, minvalue=-6.0, maxvalue=6.0
)
if CELLPROB is None:
    CELLPROB = 0.0

FLOW = simpledialog.askfloat(
    "Flow threshold",
    "Umbral de flujo\n(0.4 recomendado):",
    initialvalue=0.4, minvalue=0.0, maxvalue=3.0
)
if FLOW is None:
    FLOW = 0.4

root.destroy()

# ── Busca imagenes ────────────────────────────────────────────────────────────
all_tifs = sorted(SOURCE_DIR.glob("**/*.tif"))
if CHANNEL_FILTER:
    tif_files = [f for f in all_tifs if CHANNEL_FILTER in f.name]
else:
    tif_files = all_tifs

print(f"\nModelo:   {Path(MODEL_PATH).name}")
print(f"Fuente:   {SOURCE_DIR}")
print(f"Salida:   {OUT_DIR}")
print(f"Canal:    '{CHANNEL_FILTER}'" if CHANNEL_FILTER else "Canal:    todos")
print(f"Imagenes: {len(tif_files)} encontradas")
print(f"Cellprob: {CELLPROB}  |  Flow: {FLOW}\n")

if len(tif_files) == 0:
    print("ERROR: No se encontraron imagenes.")
    sys.exit(1)

# ── Carga modelo ──────────────────────────────────────────────────────────────
from cellpose import models
print("Cargando modelo...")
model = models.CellposeModel(gpu=True, pretrained_model=MODEL_PATH)
print("Modelo listo.\n")

# ── Procesa imagenes ──────────────────────────────────────────────────────────
t_start = time.time()
total_sparkles = 0

for i, src in enumerate(tif_files, 1):
    img = tifffile.imread(str(src))

    masks, flows, styles = model.eval(
        img,
        diameter=0,
        cellprob_threshold=CELLPROB,
        flow_threshold=FLOW,
        normalize={"normalize": True, "percentile": [1, 99.9]},
    )

    n = int(masks.max())
    total_sparkles += n

    out_path = OUT_DIR / (src.stem + "_mask.tif")
    tifffile.imwrite(str(out_path), masks.astype(np.uint16))

    elapsed = time.time() - t_start
    eta = elapsed / i * (len(tif_files) - i)
    print(f"  [{i:4d}/{len(tif_files)}] {src.name[:55]:<55} "
          f"sparkles={n:3d}  eta={eta/60:.1f}min", flush=True)

total_time = time.time() - t_start
print(f"\nCompletado en {total_time/60:.1f} minutos")
print(f"Total sparkles: {total_sparkles}")
print(f"Promedio/imagen: {total_sparkles/len(tif_files):.1f}")
print(f"Mascaras guardadas en: {OUT_DIR}")
