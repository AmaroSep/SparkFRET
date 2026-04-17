"""
Entrenamiento de modelo SparkFRET desde CPSAM (base).

Configuracion:
  - Base: CPSAM original (no fine-tune encadenado)
  - Normalizacion: percentile [1, 60]
  - weight_decay=0.01, lr=1e-5, split aleatorio, checkpoint cada 25 epochs

Uso: python train_model.py
"""
import logging, time, sys, random
import numpy as np
from pathlib import Path
from cellpose import models, train, io

# ── Configuracion ─────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
TRAIN_DIR      = BASE_DIR / "training_data" / "raw"
MODEL_SAVE_DIR = BASE_DIR / "models"
MODEL_NAME     = "sparkle_fret_v1"
CHANNEL_FILTER = ""   # vacío = aceptar cualquier nombre de archivo
N_EPOCHS       = 500

PRETRAINED_MODEL = None  # CPSAM base — borrón y cuenta nueva

# LR estandar para entrenamiento desde CPSAM
LEARNING_RATE = 1e-5

# MEJORA 2: semilla para split aleatorio reproducible
VAL_SEED = 42

# ── Logging con progreso ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
cp_logger = logging.getLogger("cellpose.train")
cp_logger.setLevel(logging.INFO)
for name in ["cellpose.models", "cellpose.dynamics", "cellpose.io"]:
    logging.getLogger(name).setLevel(logging.WARNING)

class ProgressHandler(logging.Handler):
    def __init__(self, n_epochs):
        super().__init__()
        self.n_epochs   = n_epochs
        self.start_time = time.time()

    def emit(self, record):
        msg = record.getMessage()
        if "train_loss" in msg:
            try:
                parts      = msg.split(",")
                iepoch     = int(parts[0].strip()) + 1
                train_loss = float(parts[1].split("=")[1].strip())
                elapsed    = time.time() - self.start_time
                eta        = elapsed / iepoch * (self.n_epochs - iepoch)
                pct        = iepoch / self.n_epochs * 100
                print(
                    f"  epoch {iepoch:>4}/{self.n_epochs}  "
                    f"[{pct:5.1f}%]  loss={train_loss:.4f}  "
                    f"elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min",
                    flush=True
                )
            except Exception:
                pass

cp_logger.addHandler(ProgressHandler(N_EPOCHS))

# ── Carga imagenes (_seg.npy) ─────────────────────────────────────────────────
seg_files = sorted(TRAIN_DIR.glob("*_seg.npy"))

print(f"\nCarpeta:         {TRAIN_DIR}")
print(f"Canal:           {'todos (sin filtro)' if not CHANNEL_FILTER else CHANNEL_FILTER}")
print(f"Modelo base:     {PRETRAINED_MODEL or 'CPSAM (base)'}")
print(f"Modelo salida:   {MODEL_NAME}")
print(f"Learning rate:   {LEARNING_RATE}")
print(f"Epochs:          {N_EPOCHS}")
print(f"Mascaras npy:    {len(seg_files)} encontradas\n")

train_images, train_labels = [], []
skipped = 0
missing_tif = []

for seg_path in seg_files:
    img_name = seg_path.name.replace("_seg.npy", ".tif")
    img_path = TRAIN_DIR / img_name
    # MEJORA 3: reporte claro de seg sin imagen
    if not img_path.exists():
        missing_tif.append(seg_path.name)
        skipped += 1
        continue
    img      = io.imread(str(img_path))
    seg_data = np.load(str(seg_path), allow_pickle=True).item()
    masks    = seg_data["masks"]
    if masks.max() < 1:
        print(f"  SKIP (sin mascaras): {img_name}")
        skipped += 1
        continue
    print(f"  {img_name}: {masks.max()} sparkles")
    train_images.append(img)
    train_labels.append(masks)

if missing_tif:
    print(f"\n  ADVERTENCIA: {len(missing_tif)} _seg.npy sin .tif correspondiente:")
    for f in missing_tif:
        print(f"    - {f}")

total = len(train_images)
print(f"\nTotal: {total} imagenes | Omitidas: {skipped}")
if total < 2:
    print("ERROR: necesitas al menos 2 imagenes.")
    sys.exit(1)

# MEJORA 2: split aleatorio en vez de ultimas N imagenes
n_val = max(1, total // 10)
rng   = random.Random(VAL_SEED)
indices = list(range(total))
rng.shuffle(indices)
val_idx   = set(indices[:n_val])
train_idx = [i for i in range(total) if i not in val_idx]

val_images   = [train_images[i] for i in sorted(val_idx)]
val_labels   = [train_labels[i] for i in sorted(val_idx)]
train_images = [train_images[i] for i in train_idx]
train_labels = [train_labels[i] for i in train_idx]
print(f"Train: {len(train_images)} | Val: {len(val_images)} (split aleatorio, seed={VAL_SEED})\n")

# ── Entrenamiento ─────────────────────────────────────────────────────────────
if PRETRAINED_MODEL:
    print(f"Cargando modelo propio: {PRETRAINED_MODEL}")
    model = models.CellposeModel(gpu=True, pretrained_model=PRETRAINED_MODEL)
else:
    print("Cargando modelo base (CPSAM)...")
    model = models.CellposeModel(gpu=True)

print(f"Entrenando {N_EPOCHS} epochs...\n")
t0 = time.time()

model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=train_images,
    train_labels=train_labels,
    test_data=val_images,
    test_labels=val_labels,
    normalize={"normalize": True, "percentile": [1, 60]},
    n_epochs=N_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    batch_size=1,
    save_path=str(MODEL_SAVE_DIR),
    model_name=MODEL_NAME,
    save_every=25,
    min_train_masks=1,
)

total_time = time.time() - t0
print(f"\nEntrenamiento completado en {total_time/60:.1f} minutos")
print(f"Modelo guardado en: {model_path}")

valid = train_losses[train_losses > 0]
if len(valid) >= 2:
    print(f"\nLoss inicial : {valid[0]:.4f}")
    print(f"Loss final   : {valid[-1]:.4f}")
    print(f"Mejora       : {(1 - valid[-1]/valid[0])*100:.1f}%")

# Mostrar curva de loss resumida
if len(valid) > 10:
    step = max(1, len(valid) // 10)
    print(f"\nCurva de loss (cada {step} epochs):")
    for i in range(0, len(valid), step):
        print(f"  epoch {i+1:>4}: {valid[i]:.4f}")
    print(f"  epoch {len(valid):>4}: {valid[-1]:.4f}")
