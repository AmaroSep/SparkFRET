"""
Analisis de sparkles FRET biosensor.
Combina output de CellProfiler con diseno experimental de placa.

Logica identica al script R original:
  %PositiveFRET   = (sparkles_en_celulas / total_celulas) x 100
  mean_intensity  = IntegratedIntensity_FRET promedio por imagen
  Integrated_FRET = %PositiveFRET x mean_intensity

Uso: D:/Cellpose/venv/Scripts/python D:/Cellpose/analyze_fret.py
"""

import sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from scipy import stats

import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox

BASE_DIR = Path(__file__).parent

# ── GUI con tkinter (Windows) ─────────────────────────────────────────────────
root = tk.Tk()
root.withdraw()

messagebox.showinfo(
    "Analisis FRET — SparkFRET",
    "A continuacion selecciona:\n"
    "  1. Sparks.csv\n"
    "  2. Cells.csv  (opcional)\n"
    "  3. Image.csv\n"
    "  4. Plate info (.xlsx)\n"
    "  5. Columnas de agrupacion\n"
    "  6. Carpeta de salida"
)

default_cp = str(BASE_DIR / "results")

sparks_path = filedialog.askopenfilename(
    title="1/6 — Sparks.csv (CellProfiler output)",
    initialdir=default_cp,
    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
)
if not sparks_path:
    print("Cancelado."); sys.exit(0)

cells_path = filedialog.askopenfilename(
    title="2/6 — Cells.csv (opcional, cancela para omitir)",
    initialdir=str(Path(sparks_path).parent),
    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
)

images_path = filedialog.askopenfilename(
    title="3/6 — Image.csv (CellProfiler output)",
    initialdir=str(Path(sparks_path).parent),
    filetypes=[("CSV", "*.csv"), ("Todos", "*.*")]
)
if not images_path:
    print("Cancelado."); sys.exit(0)

plate_path = filedialog.askopenfilename(
    title="4/6 — Plate info (.xlsx)",
    initialdir=str(Path.home()),
    filetypes=[("Excel", "*.xlsx *.xls"), ("Todos", "*.*")]
)
if not plate_path:
    print("Cancelado."); sys.exit(0)

cols_str = simpledialog.askstring(
    "Columnas de agrupacion",
    "5/6 — Columnas del layout de placa\n"
    "(en orden, separadas por coma)\n"
    "1a columna = eje X de la grafica\n"
    "2a columna = paneles",
    initialvalue="Disease, Fraction, Sample, Replicate"
)
if not cols_str:
    print("Cancelado."); sys.exit(0)
GROUP_COLS = [c.strip() for c in cols_str.split(",") if c.strip()]

out_dir = filedialog.askdirectory(
    title="6/6 — Carpeta donde guardar los resultados",
    initialdir=str(BASE_DIR / "results")
)
if not out_dir:
    print("Cancelado."); sys.exit(0)

root.destroy()

OUT = Path(out_dir)
OUT.mkdir(parents=True, exist_ok=True)

print(f"\nColumnas de agrupacion: {GROUP_COLS}")

# ── Carga datos ───────────────────────────────────────────────────────────────
print("Cargando datos...")
sparks = pd.read_csv(sparks_path)
images = pd.read_csv(images_path)
plate  = pd.read_excel(plate_path)

print(f"  Sparkles:  {len(sparks)}")
print(f"  Imagenes:  {len(images)}")
print(f"  Pocillos:  {len(plate)}")

if cells_path:
    cells = pd.read_csv(cells_path)
    print(f"  Celulas:   {len(cells)}")

# Verifica columnas del layout
missing_cols = [c for c in GROUP_COLS if c not in plate.columns]
if missing_cols:
    print(f"\nAVISO: columnas no encontradas en Plate info: {missing_cols}")
    print(f"  Columnas disponibles: {list(plate.columns)}")
    GROUP_COLS = [c for c in GROUP_COLS if c in plate.columns]
    print(f"  Usando: {GROUP_COLS}")

# ── Detecta columnas clave automaticamente ────────────────────────────────────
def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

COL_PARENT   = find_col(sparks, ["Parent_Cells", "Parent_Cells_YFP_background", "Parent_Sparks"])
COL_INT_FRET = find_col(sparks, ["Intensity_IntegratedIntensity_FRET",
                                  "Intensity_IntegratedIntensity_FRET_V2"])
COL_N_CELLS  = find_col(images,  ["Count_Cells", "Count_Cells_YFP_background", "Count_Sparks"])
COL_WELL     = find_col(images,  ["Metadata_Well"])

if not COL_PARENT:
    print("ERROR: No se encontro columna Parent_Cells en Sparks.csv")
    print(f"  Columnas disponibles: {list(sparks.columns)}")
    sys.exit(1)
if not COL_INT_FRET:
    print("ERROR: No se encontro columna Intensity_IntegratedIntensity_FRET")
    print(f"  Columnas disponibles: {list(sparks.columns)}")
    sys.exit(1)
if not COL_N_CELLS:
    print("ERROR: No se encontro columna Count_Cells en Image.csv")
    print(f"  Columnas disponibles: {list(images.columns)}")
    sys.exit(1)

if not COL_WELL:
    def extract_well(filename):
        m = re.match(r"^([A-H]\d{1,2})ROI", str(filename))
        return m.group(1) if m else None
    fn_col = find_col(images, ["FileName_FRET", "FileName_CFP", "FileName_SparkMask"])
    if fn_col:
        images["Metadata_Well"] = images[fn_col].apply(extract_well)
        COL_WELL = "Metadata_Well"
    else:
        print("ERROR: No se puede determinar el Well desde Image.csv")
        print(f"  Columnas disponibles: {list(images.columns)}")
        sys.exit(1)

print(f"\n  Columna parent:   {COL_PARENT}")
print(f"  Columna int FRET: {COL_INT_FRET}")
print(f"  Columna celulas:  {COL_N_CELLS}")
print(f"  Columna well:     {COL_WELL}")

# ── Logica identica al script R ───────────────────────────────────────────────
# 1. Solo sparkles dentro de una celula
sparks_in_cells = sparks[sparks[COL_PARENT] != 0].copy()
print(f"\nSparkles en celulas: {len(sparks_in_cells)} / {len(sparks)} total")

# 2. Conteo de sparkles por imagen
sparks_per_image = (sparks_in_cells
    .groupby("ImageNumber").size()
    .reset_index(name="Sparks_in_cells"))

# 3. Intensidad media por imagen
mean_intensity = (sparks_in_cells
    .groupby("ImageNumber")[COL_INT_FRET].mean()
    .reset_index().rename(columns={COL_INT_FRET: "mean_intensity"}))

# 4. Tabla por imagen
img_result = (
    images[["ImageNumber", COL_WELL, COL_N_CELLS]]
    .rename(columns={COL_WELL: "Well", COL_N_CELLS: "Count_Cells"})
    .merge(sparks_per_image, on="ImageNumber", how="left")
    .merge(mean_intensity,   on="ImageNumber", how="left")
)
img_result["Sparks_in_cells"] = img_result["Sparks_in_cells"].fillna(0)
img_result["mean_intensity"]   = img_result["mean_intensity"].fillna(0)

# 5. Metricas derivadas
img_result["pct_positive_FRET"] = (
    img_result["Sparks_in_cells"] /
    img_result["Count_Cells"].replace(0, np.nan)
) * 100
img_result["Integrated_FRET"] = (
    img_result["pct_positive_FRET"] * img_result["mean_intensity"]
)
img_result[["pct_positive_FRET", "Integrated_FRET"]] = (
    img_result[["pct_positive_FRET", "Integrated_FRET"]].fillna(0))

# 6. Resumen por Well
well_result = (img_result.groupby("Well").agg(
    Count_Cells       = ("Count_Cells",       "mean"),
    mean_intensity    = ("mean_intensity",     "mean"),
    pct_positive_FRET = ("pct_positive_FRET",  "mean"),
    Integrated_FRET   = ("Integrated_FRET",    "mean"),
    n_images          = ("ImageNumber",         "count"),
).reset_index())

# 7. Une con layout de placa
result_layout = well_result.merge(plate, on="Well", how="left")

# 8. Resumen por grupo
avail_group = [c for c in GROUP_COLS if c in result_layout.columns]

result_mean = (
    result_layout.dropna(subset=[avail_group[0]] if avail_group else [])
    .groupby(avail_group).agg(
        Count_Cells       = ("Count_Cells",       "mean"),
        mean_intensity    = ("mean_intensity",     "mean"),
        pct_positive_FRET = ("pct_positive_FRET",  "mean"),
        Integrated_FRET   = ("Integrated_FRET",    "mean"),
        n_wells           = ("Well",               "count"),
    ).reset_index()
)

# ── Imprime resumen ───────────────────────────────────────────────────────────
pd.set_option("display.float_format", "{:.3f}".format)
print("\n=== RESUMEN POR GRUPO ===")
show_cols = avail_group + ["Count_Cells", "pct_positive_FRET",
                            "mean_intensity", "Integrated_FRET", "n_wells"]
print(result_mean[show_cols].to_string(index=False))

# ── Guarda CSVs ───────────────────────────────────────────────────────────────
img_result.to_csv(   OUT / "image_result.csv",  index=False)
result_layout.to_csv(OUT / "well_result.csv",   index=False)
result_mean.to_csv(  OUT / "group_summary.csv", index=False)
print(f"\nCSVs guardados en: {OUT}")

# ── Figura de publicacion ─────────────────────────────────────────────────────
X_COL   = avail_group[0] if avail_group else "Well"
FAC_COL = avail_group[1] if len(avail_group) > 1 else None

df_plot  = result_layout.dropna(subset=[X_COL])
x_vals   = sorted(df_plot[X_COL].dropna().unique().tolist())
fac_vals = (sorted(df_plot[FAC_COL].dropna().unique().tolist())
            if FAC_COL else ["Todos"])

BASE_PALETTE = ["#4CAF50", "#FF9800", "#F44336", "#2196F3",
                "#9C27B0", "#00BCD4", "#FF5722", "#607D8B"]
COLORS = {v: BASE_PALETTE[i % len(BASE_PALETTE)] for i, v in enumerate(x_vals)}

METRICS = [
    ("Count_Cells",       "Celulas / imagen"),
    ("pct_positive_FRET", "% FRET positivo"),
    ("mean_intensity",    "Intensidad integrada media"),
    ("Integrated_FRET",   "Integrated FRET  (%pos x int)"),
]

ncols = len(fac_vals)
nrows = len(METRICS)
fig, axes = plt.subplots(nrows, ncols,
                         figsize=(5 * ncols, 4 * nrows),
                         squeeze=False)

exp_title = " · ".join(avail_group)
fig.suptitle(f"Biosensor FRET — {exp_title}",
             fontsize=14, fontweight="bold", y=1.01)

JITTER = 0.15
np.random.seed(42)

for row, (col, ylabel) in enumerate(METRICS):
    for c, fac in enumerate(fac_vals):
        ax = axes[row, c]

        if FAC_COL:
            data = df_plot[df_plot[FAC_COL] == fac]
        else:
            data = df_plot

        for xi, xv in enumerate(x_vals):
            vals = data[data[X_COL] == xv][col].dropna()
            if vals.empty:
                continue
            color = COLORS.get(xv, "gray")
            ax.bar(xi, vals.mean(), width=0.55,
                   color=color, alpha=0.6, zorder=2)
            ax.errorbar(xi, vals.mean(), yerr=vals.sem(),
                        fmt="none", color="black",
                        capsize=6, linewidth=2, zorder=4)
            xj = np.random.uniform(-JITTER, JITTER, size=len(vals))
            ax.scatter(xi + xj, vals, color=color,
                       edgecolors="black", linewidths=0.5,
                       s=45, zorder=5, alpha=0.9)

        # Significancia (t-test por pares)
        y_max  = data[col].max() if not data[col].empty else 1
        offset = y_max * 0.08
        pairs  = [(i, j) for i in range(len(x_vals))
                  for j in range(i + 1, len(x_vals))]
        for (i, j) in pairs:
            v1 = data[data[X_COL] == x_vals[i]][col].dropna()
            v2 = data[data[X_COL] == x_vals[j]][col].dropna()
            if len(v1) < 2 or len(v2) < 2:
                continue
            _, p = stats.ttest_ind(v1, v2)
            if p < 0.05:
                y_bar = max(v1.max(), v2.max()) + offset
                ax.plot([i, i, j, j],
                        [y_bar, y_bar + offset * 0.3,
                         y_bar + offset * 0.3, y_bar],
                        color="black", linewidth=1)
                sig = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                ax.text((i + j) / 2, y_bar + offset * 0.4, sig,
                        ha="center", va="bottom", fontsize=9)

        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels(x_vals, fontsize=10, rotation=15, ha="right")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(str(fac), fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.6, len(x_vals) - 0.4)
        ax.set_ylim(bottom=0)

patches = [mpatches.Patch(color=COLORS[v], label=str(v))
           for v in x_vals if v in COLORS]
fig.legend(handles=patches, loc="lower center",
           ncol=min(len(x_vals), 5), fontsize=10,
           bbox_to_anchor=(0.5, -0.02), frameon=False)

plt.tight_layout()
plt.savefig(OUT / "fret_figure.png", dpi=200, bbox_inches="tight")
plt.savefig(OUT / "fret_figure.svg",          bbox_inches="tight")
print("  fret_figure.png / .svg")
print(f"\nListo.")
