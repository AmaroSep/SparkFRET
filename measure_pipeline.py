"""
measure_pipeline.py — Medicion sobre mascaras pre-generadas.

Carga mascaras de sparkles (*_mask.tif) y celulas (*_cells.tif) producidas
por el modelo Cellpose de la Columna 1, mide intensidades FRET y morfologia,
y exporta Sparks.csv / Cells.csv / Image.csv.

Uso:
  python measure_pipeline.py \
    --fret_dir      D:/data/fret \
    --sparkle_mask_dir D:/data/masks \
    --cell_mask_dir    D:/data/masks_cells \
    --out_dir          D:/Cellpose/results/analysis
"""

import re, argparse, warnings
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from skimage import measure

warnings.filterwarnings("ignore")

# ── Regex metadatos del nombre de archivo ──────────────────────────────────────
RE_WELL = re.compile(r"^([A-H]\d{1,2})ROI(\d+)_", re.IGNORECASE)

def extract_meta(fname):
    m = RE_WELL.match(fname)
    if m:
        return m.group(1), m.group(2), f"{m.group(1)}ROI{m.group(2)}"
    return None, None, None


# ── Debug overlay ─────────────────────────────────────────────────────────────
def _save_debug_overlay(bg_img, cells_labeled, sparks_labeled, out_dir, roi):
    """Guarda PNG con 4 paneles: FRET | Células | Sparkles | Merge."""
    debug_dir = Path(out_dir) / "debug"
    debug_dir.mkdir(exist_ok=True)

    vmin = np.percentile(bg_img, 1)
    vmax = np.percentile(bg_img, 99)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[0].set_title("FRET original", fontsize=9)
    axes[0].axis("off")

    axes[1].imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[1].imshow(np.ma.masked_where(cells_labeled == 0, cells_labeled),
                   cmap="Greens", alpha=0.55)
    axes[1].set_title(f"Células (n={cells_labeled.max()})", fontsize=9)
    axes[1].axis("off")

    axes[2].imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[2].imshow(np.ma.masked_where(sparks_labeled == 0, sparks_labeled),
                   cmap="Reds", alpha=0.7)
    axes[2].set_title(f"Sparkles (n={int(sparks_labeled.max())})", fontsize=9)
    axes[2].axis("off")

    axes[3].imshow(bg_img, cmap="gray", vmin=vmin, vmax=vmax)
    axes[3].imshow(np.ma.masked_where(cells_labeled == 0, cells_labeled),
                   cmap="Greens", alpha=0.4)
    axes[3].imshow(np.ma.masked_where(sparks_labeled == 0, sparks_labeled),
                   cmap="Reds", alpha=0.7)
    axes[3].set_title("Merge células + sparkles", fontsize=9)
    axes[3].axis("off")

    plt.suptitle(roi, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(debug_dir / f"{roi}.png"), dpi=100, bbox_inches="tight")
    plt.close()


# ── Medicion de forma ─────────────────────────────────────────────────────────
SHAPE_PROPS = [
    "label", "area", "equivalent_diameter_area", "eccentricity",
    "solidity", "major_axis_length", "minor_axis_length",
    "perimeter", "centroid"
]

def measure_shape(labeled):
    if labeled.max() == 0:
        return pd.DataFrame()
    rp = measure.regionprops_table(labeled, properties=SHAPE_PROPS)
    df = pd.DataFrame(rp)
    df.rename(columns={
        "label":                    "ObjectNumber",
        "area":                     "AreaShape_Area",
        "equivalent_diameter_area": "AreaShape_EquivalentDiameter",
        "eccentricity":             "AreaShape_Eccentricity",
        "solidity":                 "AreaShape_Solidity",
        "major_axis_length":        "AreaShape_MajorAxisLength",
        "minor_axis_length":        "AreaShape_MinorAxisLength",
        "centroid-0":               "centroid_row",
        "centroid-1":               "centroid_col",
    }, inplace=True)
    df["AreaShape_FormFactor"] = (
        4 * np.pi * df["AreaShape_Area"] /
        (df["perimeter"].replace(0, np.nan) ** 2)
    ).fillna(0)
    df["AreaShape_Compactness"] = (
        1 / df["AreaShape_FormFactor"].replace(0, np.nan)
    ).fillna(0)
    df.drop(columns=["perimeter"], inplace=True)
    return df


# ── Medicion de intensidad ────────────────────────────────────────────────────
def measure_intensity(labeled, intensity_img, channel_name):
    if labeled.max() == 0:
        return pd.DataFrame()
    img_f = intensity_img.astype(float)
    rp = measure.regionprops_table(
        labeled, intensity_image=img_f,
        properties=["label", "intensity_mean", "intensity_max"]
    )
    df = pd.DataFrame(rp)
    df.rename(columns={"label": "ObjectNumber"}, inplace=True)
    shape_df = pd.DataFrame(
        measure.regionprops_table(labeled, properties=["label", "area"])
    ).rename(columns={"label": "ObjectNumber"})
    df = df.merge(shape_df, on="ObjectNumber")
    df[f"Intensity_MeanIntensity_{channel_name}"]       = df["intensity_mean"]
    df[f"Intensity_MaxIntensity_{channel_name}"]        = df["intensity_max"]
    df[f"Intensity_IntegratedIntensity_{channel_name}"] = df["intensity_mean"] * df["area"]
    df.drop(columns=["intensity_mean", "intensity_max", "area"], inplace=True)
    return df


# ── RelateObjects: asigna Parent_Cells a cada Spark ───────────────────────────
def relate_sparks_to_cells(sparks_labeled, cells_labeled):
    """Para cada Spark, busca el label de Cells que se superpone más."""
    spark_ids = np.unique(sparks_labeled[sparks_labeled > 0])
    parents = {}
    for sid in spark_ids:
        mask = sparks_labeled == sid
        cells_under = cells_labeled[mask]
        cells_nz = cells_under[cells_under > 0]
        if len(cells_nz) == 0:
            parents[sid] = 0
        else:
            parents[sid] = int(np.bincount(cells_nz).argmax())
    return parents


# ── Normaliza imagen para lectura ─────────────────────────────────────────────
def read_tif(path):
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        vars_ = [img[c].var() for c in range(img.shape[0])]
        img = img[int(np.argmax(vars_))]
    return img


# ── Pipeline principal ────────────────────────────────────────────────────────
def run(fret_dir, sparkle_mask_dir, cell_mask_dir, out_dir, progress_cb=None):
    fret_dir         = Path(fret_dir)
    sparkle_mask_dir = Path(sparkle_mask_dir)
    cell_mask_dir    = Path(cell_mask_dir)
    out_dir          = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fret_files  = {extract_meta(f.name)[2]: f
                   for f in sorted(fret_dir.glob("*.tif"))
                   if "FRET" in f.name and "_mask" not in f.name}
    spark_files = {extract_meta(f.name)[2]: f
                   for f in sorted(sparkle_mask_dir.glob("*_mask*.tif"))}
    cell_files  = {extract_meta(f.name)[2]: f
                   for f in sorted(cell_mask_dir.glob("*_cells*.tif"))}

    rois = sorted(set(fret_files) & set(spark_files) & set(cell_files))
    print(f"ROIs encontrados: {len(rois)}  "
          f"(FRET: {len(fret_files)}, spark masks: {len(spark_files)}, cell masks: {len(cell_files)})")

    if not rois:
        print("AVISO: No hay ROIs con las tres fuentes disponibles. Verifica las carpetas.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    all_sparks = []
    all_cells  = []
    all_images = []
    img_num    = 1

    for roi in rois:
        well, roi_num, _ = extract_meta(fret_files[roi].name)
        print(f"  [{img_num:>4}] {roi} ...", end=" ", flush=True)
        if progress_cb:
            progress_cb(img_num, len(rois), roi)

        fret_img       = read_tif(fret_files[roi])
        sparks_labeled = tifffile.imread(str(spark_files[roi])).astype(np.uint16)
        cells_labeled  = tifffile.imread(str(cell_files[roi])).astype(np.uint16)

        if cells_labeled.shape != sparks_labeled.shape:
            from skimage.transform import resize as sk_resize
            cells_labeled = sk_resize(
                cells_labeled, sparks_labeled.shape,
                order=0, preserve_range=True, anti_aliasing=False
            ).astype(np.uint16)

        n_cells  = int(cells_labeled.max())
        n_sparks = int(sparks_labeled.max())

        _save_debug_overlay(fret_img, cells_labeled, sparks_labeled, out_dir, roi)

        s_shape = measure_shape(sparks_labeled)
        s_int   = measure_intensity(sparks_labeled, fret_img, "FRET")
        if s_shape.empty or s_int.empty:
            sparks_df = pd.DataFrame()
        else:
            sparks_df = s_shape.merge(s_int, on="ObjectNumber", how="outer")

        cells_df = measure_shape(cells_labeled)

        if not sparks_df.empty and n_cells > 0:
            parents = relate_sparks_to_cells(sparks_labeled, cells_labeled)
            sparks_df["Parent_Cells"] = sparks_df["ObjectNumber"].map(parents).fillna(0).astype(int)
        elif not sparks_df.empty:
            sparks_df["Parent_Cells"] = 0

        meta = {
            "ImageNumber":     img_num,
            "Metadata_Well":   well,
            "Metadata_ROI":    roi_num,
            "Metadata_ROI_ID": roi,
            "FileName_FRET":   fret_files[roi].name,
        }
        for df in [sparks_df, cells_df]:
            if not df.empty:
                for k, v in meta.items():
                    df.insert(0, k, v)

        all_sparks.append(sparks_df)
        all_cells.append(cells_df)

        sparks_in_cells = (0 if sparks_df.empty
                           else (sparks_df["Parent_Cells"] > 0).sum())
        all_images.append({
            **meta,
            "Count_Sparks":        n_sparks,
            "Count_Cells":         n_cells,
            "Count_SparksInCells": int(sparks_in_cells),
        })

        print(f"cells={n_cells}  sparks={n_sparks}  in_cells={sparks_in_cells}")
        img_num += 1

    spark_frames = [df for df in all_sparks if not df.empty]
    cell_frames  = [df for df in all_cells  if not df.empty]

    sparks_out = pd.concat(spark_frames, ignore_index=True) if spark_frames else pd.DataFrame()
    cells_out  = pd.concat(cell_frames,  ignore_index=True) if cell_frames  else pd.DataFrame()
    images_out = pd.DataFrame(all_images)

    sparks_out.to_csv(out_dir / "Sparks.csv",  index=False)
    cells_out.to_csv( out_dir / "Cells.csv",   index=False)
    images_out.to_csv(out_dir / "Image.csv",   index=False)

    with pd.ExcelWriter(out_dir / "summary.xlsx", engine="openpyxl") as xw:
        images_out.to_excel(xw, sheet_name="Image",  index=False)
        sparks_out.to_excel(xw, sheet_name="Sparks", index=False)
        cells_out.to_excel( xw, sheet_name="Cells",  index=False)

    print(f"\nGuardado en {out_dir}")
    print(f"  Sparks.csv : {len(sparks_out)} filas")
    print(f"  Cells.csv  : {len(cells_out)} filas")
    print(f"  Image.csv  : {len(images_out)} filas")
    return sparks_out, cells_out, images_out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fret_dir",          required=True)
    p.add_argument("--sparkle_mask_dir",  required=True)
    p.add_argument("--cell_mask_dir",     required=True)
    p.add_argument("--out_dir",           required=True)
    args = p.parse_args()
    run(args.fret_dir, args.sparkle_mask_dir, args.cell_mask_dir, args.out_dir)
