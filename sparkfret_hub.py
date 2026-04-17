"""
SparkFRET Hub — Unified pipeline
Column 1: Cellpose detection
Column 2: FRET analysis (CellProfiler CSVs + Plate info)
Column 3: Characterization and patterns

Usage: D:/Cellpose/venv/Scripts/streamlit run D:/Cellpose/sparkfret_hub.py
"""

import subprocess, re, warnings, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import streamlit as st
from pathlib import Path
from scipy import stats
from skimage.measure import regionprops_table
import tifffile

warnings.filterwarnings("ignore")

# ── Browsers nativos (tkinter local) ─────────────────────────────────────────
import tkinter as tk
from tkinter import filedialog as tkfd

def _browse_folder(state_key, title):
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
    path = tkfd.askdirectory(title=title)
    root.destroy()
    if path:
        st.session_state[state_key] = path

def _browse_file(state_key, title, filetypes=None):
    root = tk.Tk(); root.withdraw(); root.wm_attributes("-topmost", 1)
    path = tkfd.askopenfilename(title=title,
                                filetypes=filetypes or [("All","*.*")])
    root.destroy()
    if path:
        st.session_state[state_key] = path

def path_input(label, state_key, default="", is_folder=False,
               filetypes=None, title=None, browse=True):
    """Text input + … button that opens the native Windows file browser."""
    if state_key not in st.session_state:
        st.session_state[state_key] = default
    if browse:
        c1, c2 = st.columns([5, 1])
    else:
        c1 = st.container()
    with c1:
        # Usar state_key directamente — Streamlit sincroniza
        # session_state[state_key] con el widget automaticamente
        val = st.text_input(label, key=state_key)
    if browse:
        with c2:
            st.write(""); st.write("")  # alinea con el input
            if is_folder:
                st.button("…", key=f"_btn_{state_key}",
                          on_click=_browse_folder,
                          args=(state_key, title or label))
            else:
                st.button("…", key=f"_btn_{state_key}",
                          on_click=_browse_file,
                          args=(state_key, title or label, filetypes))
    return val

# ── Config pagina ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SparkFRET Hub",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    h1 { font-size: 1.6rem !important; }
    h2 { font-size: 1.1rem !important; border-bottom: 2px solid #444; padding-bottom: 4px; }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)

st.title("SparkFRET Hub")

DISEASE_ORDER  = ["Control", "AsymAD", "AD"]
DISEASE_COLORS = {"Control": "#4CAF50", "AsymAD": "#FF9800", "AD": "#F44336"}

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "masks_dir":           None,
    "detection_log":       [],
    "cell_detection_log":  [],
    "cell_masks_dir":      None,
    "sparkles_df":         None,
    "fret_summary":        None,
    "patterns_done":       False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading Cellpose model...")
def load_model(model_path):
    from cellpose import models
    return models.CellposeModel(gpu=True, pretrained_model=model_path)

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf

def extract_well(s):
    m = re.match(r"^([A-H]\d{1,2})ROI", str(s))
    return m.group(1) if m else None

def filter_outliers(df, features, method="IQR", threshold=1.5):
    """Remove rows where any feature is an outlier. Returns filtered df and count removed."""
    mask = pd.Series(True, index=df.index)
    for feat in features:
        col = df[feat].dropna()
        if col.empty:
            continue
        if method == "IQR":
            Q1, Q3 = col.quantile(0.25), col.quantile(0.75)
            iqr = Q3 - Q1
            mask &= df[feat].between(Q1 - threshold * iqr, Q3 + threshold * iqr)
        else:  # Z-score
            mu, sigma = col.mean(), col.std()
            if sigma > 0:
                mask &= ((df[feat] - mu) / sigma).abs() <= threshold
    removed = (~mask).sum()
    return df[mask].copy(), int(removed)


def _match_and_ratio(sparks_log, cells_log):
    """Match sparkle and cell detection logs by ROI stem, return (per_img list, overall float|None).

    Matching key: characters before the first underscore-separated channel suffix.
    e.g. 'A1ROI1_FRET.tif' and 'A1ROI1_V2.tif' both yield stem 'A1ROI1'.
    """
    def _stem(filename):
        # Take everything up to and including the ROI number, drop channel suffix
        m = re.match(r"^([A-Za-z0-9]+ROI\d+)", str(filename))
        return m.group(1) if m else Path(filename).stem

    # If two files share a stem (e.g. same ROI, different channel), last entry wins.
    # Expected usage: one sparkle file and one cell file per ROI.
    s_by_stem = {_stem(r["file"]): r for r in sparks_log}
    c_by_stem = {_stem(r["file"]): r for r in cells_log}

    per_img = []
    for stem, sr in s_by_stem.items():
        if stem not in c_by_stem:
            continue
        n_cells = c_by_stem[stem]["cells"]
        if n_cells == 0:
            continue
        ratio = sr["sparkles"] / n_cells
        per_img.append({"stem": stem, "well": sr.get("well"), "ratio": ratio,
                         "sparkles": sr["sparkles"], "cells": n_cells})

    if not per_img:
        return per_img, None

    total_s = sum(r["sparkles"] for r in per_img)
    total_c = sum(r["cells"]    for r in per_img)
    overall = total_s / total_c if total_c > 0 else None
    return per_img, overall


# ═════════════════════════════════════════════════════════════════════════════
col1, col2, col3 = st.columns(3, gap="medium")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COLUMNA 1 — DETECCIÓN                                                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with col1:
    st.header("1 · Cellpose Detection")

    img_dir = path_input(
        "FRET + Cell images folder", "img_dir",
        default="C:/Users/aamar/OneDrive - Baylor College of Medicine/CLR_Labs/Book/LabBook/Results/Cytation/260413_041955_Nur samples F9 and sark",
        is_folder=True
    )

    model_path = path_input(
        "Cellpose model", "model_path",
        default="D:/Cellpose/models/sparkle_fret_v9",
        filetypes=[("Modelo","*"), ("Todos","*.*")]
    )

    with st.expander("Inference parameters"):
        upper      = st.slider("upper percentile",  50, 99,  99,        key="upper")
        flow_thr   = st.slider("flow_threshold",    0.0, 1.0, 0.8, 0.05, key="flow_thr")
        cellprob   = st.slider("cellprob_threshold",-4.0, 2.0,-3.0, 0.1, key="cellprob")

    masks_out = path_input(
        "Output masks folder", "masks_out",
        default="D:/Cellpose/results/masks", is_folder=True
    )

    channel_filter = path_input(
        "Filename filter (empty = all)", "ch_filter",
        default="", browse=False
    )

    run_detect = st.button("▶  Detect sparkles", type="primary", key="btn_detect")

    if run_detect:
        img_path = Path(img_dir)
        out_path = Path(masks_out)
        out_path.mkdir(parents=True, exist_ok=True)

        all_tifs = sorted(img_path.glob("*.tif")) + sorted(img_path.glob("*.tiff"))
        fret_files = [f for f in all_tifs
                      if "FRET" in f.name and "_mask" not in f.name]
        if channel_filter:
            fret_files = [f for f in fret_files if channel_filter in f.name]

        if not fret_files:
            st.error(f"No FRET images found in: `{img_path}`")
            if all_tifs:
                st.info(f"TIF files found ({len(all_tifs)}): "
                        + ", ".join(f.name for f in all_tifs[:5])
                        + ("..." if len(all_tifs) > 5 else ""))
            else:
                st.warning("The folder contains no .tif or .tiff files")
        else:
            model = load_model(model_path)
            log = []
            bar = st.progress(0, text="Processing images...")

            for i, fret_path in enumerate(fret_files):
                img = tifffile.imread(str(fret_path)).astype(np.float32)
                masks, _, _ = model.eval(
                    img,
                    channels=[0, 0],
                    flow_threshold=flow_thr,
                    cellprob_threshold=cellprob,
                    normalize={"normalize": True, "percentile": [1, upper]},
                )
                # Guarda máscara como uint16 TIFF
                stem = fret_path.stem  # nombre sin extensión
                mask_name = stem + "_mask.tif"
                tifffile.imwrite(str(out_path / mask_name), masks.astype(np.uint16))

                well = extract_well(fret_path.name)
                log.append({"well": well, "file": fret_path.name,
                             "sparkles": int(masks.max())})
                bar.progress((i + 1) / len(fret_files),
                             text=f"{i+1}/{len(fret_files)} — {well}: {masks.max()} sparkles")

            bar.empty()
            st.session_state.masks_dir = str(out_path)
            st.session_state.detection_log = log
            st.success(f"✓ {len(fret_files)} images processed → {out_path}")

    # Resultados detección
    if st.session_state.detection_log:
        log_df = pd.DataFrame(st.session_state.detection_log)
        st.metric("Total sparkles detected", int(log_df["sparkles"].sum()))
        st.metric("Mean per image",           f"{log_df['sparkles'].mean():.1f}")

        fig, ax = plt.subplots(figsize=(4, 2.5))
        ax.bar(range(len(log_df)), log_df["sparkles"],
               color="#1E88E5", alpha=0.8, width=0.8)
        ax.set_xlabel("Image", fontsize=8)
        ax.set_ylabel("Sparkles", fontsize=8)
        ax.set_title("Sparkles per image", fontsize=9)
        ax.tick_params(labelsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ── Detección células ─────────────────────────────────────────────────────
    st.markdown("#### 🟢 Cells")

    cell_img_dir = path_input(
        "Images folder (CFP / YFP / V2)", "cell_img_dir",
        default="", is_folder=True
    )

    cell_model_path = path_input(
        "Cell model", "cell_model_path",
        default="D:/Cellpose/models/cell_mask_v1",
        filetypes=[("Modelo", "*"), ("Todos", "*.*")]
    )

    with st.expander("Inference parameters — cells"):
        cell_upper    = st.slider("upper percentile",   50, 99,   99,  key="cell_upper")
        cell_flow_thr = st.slider("flow_threshold",     0.0, 1.0, 0.4, 0.05, key="cell_flow_thr")
        cell_cellprob = st.slider("cellprob_threshold", -4.0, 2.0, 0.0, 0.1, key="cell_cellprob")

    cell_masks_out = path_input(
        "Output cell masks folder", "cell_masks_out",
        default="D:/Cellpose/results/masks_cells", is_folder=True
    )

    cell_channel_filter = path_input(
        "Filename filter (empty = all)", "cell_ch_filter",
        default="", browse=False
    )

    run_cells = st.button("▶  Detect cells", type="primary", key="btn_detect_cells")

    if run_cells:
        c_img_path = Path(cell_img_dir) if cell_img_dir else None
        c_out_path = Path(cell_masks_out)

        if not c_img_path or not c_img_path.exists():
            st.error("Select a valid images folder.")
        else:
            c_out_path.mkdir(parents=True, exist_ok=True)

            all_tifs_c = sorted(c_img_path.glob("*.tif")) + sorted(c_img_path.glob("*.tiff"))
            cell_files = [f for f in all_tifs_c if "_mask" not in f.name and "_cells" not in f.name]
            if cell_channel_filter:
                cell_files = [f for f in cell_files if cell_channel_filter in f.name]

            if not cell_files:
                st.error(f"No images found in: `{c_img_path}`")
                if all_tifs_c:
                    st.info(f"TIF files found ({len(all_tifs_c)}): "
                            + ", ".join(f.name for f in all_tifs_c[:5])
                            + ("..." if len(all_tifs_c) > 5 else ""))
                else:
                    st.warning("The folder contains no .tif or .tiff files")
            else:
                cell_model = load_model(cell_model_path)
                cell_log = []
                bar_c = st.progress(0, text="Processing images (cells)...")

                for i, img_path_c in enumerate(cell_files):
                    img_c = tifffile.imread(str(img_path_c)).astype(np.float32)
                    masks_c, _, _ = cell_model.eval(
                        img_c,
                        channels=[0, 0],
                        flow_threshold=cell_flow_thr,
                        cellprob_threshold=cell_cellprob,
                        normalize={"normalize": True, "percentile": [1, cell_upper]},
                    )
                    stem_c = img_path_c.stem
                    mask_name_c = stem_c + "_cells.tif"
                    tifffile.imwrite(str(c_out_path / mask_name_c), masks_c.astype(np.uint16))

                    well_c = extract_well(img_path_c.name)
                    cell_log.append({"well": well_c, "file": img_path_c.name,
                                     "cells": int(masks_c.max())})
                    bar_c.progress((i + 1) / len(cell_files),
                                   text=f"{i+1}/{len(cell_files)} — {well_c}: {masks_c.max()} cells")

                bar_c.empty()
                st.session_state.cell_detection_log = cell_log
                st.session_state.cell_masks_dir = str(c_out_path)
                st.success(f"✓ {len(cell_files)} images processed → {c_out_path}")

    if st.session_state.cell_detection_log:
        clog_df = pd.DataFrame(st.session_state.cell_detection_log)
        st.metric("Total cells detected", int(clog_df["cells"].sum()))
        st.metric("Mean per image",      f"{clog_df['cells'].mean():.1f}")

        fig_c, ax_c = plt.subplots(figsize=(4, 2.5))
        ax_c.bar(range(len(clog_df)), clog_df["cells"],
                 color="#2E7D32", alpha=0.8, width=0.8)
        ax_c.set_xlabel("Image", fontsize=8)
        ax_c.set_ylabel("Cells", fontsize=8)
        ax_c.set_title("Cells per image", fontsize=9)
        ax_c.tick_params(labelsize=7)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
        st.pyplot(fig_c)
        plt.close(fig_c)

    # ── Ratio combinado ───────────────────────────────────────────────────────
    if st.session_state.detection_log and st.session_state.cell_detection_log:
        per_img_ratio, overall_ratio = _match_and_ratio(
            st.session_state.detection_log,
            st.session_state.cell_detection_log,
        )
        st.divider()
        st.markdown("#### ⚡ Combined ratio")
        if overall_ratio is None:
            st.warning("No matching images found between sparkles and cells.")
        else:
            st.metric("Sparkles / cell (global)", f"{overall_ratio:.2f}")
            ratio_df = pd.DataFrame(per_img_ratio)
            fig_r, ax_r = plt.subplots(figsize=(4, 2.5))
            ax_r.bar(range(len(ratio_df)), ratio_df["ratio"],
                     color="#F57F17", alpha=0.85, width=0.8)
            ax_r.set_xlabel("Image", fontsize=8)
            ax_r.set_ylabel("Sparkles / cell", fontsize=8)
            ax_r.set_title("Ratio per image", fontsize=9)
            ax_r.tick_params(labelsize=7)
            ax_r.spines["top"].set_visible(False)
            ax_r.spines["right"].set_visible(False)
            st.pyplot(fig_r)
            plt.close(fig_r)

    st.divider()
    st.subheader("Step 2 — CellProfiler")
    st.caption(
        "With the generated masks, run the CellProfiler pipeline "
        "using the masks folder as input. "
        "The resulting CSVs are loaded in column 2."
    )
    if st.button("Open CellProfiler", key="btn_cp"):
        cp_exe = r"C:\Program Files\CellProfiler\CellProfiler.exe"
        try:
            subprocess.Popen([cp_exe])
            st.success("CellProfiler launched.")
        except FileNotFoundError:
            st.error(f"Not found: {cp_exe}\nOpen CellProfiler manually.")

def _do_fret_analysis(sparks, images, plate, out2):
    """Compute FRET metrics, average replicates, and save to session_state."""
    if "Metadata_Well" in sparks.columns:
        sparks["Well"] = sparks["Metadata_Well"]
    elif "FileName_FRET" in sparks.columns:
        sparks["Well"] = sparks["FileName_FRET"].apply(extract_well)

    if "Metadata_Well" not in images.columns:
        fn_col = next((c for c in images.columns if "FileName" in c), None)
        if fn_col:
            images["Metadata_Well"] = images[fn_col].apply(extract_well)

    par_col = next((c for c in ["Parent_Cells","Parent_Cells_YFP_background"]
                    if c in sparks.columns), None)
    if par_col:
        sparks = sparks[sparks[par_col] > 0].copy()

    int_col = next((c for c in sparks.columns
                    if "IntegratedIntensity_FRET" in c and "Edge" not in c), None)
    n_col   = next((c for c in images.columns if "Count_Cells" in c), None)

    s_per_img = sparks.groupby("ImageNumber").size().reset_index(name="Sparks_in_cells")
    mean_int  = (sparks.groupby("ImageNumber")[int_col].mean()
                 .reset_index().rename(columns={int_col: "mean_intensity"})
                 if int_col else None)

    img_cols = ["ImageNumber", "Metadata_Well", n_col]
    img_res  = (images[[c for c in img_cols if c and c in images.columns]]
                .rename(columns={"Metadata_Well": "Well", n_col: "Count_Cells"})
                .merge(s_per_img, on="ImageNumber", how="left"))
    if mean_int is not None:
        img_res = img_res.merge(mean_int, on="ImageNumber", how="left")
        img_res["mean_intensity"] = img_res["mean_intensity"].fillna(0)
    img_res["Sparks_in_cells"]   = img_res["Sparks_in_cells"].fillna(0)
    img_res["pct_positive_FRET"] = (
        img_res["Sparks_in_cells"] /
        img_res["Count_Cells"].replace(0, np.nan) * 100).fillna(0)
    if "mean_intensity" in img_res.columns:
        img_res["Integrated_FRET"] = img_res["pct_positive_FRET"] * img_res["mean_intensity"]

    well_agg = {k: (k, "mean") for k in ["Count_Cells","pct_positive_FRET",
                                           "mean_intensity","Integrated_FRET"]
                if k in img_res.columns}
    well_res = img_res.groupby("Well").agg(**well_agg).reset_index()
    result   = well_res.merge(plate, on="Well", how="left")

    metric_cols = [c for c in ["Count_Cells","pct_positive_FRET","mean_intensity","Integrated_FRET"]
                   if c in result.columns]
    grp_sample  = [c for c in ["Sample","Disease","Fraction"] if c in result.columns]
    sample_res  = (result.dropna(subset=grp_sample[:1])
                   .groupby(grp_sample)[metric_cols].mean().reset_index()
                   if grp_sample else result)

    Path(out2).mkdir(parents=True, exist_ok=True)
    result.to_csv(Path(out2) / "well_result.csv",    index=False)
    sample_res.to_csv(Path(out2) / "sample_result.csv", index=False)
    st.session_state.fret_summary = sample_res
    st.info(f"Technical replicates averaged → {len(sample_res)} samples")
    st.success("✓ Analysis complete")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COLUMNA 2 — ANÁLISIS FRET                                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with col2:
    st.header("2 · Measurement & FRET Analysis")

    col2_mode = st.radio(
        "Mode",
        ["Measure + Analyze (masks)", "Analyze existing CSVs"],
        horizontal=True, key="col2_mode"
    )

    if col2_mode == "Measure + Analyze (masks)":
        fret_dir2      = path_input("FRET images folder", "fret_dir2",
                                    default="", is_folder=True)
        mask_dir2      = path_input("Sparkle masks folder (*_mask.tif)", "masks_dir",
                                    default=st.session_state.get("masks_dir", ""),
                                    is_folder=True)
        cell_mask_dir2 = path_input("Cell masks folder (*_cells.tif)", "cell_masks_dir",
                                    default=st.session_state.get("cell_masks_dir", ""),
                                    is_folder=True)
        plate_path     = path_input("Plate info (.xlsx)", "plate_xlsx",
                                    filetypes=[("Excel","*.xlsx *.xls"),("All","*.*")])
        out2           = path_input("Output folder", "out2",
                                    default="D:/Cellpose/results/analysis", is_folder=True)

        x_col  = st.selectbox("X axis (groups)", ["Disease","Fraction","Sample"], key="xcol")
        fac_col= st.selectbox("Panels",          ["Fraction","Disease","—"],      key="faccol")
        if fac_col == "—": fac_col = None

        run_fret = st.button("▶  Measure + Analyze", type="primary", key="btn_fret")

        if run_fret:
            missing = [p for p in [fret_dir2, mask_dir2, cell_mask_dir2, plate_path] if not p]
            if missing:
                st.error("Required: FRET folder, sparkle masks, cell masks, and Plate info.")
            else:
                prog_bar = st.progress(0, text="Measuring ROIs...")
                try:
                    import measure_pipeline as mp
                    sparks, cells_df_raw, images_raw = mp.run(
                        fret_dir=fret_dir2,
                        sparkle_mask_dir=mask_dir2,
                        cell_mask_dir=cell_mask_dir2,
                        out_dir=out2,
                        progress_cb=lambda i, n, roi: prog_bar.progress(
                            i / n, text=f"[{i}/{n}] {roi}"),
                    )
                    for k, v in [("seg_sparks_path", "Sparks.csv"),
                                 ("seg_cells_path",  "Cells.csv"),
                                 ("seg_images_path", "Image.csv")]:
                        st.session_state[k] = str(Path(out2) / v)
                    prog_bar.empty()
                    st.success(f"✓ Measurement complete — {len(sparks)} sparkles | "
                               f"{len(images_raw)} images")
                except Exception as e:
                    prog_bar.empty()
                    st.error(f"Error during measurement: {e}")
                    st.stop()

                with st.spinner("Analyzing FRET..."):
                    sparks = pd.read_csv(st.session_state["seg_sparks_path"])
                    images = pd.read_csv(st.session_state["seg_images_path"])
                    plate  = pd.read_excel(plate_path)
                    plate.columns = [c.strip() for c in plate.columns]
                    _do_fret_analysis(sparks, images, plate, out2)

    else:  # Analyze existing CSVs
        sparks_path = path_input("Sparks.csv",  "sparks_csv",
                                 default=st.session_state.get("seg_sparks_path",""),
                                 filetypes=[("CSV","*.csv"),("Todos","*.*")])
        cells_path  = path_input("Cells.csv",   "cells_csv",
                                 default=st.session_state.get("seg_cells_path",""),
                                 filetypes=[("CSV","*.csv"),("Todos","*.*")])
        images_path = path_input("Image.csv",   "images_csv",
                                 default=st.session_state.get("seg_images_path",""),
                                 filetypes=[("CSV","*.csv"),("Todos","*.*")])
        plate_path  = path_input("Plate info (.xlsx)", "plate_xlsx",
                                 filetypes=[("Excel","*.xlsx *.xls"),("All","*.*")])
        out2        = path_input("Output folder", "out2",
                                 default="D:/Cellpose/results/analysis", is_folder=True)

        x_col  = st.selectbox("X axis (groups)", ["Disease","Fraction","Sample"], key="xcol")
        fac_col= st.selectbox("Panels",          ["Fraction","Disease","—"],      key="faccol")
        if fac_col == "—": fac_col = None

        run_fret = st.button("▶  Analyze FRET", type="primary", key="btn_fret")

        if run_fret:
            missing = [p for p in [sparks_path, images_path, plate_path] if not p]
            if missing:
                st.error("Select Sparks.csv, Image.csv and Plate info.")
            else:
                with st.spinner("Analyzing..."):
                    sparks = pd.read_csv(sparks_path)
                    images = pd.read_csv(images_path)
                    plate  = pd.read_excel(plate_path)
                    plate.columns = [c.strip() for c in plate.columns]
                    _do_fret_analysis(sparks, images, plate, out2)

    # ── Filtro outliers col2 ───────────────────────────────────────────────────
    with st.expander("Filter outliers"):
        filter2_on     = st.checkbox("Enable outlier filter", value=False, key="filter2_on")
        outlier2_method= st.radio("Method", ["IQR", "Z-score"], horizontal=True, key="outlier2_method")
        outlier2_thresh= st.slider(
            "Threshold  (IQR: k×IQR  |  Z-score: max σ)",
            min_value=1.0, max_value=5.0, value=1.5, step=0.5, key="outlier2_thresh"
        )

    # Resultados FRET
    if st.session_state.fret_summary is not None:
        result = st.session_state.fret_summary.copy()

        if filter2_on:
            metric_filt = [c for c in ["pct_positive_FRET","mean_intensity","Integrated_FRET"]
                           if c in result.columns]
            result, n_rem2 = filter_outliers(result, metric_filt,
                                             method=outlier2_method, threshold=outlier2_thresh)
            st.info(f"Outlier filter ({outlier2_method}, threshold={outlier2_thresh}): "
                    f"{n_rem2} samples removed — {len(result)} remaining")

        grp_cols = [c for c in [x_col, fac_col] if c and c in result.columns]

        if grp_cols:
            summary = (result.dropna(subset=grp_cols[:1])
                       .groupby(grp_cols)[["Count_Cells","pct_positive_FRET",
                                           "mean_intensity","Integrated_FRET"]]
                       .mean().round(2))
            st.dataframe(summary, use_container_width=True)

        # Figura rápida: % FRET positivo por grupo
        if x_col in result.columns and "pct_positive_FRET" in result.columns:
            df_p = result.dropna(subset=[x_col])
            x_vals = ([d for d in DISEASE_ORDER if d in df_p[x_col].unique()]
                      if x_col == "Disease"
                      else sorted(df_p[x_col].dropna().unique()))
            fac_vals = (sorted(df_p[fac_col].dropna().unique())
                        if fac_col and fac_col in df_p.columns else [None])

            fig, axes = plt.subplots(1, len(fac_vals),
                                     figsize=(4 * len(fac_vals), 3.5),
                                     squeeze=False)
            np.random.seed(42)
            for ci, fv in enumerate(fac_vals):
                ax = axes[0, ci]
                data = df_p[df_p[fac_col] == fv] if fv and fac_col else df_p
                for xi, xv in enumerate(x_vals):
                    vals = data[data[x_col] == xv]["pct_positive_FRET"].dropna()
                    if vals.empty: continue
                    col_c = DISEASE_COLORS.get(xv, "#1E88E5")
                    ax.bar(xi, vals.mean(), 0.55, color=col_c, alpha=0.6)
                    ax.errorbar(xi, vals.mean(), vals.sem(),
                                fmt="none", color="black", capsize=5, lw=1.5)
                    ax.scatter(xi + np.random.uniform(-0.12,0.12,len(vals)),
                               vals, color=col_c, edgecolors="black",
                               lw=0.4, s=35, zorder=5, alpha=0.9)
                ax.set_xticks(range(len(x_vals)))
                ax.set_xticklabels(x_vals, fontsize=8, rotation=15, ha="right")
                ax.set_ylabel("% FRET positive", fontsize=9)
                ax.set_title(str(fv) if fv else "All", fontsize=10, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.set_ylim(bottom=0)
            plt.tight_layout()
            st.pyplot(fig)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  COLUMNA 3 — CARACTERIZACIÓN Y PATRONES                                  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
with col3:
    st.header("3 · Characterization & Patterns")

    sparks3_path = path_input("Sparks.csv", "sparks3",
                              default=st.session_state.get("sparks_csv", ""),
                              filetypes=[("CSV","*.csv"),("All","*.*")])
    cells3_path  = path_input("Cells.csv",  "cells3",
                              default=st.session_state.get("cells_csv", ""),
                              filetypes=[("CSV","*.csv"),("All","*.*")])
    plate3_path  = path_input("Plate info (.xlsx)", "plate3",
                              default=st.session_state.get("plate_xlsx", ""),
                              filetypes=[("Excel","*.xlsx *.xls"),("All","*.*")])
    out3         = path_input("Output folder", "out3",
                              default="D:/Cellpose/results/patterns", is_folder=True)

    analysis_opts = st.multiselect(
        "Analyses to run",
        ["Correlation", "PCA", "UMAP", "Patient heatmap", "Random Forest", "Statistical tests"],
        default=["PCA", "UMAP", "Patient heatmap", "Random Forest"],
        key="analysis_opts"
    )

    run_pat = st.button("▶  Characterize & analyze", type="primary", key="btn_pat")

    if run_pat:
        missing3 = [p for p in [sparks3_path, plate3_path] if not p]
        if missing3:
            st.error("You need at least Sparks.csv and Plate info.")
        else:
            with st.spinner("Preparing data..."):
                from sklearn.preprocessing import StandardScaler
                from sklearn.decomposition import PCA as skPCA
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.model_selection import cross_val_score

                sparks3 = pd.read_csv(sparks3_path)
                plate3  = pd.read_excel(plate3_path)
                plate3.columns = [c.strip() for c in plate3.columns]
                cells3  = pd.read_csv(cells3_path) if cells3_path else None

                # Well
                if "Metadata_Well" in sparks3.columns:
                    sparks3["Well"] = sparks3["Metadata_Well"]
                elif "FileName_FRET" in sparks3.columns:
                    sparks3["Well"] = sparks3["FileName_FRET"].apply(extract_well)

                par_col3 = next((c for c in ["Parent_Cells","Parent_Cells_YFP_background"]
                                 if c in sparks3.columns), None)
                if par_col3:
                    sparks3 = sparks3[sparks3[par_col3] > 0].copy()

                # Renombra features
                SHAPE = {"AreaShape_Area":"area_px","AreaShape_EquivalentDiameter":"diameter_px",
                         "AreaShape_FormFactor":"circularity","AreaShape_Eccentricity":"eccentricity",
                         "AreaShape_Solidity":"solidity","AreaShape_Compactness":"compactness",
                         "AreaShape_MajorAxisLength":"major_axis_px","AreaShape_MinorAxisLength":"minor_axis_px"}
                FRET_C = {"Intensity_IntegratedIntensity_FRET":"fret_integrated",
                          "Intensity_MeanIntensity_FRET":"fret_mean",
                          "Intensity_MaxIntensity_FRET":"fret_max"}
                remap = {**{k:v for k,v in SHAPE.items() if k in sparks3.columns},
                         **{k:v for k,v in FRET_C.items() if k in sparks3.columns}}
                sparks3.rename(columns=remap, inplace=True)

                # FRET/Cell ratio
                cfp_col = next((c for c in (cells3.columns if cells3 is not None else [])
                                if "MeanIntensity_CFP" in c), None)
                if cfp_col and par_col3 and cells3 is not None:
                    cb = (cells3[["ImageNumber","ObjectNumber",cfp_col]]
                          .rename(columns={"ObjectNumber":par_col3, cfp_col:"cell_bg"}))
                    sparks3 = sparks3.merge(cb, on=["ImageNumber",par_col3], how="left")
                    sparks3["fret_over_cell"] = sparks3["fret_mean"] / sparks3["cell_bg"].replace(0, np.nan)

                sparks3 = sparks3.merge(plate3, on="Well", how="left")

                FEATURES = [f for f in ["area_px","diameter_px","circularity","eccentricity",
                                         "solidity","compactness","major_axis_px","minor_axis_px",
                                         "fret_integrated","fret_mean","fret_over_cell"]
                            if f in sparks3.columns]

                df_c = sparks3.dropna(subset=["Disease"] + FEATURES).copy()

                diseases3 = [d for d in DISEASE_ORDER if d in df_c["Disease"].unique()]
                fractions3 = sorted(df_c["Fraction"].dropna().unique()) if "Fraction" in df_c.columns else []

                X = df_c[FEATURES].values
                X_sc = StandardScaler().fit_transform(X)

                Path(out3).mkdir(parents=True, exist_ok=True)
                sparks3.to_csv(Path(out3) / "sparkle_features.csv", index=False)
                st.success(f"✓ {len(df_c)} sparkles | features: {FEATURES}")

            # ── PCA ────────────────────────────────────────────────────────
            if "PCA" in analysis_opts:
                with st.spinner("PCA..."):
                    pca = skPCA(n_components=2)
                    X_pca = pca.fit_transform(X_sc)
                    var = pca.explained_variance_ratio_ * 100

                    idx = (np.random.choice(len(df_c), 4000, replace=False)
                           if len(df_c) > 4000 else np.arange(len(df_c)))
                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    for d in diseases3:
                        mask = df_c["Disease"].iloc[idx].values == d
                        ax.scatter(X_pca[idx][mask,0], X_pca[idx][mask,1],
                                   c=DISEASE_COLORS.get(d,"gray"), label=d,
                                   alpha=0.35, s=6, edgecolors="none")
                    ax.set_xlabel(f"PC1 ({var[0]:.1f}%)", fontsize=9)
                    ax.set_ylabel(f"PC2 ({var[1]:.1f}%)", fontsize=9)
                    ax.set_title("PCA — Sparkles", fontsize=10, fontweight="bold")
                    ax.legend(fontsize=8, markerscale=3, frameon=False)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

            # ── UMAP ───────────────────────────────────────────────────────
            if "UMAP" in analysis_opts:
                with st.spinner("UMAP (1-2 min)..."):
                    import umap as umap_lib
                    N = min(len(df_c), 6000)
                    idx_u = (np.random.choice(len(df_c), N, replace=False)
                             if len(df_c) > N else np.arange(len(df_c)))
                    reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1,
                                            random_state=42, verbose=False)
                    Xu = reducer.fit_transform(X_sc[idx_u])
                    df_u = df_c.iloc[idx_u].copy()

                    fig, ax = plt.subplots(figsize=(4.5, 3.5))
                    for d in diseases3:
                        mask = df_u["Disease"].values == d
                        ax.scatter(Xu[mask,0], Xu[mask,1],
                                   c=DISEASE_COLORS.get(d,"gray"), label=d,
                                   alpha=0.35, s=6, edgecolors="none")
                    ax.set_xlabel("UMAP1", fontsize=9)
                    ax.set_ylabel("UMAP2", fontsize=9)
                    ax.set_title("UMAP — Sparkles", fontsize=10, fontweight="bold")
                    ax.legend(fontsize=8, markerscale=3, frameon=False)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

            # ── Heatmap pacientes ──────────────────────────────────────────
            if "Patient heatmap" in analysis_opts:
                with st.spinner("Heatmap..."):
                    grp = [c for c in ["Sample","Disease","Fraction"] if c in df_c.columns]
                    pat = df_c.groupby(grp)[FEATURES].median().reset_index().dropna(subset=FEATURES)
                    fz  = StandardScaler().fit_transform(pat[FEATURES].values)
                    fz_df = pd.DataFrame(fz, columns=FEATURES)
                    row_lbl = [f"{r.get('Sample','?')} ({r.get('Disease','?')})"
                               for _, r in pat.iterrows()]
                    row_col = pd.Series(
                        [DISEASE_COLORS.get(d,"#aaa")
                         for d in pat.get("Disease",[""] * len(pat))],
                        name="Disease")
                    clg = sns.clustermap(
                        fz_df, row_colors=row_col,
                        cmap="RdBu_r", center=0,
                        yticklabels=row_lbl, xticklabels=FEATURES,
                        figsize=(max(8, len(FEATURES)*0.85),
                                 max(6, len(pat)*0.35)),
                        dendrogram_ratio=0.12,
                        cbar_pos=(0.02, 0.82, 0.03, 0.12),
                    )
                    clg.ax_heatmap.set_xticklabels(
                        clg.ax_heatmap.get_xticklabels(),
                        rotation=35, ha="right", fontsize=7)
                    clg.ax_heatmap.set_yticklabels(
                        clg.ax_heatmap.get_yticklabels(), fontsize=6)
                    patches = [mpatches.Patch(color=DISEASE_COLORS.get(d,"gray"),label=d)
                               for d in diseases3]
                    clg.fig.legend(handles=patches, loc="upper right",
                                   fontsize=8, bbox_to_anchor=(1.12,1.0), frameon=False)
                    clg.fig.suptitle("Hierarchical heatmap — patients",
                                     fontsize=10, fontweight="bold", y=1.02)
                    buf = io.BytesIO()
                    clg.fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    plt.close("all")
                    st.image(buf, use_container_width=True)

            # ── Random Forest ──────────────────────────────────────────────
            if "Random Forest" in analysis_opts:
                with st.spinner("Random Forest..."):
                    from sklearn.ensemble import RandomForestClassifier
                    from sklearn.model_selection import cross_val_score
                    df_rf = df_c[df_c["Disease"].isin(["Control","AD"])].copy()
                    if len(df_rf) >= 20:
                        Xrf = df_rf[FEATURES].values
                        yrf = (df_rf["Disease"]=="AD").astype(int).values
                        rf  = RandomForestClassifier(200, max_depth=6,
                                                     random_state=42, n_jobs=-1)
                        auc = cross_val_score(rf, Xrf, yrf, cv=5,
                                             scoring="roc_auc").mean()
                        rf.fit(Xrf, yrf)
                        imp = pd.DataFrame({"feature":FEATURES,
                                            "importance":rf.feature_importances_}
                                           ).sort_values("importance", ascending=False)

                        fig, ax = plt.subplots(figsize=(4, len(FEATURES)*0.42+0.5))
                        cols_rf = ["#E53935" if i==0 else "#1E88E5" if i==1
                                   else "#888" for i in range(len(imp))]
                        ax.barh(imp["feature"][::-1], imp["importance"][::-1],
                                color=cols_rf[::-1])
                        ax.set_xlabel("Importancia (Gini)", fontsize=9)
                        ax.set_title(f"Random Forest\nAUC = {auc:.3f}",
                                     fontsize=10, fontweight="bold")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        imp.to_csv(Path(out3)/"feature_importance.csv", index=False)
                    else:
                        st.warning("Not enough data for Random Forest.")

            # ── Tests estadísticos ─────────────────────────────────────────
            if "Statistical tests" in analysis_opts:
                with st.spinner("Statistical tests..."):
                    # Promedia réplicas técnicas: una mediana por Sample+Fraction
                    grp_s = [c for c in ["Sample","Disease","Fraction"]
                             if c in df_c.columns]
                    if "Sample" in df_c.columns:
                        sample_med = (df_c.dropna(subset=grp_s)
                                      .groupby(grp_s)[FEATURES]
                                      .median().reset_index())
                        st.caption(f"Tests on {len(sample_med)} samples "
                                   f"(technical replicates averaged by Sample)")
                    else:
                        sample_med = df_c  # fallback si no hay columna Sample

                    rows = []
                    pairs = [(diseases3[i], diseases3[j])
                             for i in range(len(diseases3))
                             for j in range(i+1, len(diseases3))]
                    fracs = fractions3 if fractions3 else [None]
                    for feat in FEATURES:
                        for g1, g2 in pairs:
                            for frac in fracs:
                                sub = (sample_med[sample_med["Fraction"]==frac]
                                       if frac and "Fraction" in sample_med.columns
                                       else sample_med)
                                v1 = sub[sub["Disease"]==g1][feat].dropna()
                                v2 = sub[sub["Disease"]==g2][feat].dropna()
                                if len(v1)<3 or len(v2)<3: continue
                                _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                                rows.append({"feature":feat,"group1":g1,"group2":g2,
                                             "fraction":frac or "all",
                                             "n1":len(v1),"n2":len(v2),
                                             "p":round(p,5),
                                             "sig": "***" if p<0.001 else "**" if p<0.01
                                                    else "*" if p<0.05 else ""})
                    st_df = pd.DataFrame(rows)
                    sig_df = st_df[st_df["sig"]!=""].sort_values("p")
                    st.markdown(f"**Significant (p<0.05): {len(sig_df)} / {len(st_df)}**")
                    st.dataframe(sig_df[["feature","group1","group2",
                                         "fraction","n1","n2","p","sig"]],
                                 use_container_width=True, height=250)
                    st_df.to_csv(Path(out3)/"statistical_tests.csv", index=False)

            st.success(f"✓ Analysis complete → {out3}")
