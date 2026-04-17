# SparkFRET

**Automated detection and quantification of FRET sparkle events in biosensor microscopy images.**

Developed at Baylor College of Medicine — CLR Lab.

---

## What it does

SparkFRET provides a complete pipeline for:
1. **Sparkle detection** — custom Cellpose model trained on FRET biosensor images
2. **Cell segmentation** — custom Cellpose model trained on cell morphology images
3. **FRET quantification** — sparkles per cell, integrated FRET intensity, FRET ratios
4. **Multivariate characterization** — PCA, UMAP, Random Forest, statistical tests, publication-ready figures

All steps are accessible through a single **Streamlit web hub** (`sparkfret_hub.py`).

---

## System requirements

| Component | Requirement |
|-----------|------------|
| OS | Windows 10/11 (64-bit) |
| Python | 3.10 – 3.13 |
| GPU | NVIDIA GPU recommended (CUDA 12.x), CPU fallback available |
| RAM | 16 GB minimum, 32 GB recommended |
| Disk | ~5 GB (models + venv) |

---

## Installation

### Windows (automatic)

```bat
install.bat
```

This creates a virtual environment, detects your GPU, installs PyTorch + all dependencies, and creates a `launch_hub.bat` shortcut.

### Manual installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Install PyTorch (choose one):
# NVIDIA GPU (CUDA 12.1):
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 3. Install all other dependencies
venv\Scripts\pip install -r requirements.txt
```

---

## Usage

### Launch the hub

```bat
launch_hub.bat
```

Or manually:
```bash
venv\Scripts\streamlit run sparkfret_hub.py
```

The hub opens in your browser at `http://localhost:8501`.

### Hub workflow

```
Col 1: Detection              Col 2: FRET Analysis          Col 3: Characterization
──────────────────────        ─────────────────────────     ────────────────────────
Select FRET image folder  →   Load sparkle + cell masks →   Load Sparks.csv
Select sparkle model          Quantify FRET per sparkle     Load Plate info (.xlsx)
Run Cellpose (sparkles)       Compute sparkles/cell ratio   PCA / UMAP
Select cell image folder      Export Excel + CSV            Correlation matrix
Select cell model                                           Random Forest classifier
Run Cellpose (cells)                                        Patient heatmap
View sparkles/cell ratio                                    Statistical tests
```

**Col 1** runs two independent segmentation steps using custom Cellpose models:
- **Sparkles** — custom model (`sparkle_fret_v9`) applied to the FRET channel images
- **Cells** — custom model (`cell_mask_v1`) applied to a separate image folder (e.g. CFP or YFP channel), with optional filename filter to select the right channel

Both mask sets are saved to disk and passed automatically to Col 2.

**Col 2** performs mask-based FRET analysis — reads the pre-computed sparkle and cell masks from Col 1, quantifies FRET intensity per sparkle, computes sparkles/cell ratio, and exports results to Excel + CSV.

**Col 3** performs multivariate characterization of sparkle morphology and FRET features: PCA, UMAP, correlation matrix, Random Forest classification, per-patient heatmap, and statistical tests — grouped by experimental condition from the Plate info file.

---

## File structure

```
SparkFRET/
├── sparkfret_hub.py          # Main Streamlit hub (Col 1, 2, 3)
├── measure_pipeline.py       # Mask-based FRET quantification (pure Python)
├── train_model.py            # Custom model training script
├── batch_generate_masks.py   # Batch mask generation (CLI)
├── predict_and_review.py     # Active learning helper
├── analyze_fret.py           # Standalone FRET analysis
├── analyze_patterns.py       # Pattern analysis utilities
├── sparkle_fret.cppipe       # CellProfiler pipeline (optional)
├── requirements.txt          # Python dependencies
├── install.bat               # Windows auto-installer
├── launch_hub.bat            # Hub launcher
└── tests/                    # Unit tests
    ├── test_hub_helpers.py
    └── test_measure_pipeline.py
```

> **Models and data are not included.** Place trained Cellpose models in a `models/` folder alongside the scripts. Place images in a local folder of your choice — the hub uses folder browser dialogs to select paths at runtime.

---

## Models

Trained models are not included in this repository due to file size.

| Model | Detects | Performance |
|-------|---------|-------------|
| `sparkle_fret_v9` | FRET sparkles | Loss 0.0061 (500 epochs) |
| `cell_mask_v1` | Cells | — |

Download models from [Releases](../../releases) and place them in the `models/` folder.

**Recommended inference parameters — `sparkle_fret_v9`:**
- `flow_threshold = 0.8`
- `cellprob_threshold = -3.0`
- `normalize upper percentile = 99`

**Recommended inference parameters — `cell_mask_v1`:**
- `flow_threshold = 0.4`
- `cellprob_threshold = 0.0`
- `normalize upper percentile = 99`

---

## Image naming convention

Images must follow the Cytation naming pattern:

```
{Well}ROI{N}_CFP YFP FRET V2.tif   ← FRET channel  → sparkle detection (Col 1, sparkles)
{Well}ROI{N}_CFP YFP FRET V2.tif   ← or any channel → cell detection (Col 1, cells — use filename filter)
```

Example: `A1ROI1_CFP YFP FRET V2.tif`

The cell detection step accepts any image folder and a filename filter string to select the correct channel (e.g. `CFP FRET V2` to use the donor channel for cell segmentation).

Output masks follow this convention:
```
{stem}_mask.tif    ← Sparkle mask (generated by Col 1 sparkle detection)
{stem}_cells.tif   ← Cell mask   (generated by Col 1 cell detection)
```

---

## CellProfiler (optional)

The standard workflow uses pure-Python mask-based analysis (`measure_pipeline.py`) and does not require CellProfiler. A CellProfiler pipeline (`sparkle_fret.cppipe`) is included for users who prefer it or need an independent validation.

### Using the included pipeline

The pipeline performs:
1. Loads FRET images + sparkle masks generated by SparkFRET
2. Converts masks to objects (Sparks)
3. Detects cells from the CFP channel (IdentifyPrimaryObjects)
4. Assigns each spark to its parent cell (RelateObjects)
5. Measures FRET intensity per spark and per cell
6. Exports `Sparks.csv`, `Cells.csv`, `Image.csv`

### Setup in CellProfiler

1. Download CellProfiler 4.x from https://cellprofiler.org/releases
2. Open CellProfiler → `File > Open Pipeline` → select `sparkle_fret.cppipe`
3. In the **Images** module, drag the folder with original FRET `.tif` images and the folder with `_mask.tif` files
4. In **NamesAndTypes**, click **Update** and verify two columns: `FRET` and `SparkMask`
5. Set `File > Preferences > Default Output Folder` to your results folder
6. Click **Analyze Images**

---

## Training a new model

```bash
# Edit train_model.py to set TRAIN_DIR and MODEL_NAME, then:
venv\Scripts\python train_model.py
```

Training data: annotate images using the Cellpose GUI and save `_seg.npy` masks alongside the `.tif` images in `training_data/raw/`. The same workflow applies for both sparkle and cell models.

---

## Output files

After running the full pipeline through the hub:

| File | Content |
|------|---------|
| `<output_dir>/masks/` | Sparkle masks (uint16 TIF, `_mask.tif` per image) |
| `<output_dir>/cell_masks/` | Cell masks (uint16 TIF, `_cells.tif` per image) |
| `<output_dir>/Sparks.csv` | Per-sparkle measurements (intensity, area, position) |
| `<output_dir>/Cells.csv` | Per-cell measurements |
| `<output_dir>/Image.csv` | Per-image summary (sparkle count, total FRET) |
| `<output_dir>/summary.xlsx` | Excel workbook with all sheets |
| `<output_dir>/analysis/` | Figures (PNG + SVG, publication-ready) |

---

## Changelog

### v0.3 — 2026-04-17

**Col 1 — cell detection + sparkles/cell ratio**
- Added **cell detection sub-section**: runs custom Cellpose cell model (`cell_mask_v1`) to generate cell masks alongside sparkle masks.
- Added **sparkles/cell ratio panel**: after both segmentation steps complete, Col 1 displays a combined figure showing detected sparkles overlaid on cells and the computed ratio per image.
- Cell mask output folder is saved to session state and forwarded automatically to Col 2.

**Col 2 — mask-based FRET analysis**
- Col 2 now operates exclusively on **pre-computed masks** (sparkle masks + cell masks from Col 1). The previous mode that ran its own segmentation has been removed.
- `measure_pipeline.py` rewritten to accept existing mask TIFs, removing the CellProfiler dependency for the standard workflow.
- Guards added for empty DataFrames and images with zero ROI matches.

**General**
- All UI labels and messages translated to English.
- Fixed session-state key collision between Col 1 `img_dir` and Col 2 `fret_dir2`.
- Fixed widget key conflicts on sparkle threshold sliders.
- Chart figures now explicitly closed after rendering to prevent memory leaks.

---

## Citation

If you use SparkFRET in your research, please cite:

> [Manuscript in preparation] — CLR Lab, Baylor College of Medicine

---

## License

MIT License — see [LICENSE](LICENSE)
