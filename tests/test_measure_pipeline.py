import pytest
import numpy as np
import pandas as pd
import tifffile
from pathlib import Path
import tempfile, shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import measure_pipeline as mp


# ── extract_meta ──────────────────────────────────────────────────────────────
def test_extract_meta_valid():
    well, roi_num, roi_id = mp.extract_meta("A1ROI3_FRET.tif")
    assert well == "A1"
    assert roi_num == "3"
    assert roi_id == "A1ROI3"

def test_extract_meta_invalid():
    well, roi_num, roi_id = mp.extract_meta("random_file.tif")
    assert well is None and roi_num is None and roi_id is None


# ── measure_shape ─────────────────────────────────────────────────────────────
def test_measure_shape_two_objects():
    labeled = np.zeros((50, 50), dtype=np.uint16)
    labeled[5:15, 5:15]   = 1
    labeled[30:40, 30:40] = 2
    df = mp.measure_shape(labeled)
    assert len(df) == 2
    assert "ObjectNumber" in df.columns
    assert "AreaShape_Area" in df.columns
    assert set(df["ObjectNumber"]) == {1, 2}

def test_measure_shape_empty():
    labeled = np.zeros((50, 50), dtype=np.uint16)
    df = mp.measure_shape(labeled)
    assert df.empty


# ── relate_sparks_to_cells ────────────────────────────────────────────────────
def test_relate_sparks_fully_inside_cell():
    cells   = np.zeros((20, 20), dtype=np.uint16)
    sparks  = np.zeros((20, 20), dtype=np.uint16)
    cells[2:18, 2:18]  = 1
    sparks[5:10, 5:10] = 1
    parents = mp.relate_sparks_to_cells(sparks, cells)
    assert parents[1] == 1

def test_relate_sparks_outside_any_cell():
    cells   = np.zeros((20, 20), dtype=np.uint16)
    sparks  = np.zeros((20, 20), dtype=np.uint16)
    cells[2:8, 2:8]      = 1
    sparks[12:18, 12:18] = 1
    parents = mp.relate_sparks_to_cells(sparks, cells)
    assert parents[1] == 0


# ── run() integration ─────────────────────────────────────────────────────────
@pytest.fixture
def synthetic_dirs(tmp_path):
    fret_dir  = tmp_path / "fret"
    spark_dir = tmp_path / "sparks"
    cell_dir  = tmp_path / "cells"
    for d in [fret_dir, spark_dir, cell_dir]:
        d.mkdir()

    img = np.random.randint(100, 1000, (64, 64), dtype=np.uint16)
    spark_mask = np.zeros((64, 64), dtype=np.uint16)
    spark_mask[10:20, 10:20] = 1
    spark_mask[30:38, 30:38] = 2
    cell_mask  = np.zeros((64, 64), dtype=np.uint16)
    cell_mask[5:45, 5:45] = 1

    tifffile.imwrite(str(fret_dir  / "A1ROI1_CFP YFP FRET.tif"), img)
    tifffile.imwrite(str(spark_dir / "A1ROI1_CFP YFP FRET_mask.tif"), spark_mask)
    tifffile.imwrite(str(cell_dir  / "A1ROI1_CFP YFP FRET_cells.tif"), cell_mask)
    return fret_dir, spark_dir, cell_dir, tmp_path / "out"


def test_run_produces_csvs(synthetic_dirs):
    fret_dir, spark_dir, cell_dir, out_dir = synthetic_dirs
    sparks_df, cells_df, images_df = mp.run(
        fret_dir=str(fret_dir),
        sparkle_mask_dir=str(spark_dir),
        cell_mask_dir=str(cell_dir),
        out_dir=str(out_dir),
    )
    assert (out_dir / "Sparks.csv").exists()
    assert (out_dir / "Cells.csv").exists()
    assert (out_dir / "Image.csv").exists()
    assert len(sparks_df) == 2
    assert len(images_df) == 1
    assert images_df["Count_Cells"].iloc[0] == 1
    assert images_df["Count_Sparks"].iloc[0] == 2


def test_run_sparks_have_parent(synthetic_dirs):
    fret_dir, spark_dir, cell_dir, out_dir = synthetic_dirs
    sparks_df, _, _ = mp.run(
        fret_dir=str(fret_dir),
        sparkle_mask_dir=str(spark_dir),
        cell_mask_dir=str(cell_dir),
        out_dir=str(out_dir),
    )
    assert (sparks_df["Parent_Cells"] == 1).all()
