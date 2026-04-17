import pytest
import sys
sys.path.insert(0, "D:/Cellpose")

from sparkfret_hub import _match_and_ratio


def test_basic_ratio():
    sparks_log = [
        {"well": "A1", "file": "A1ROI1_FRET.tif",  "sparkles": 10},
        {"well": "A2", "file": "A2ROI1_FRET.tif",  "sparkles": 6},
    ]
    cells_log = [
        {"well": "A1", "file": "A1ROI1_V2.tif", "cells": 5},
        {"well": "A2", "file": "A2ROI1_V2.tif", "cells": 3},
    ]
    per_img, overall = _match_and_ratio(sparks_log, cells_log)
    assert overall == pytest.approx(16 / 8)
    assert len(per_img) == 2


def test_unmatched_images_excluded():
    sparks_log = [{"well": "A1", "file": "A1ROI1_FRET.tif", "sparkles": 10}]
    cells_log  = [{"well": "B1", "file": "B1ROI1_V2.tif",   "cells": 5}]
    per_img, overall = _match_and_ratio(sparks_log, cells_log)
    assert per_img == []
    assert overall is None


def test_zero_cells_excluded_from_ratio():
    sparks_log = [
        {"well": "A1", "file": "A1ROI1_FRET.tif", "sparkles": 8},
        {"well": "A2", "file": "A2ROI1_FRET.tif", "sparkles": 4},
    ]
    cells_log = [
        {"well": "A1", "file": "A1ROI1_V2.tif", "cells": 0},
        {"well": "A2", "file": "A2ROI1_V2.tif", "cells": 2},
    ]
    per_img, overall = _match_and_ratio(sparks_log, cells_log)
    # A1 has 0 cells → excluded from per-img; only A2 contributes
    assert len(per_img) == 1
    assert overall == pytest.approx(4 / 2)
