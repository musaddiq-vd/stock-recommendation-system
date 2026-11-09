import numpy as np
import pandas as pd

from app.sectors import quadrant, rotation


def test_rotation_marks_an_outperforming_sector_as_leading():
    idx = pd.bdate_range("2026-01-01", periods=200)
    days = np.arange(200)
    # strong sector: steady outperformance that speeds up at the end; weak sector: steady underperformance
    strong = 100 * 1.001 ** days * np.where(days > 170, 1.002 ** (days - 170), 1)
    weak = 100 * 0.9995 ** days
    closes = pd.DataFrame({f"S{i}": strong * (1 + i / 1000) for i in range(3)}
                          | {f"W{i}": weak * (1 + i / 1000) for i in range(3)}, index=idx)
    sectors = pd.Series({**{f"S{i}": "Strong" for i in range(3)}, **{f"W{i}": "Weak" for i in range(3)}})
    rot = rotation(closes, sectors)
    s, w = rot["Strong"][-1], rot["Weak"][-1]
    assert len(rot["Strong"]) == 6
    assert quadrant(s["x"], s["y"]) == "leading"
    assert w["x"] < 100


def test_small_sectors_are_skipped():
    idx = pd.bdate_range("2026-01-01", periods=120)
    closes = pd.DataFrame({"A": np.linspace(100, 120, 120), "B": np.linspace(100, 90, 120)}, index=idx)
    assert rotation(closes, pd.Series({"A": "X", "B": "X"})) == {}
