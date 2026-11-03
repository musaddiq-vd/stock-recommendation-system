import numpy as np
import pandas as pd

from app.features import add_nse, compute
from app.paper import check_bar
from tests.test_features import _df


def _nse(idx, seed=1):
    rng = np.random.default_rng(seed)
    n = len(idx)
    return pd.DataFrame({
        "deliv_per": rng.uniform(20, 70, n), "deliv_qty": rng.integers(1e5, 1e6, n).astype(float),
        "fut_oi": rng.integers(1e6, 2e6, n).astype(float), "call_oi": rng.integers(1e5, 1e6, n).astype(float),
        "put_oi": rng.integers(1e5, 1e6, n).astype(float),
    }, index=idx)


def test_nse_features_have_no_lookahead():
    df = _df()
    nse = _nse(df.index)
    full = add_nse(compute(df), nse)
    cut = add_nse(compute(df.iloc[:300]), nse.iloc[:300])
    cols = ["deliv_rel20", "deliv_qty_surge", "fut_oi_chg5", "oi_buildup", "pcr", "pcr_chg5"]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols])


def test_missing_nse_is_nan():
    f = add_nse(compute(_df()), None)
    assert f.deliv_per.isna().all() and f.pcr.isna().all()


def test_check_bar_long_and_short():
    long_t = {"side": "long", "target": 110, "stop_loss": 95}
    short_t = {"side": "short", "target": 90, "stop_loss": 105}
    assert check_bar(long_t, 111, 100) == ("target", 110)
    assert check_bar(long_t, 111, 94) == ("stop", 95)  # both touched: stop wins
    assert check_bar(long_t, 105, 99) is None
    assert check_bar(short_t, 101, 89) == ("target", 90)
    assert check_bar(short_t, 106, 89) == ("stop", 105)
