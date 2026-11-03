import numpy as np
import pandas as pd

from app.features import compute, label


def _df(n=400, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=idx)


def test_features_have_no_lookahead():
    df = _df()
    full = compute(df)
    cut = compute(df.iloc[:300])
    cols = ["rsi14", "adx14", "atr_pct", "dist_ema200", "vol_surge", "ret_20", "bb_pos"]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols])


def test_label_matches_manual_barrier():
    f = label(compute(_df(seed=3)))
    i = 250
    r = f.iloc[i]
    tgt, stp = r.close + 3.5 * r.atr, r.close - 1.75 * r.atr
    win = None
    for j in range(i + 1, i + 21):
        if f.low.iloc[j] <= stp:
            win = 0
            break
        if f.high.iloc[j] >= tgt:
            win = 1
            break
    assert r.y == (win or 0)
    assert 1 <= r.days_held <= 20


def test_last_rows_unlabelled():
    f = label(compute(_df()))
    assert f.y.iloc[-20:].isna().all()
