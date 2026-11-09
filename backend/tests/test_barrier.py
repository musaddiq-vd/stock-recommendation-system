import numpy as np

from app.features import barrier_outcomes, compute
from tests.test_features import _df


def _slow(c, h, l, atr, ta, sa, horizon, side):
    """Reference loop implementation."""
    n = len(c)
    sgn = 1 if side == "long" else -1
    y, ret, days = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - horizon):
        tgt, stp = c[i] + sgn * ta * atr[i], c[i] - sgn * sa * atr[i]
        res, r, d = 0, sgn * (c[i + horizon] / c[i] - 1), horizon
        for j in range(i + 1, i + horizon + 1):
            stop_hit = l[j] <= stp if sgn == 1 else h[j] >= stp
            tgt_hit = h[j] >= tgt if sgn == 1 else l[j] <= tgt
            if stop_hit:
                res, r, d = 0, sgn * (stp / c[i] - 1), j - i
                break
            if tgt_hit:
                res, r, d = 1, sgn * (tgt / c[i] - 1), j - i
                break
        y[i], ret[i], days[i] = res, r, d
    return y, ret, days


def test_vectorised_matches_loop():
    f = compute(_df(500, seed=7)).dropna(subset=["atr"])
    args = (f.close.values, f.high.values, f.low.values, f.atr.values)
    for side in ("long", "short"):
        for ta, sa in ((3.5, 1.75), (1.5, 2.0), (2.0, 1.0)):
            fast = barrier_outcomes(*args, ta, sa, 20, side)
            slow = _slow(*args, ta, sa, 20, side)
            for a, b in zip(fast, slow, strict=False):
                np.testing.assert_allclose(a, b, equal_nan=True)


def _slow_next_open(o, h, l, c, atr, ta, sa, horizon):
    n = len(c)
    y, ret, days = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n - horizon):
        e = o[i + 1]
        tgt, stp = e + ta * atr[i], e - sa * atr[i]
        res, r, d = 0, c[i + horizon] / e - 1, horizon
        for j in range(i + 1, i + horizon + 1):
            k = j - i
            if k > 1 and o[j] <= stp:
                res, r, d = 0, o[j] / e - 1, k
                break
            if l[j] <= stp:
                res, r, d = 0, stp / e - 1, k
                break
            if k > 1 and o[j] >= tgt:
                res, r, d = 1, o[j] / e - 1, k
                break
            if h[j] >= tgt:
                res, r, d = 1, tgt / e - 1, k
                break
        y[i], ret[i], days[i] = res, r, d
    return y, ret, days


def test_next_open_matches_loop():
    from app.features import next_open_outcomes

    df = _df(500, seed=11)
    rng = np.random.default_rng(3)
    df["open"] = df.close.shift().fillna(df.close) * (1 + rng.normal(0, 0.015, len(df)))  # real gaps
    df["high"] = df[["open", "close"]].max(axis=1) * 1.01
    df["low"] = df[["open", "close"]].min(axis=1) * 0.99
    f = compute(df).dropna(subset=["atr"])
    args = (f.open.values, f.high.values, f.low.values, f.close.values, f.atr.values)
    fast = next_open_outcomes(*args, 3.5, 2.5, 20)[:3]
    slow = _slow_next_open(*args, 3.5, 2.5, 20)
    for a, b in zip(fast, slow, strict=True):
        np.testing.assert_allclose(a, b, equal_nan=True)


def test_gap_below_stop_loses_more_than_stop():
    from app.features import next_open_outcomes

    o = np.array([100, 100, 80, 80] + [80] * 25, dtype=float)  # buy at 100, next day opens at 80
    h, l, c = o + 1, o - 1, o.copy()
    atr = np.full(len(o), 4.0)  # stop = 100 - 2.5*4 = 90
    y, ret, days, _ = next_open_outcomes(o, h, l, c, atr, 3.5, 2.5, 20)
    assert y[0] == 0 and round(ret[0], 3) == -0.2 and days[0] == 2  # filled at the 80 open, not the 90 stop
