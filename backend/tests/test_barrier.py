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
            for a, b in zip(fast, slow):
                np.testing.assert_allclose(a, b, equal_nan=True)
