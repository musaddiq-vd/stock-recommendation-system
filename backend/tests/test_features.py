import numpy as np
import pandas as pd

from app.features import NEW_PRICE_FEATURES, compute, label


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
    cols = ["rsi14", "adx14", "atr_pct", "dist_ema200", "vol_surge", "ret_20", "bb_pos", *NEW_PRICE_FEATURES]
    pd.testing.assert_frame_equal(full[cols].iloc[:300], cut[cols])


def test_label_matches_manual_barrier():
    f = label(compute(_df(seed=3)), entry="close")
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


def test_new_features_are_filled_and_ranked():
    from app.features import BASE_FEATURES, add_cross_sectional

    frames = []
    for i, sector in enumerate(["IT", "IT", "Bank"]):
        f = compute(_df(seed=i)).iloc[[-1]].assign(symbol=f"S{i}", sector=sector, date=pd.Timestamp("2023-07-03"))
        frames.append(f)
    d = add_cross_sectional(pd.concat(frames, ignore_index=True))
    # days_since_high stays empty until a stock makes a 1-year high; LightGBM handles that, as in research
    assert d[[c for c in BASE_FEATURES if c != "days_since_high"]].notna().all().all()
    assert d.xr_liquidity.between(0, 1).all() and d.days_since_high.dropna().ge(0).all()


def test_placeholder_symbols_are_dropped(tmp_path, monkeypatch):
    from app import universe

    monkeypatch.setattr(universe, "UNIVERSE_DIR", tmp_path)
    pd.DataFrame({"symbol": ["TCS", "DUMMYHEG"], "name": ["TCS", "Dummy"], "sector": ["IT", "IT"]}).to_csv(
        tmp_path / "NIFTY50.csv", index=False)
    assert universe.get_index("NIFTY50").symbol.tolist() == ["TCS"]
