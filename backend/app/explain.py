"""Plain-language extras for each pick: why the model ranks it, its risk level, a price cross-check and expiry."""
import datetime as dt

import numpy as np
import pandas as pd

# readable names for the model's features (see features.py)
LABELS = {
    "rsi14": "RSI (14 days)", "macd_hist_n": "MACD momentum", "adx14": "Trend strength (ADX)",
    "atr_pct": "Daily range (ATR %)", "dist_ema20": "Distance above 20-day average",
    "dist_ema50": "Distance above 50-day average", "dist_ema200": "Distance above 200-day average",
    "ema20_slope": "Slope of 20-day average", "vol_surge": "Volume, 5 days vs 50 days",
    "dist_52w_high": "Distance from 52-week high", "dist_52w_low": "Distance above 52-week low",
    "ret_5": "5-day return", "ret_20": "20-day return", "ret_60": "60-day return", "ret_120": "120-day return",
    "rs_20": "Return vs Nifty, 20 days", "rs_60": "Return vs Nifty, 60 days", "bb_width": "Bollinger band width",
    "bb_pos": "Position in Bollinger band", "volatility_20": "Daily volatility, 20 days",
    "up_days_10": "Up days in the last 10", "sector_ret_20": "Sector's 20-day return", "sector_rank": "Sector rank today",
    "mom_12_1": "12-month momentum (skipping last month)", "vol_ratio": "Volatility now vs 6 months",
    "days_since_high": "Time since 1-year high", "clv_10": "Closes near the day's high (10 days)",
    "dd_60": "Fall from 60-day high", "sharpe_60": "Smoothness of the 60-day trend",
    "rel_sector_20": "Return vs own sector, 20 days",
    "xr_ret_20": "Rank: 20-day return", "xr_ret_60": "Rank: 60-day return", "xr_rs_60": "Rank: strength vs Nifty",
    "xr_dist_52w_high": "Rank: closeness to 52-week high", "xr_vol_surge": "Rank: volume surge",
    "xr_atr_pct": "Rank: daily range", "xr_adx14": "Rank: trend strength", "xr_mom_12_1": "Rank: 12-month momentum",
    "xr_sharpe_60": "Rank: trend smoothness", "xr_liquidity": "Rank: liquidity",
}
PCT = {"atr_pct", "dist_ema20", "dist_ema50", "dist_ema200", "ema20_slope", "dist_52w_high", "dist_52w_low", "ret_5",
       "ret_20", "ret_60", "ret_120", "rs_20", "rs_60", "bb_width", "volatility_20", "sector_ret_20", "mom_12_1",
       "dd_60", "rel_sector_20", "macd_hist_n"}
UNSIGNED = {"atr_pct", "bb_width", "volatility_20"}  # sizes, not directions: no plus sign


def _fmt(feature: str, v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "–"
    if feature.startswith("xr_") or feature == "sector_rank":  # share of today's stocks this one is above
        return f"above {round(v * 100)}% of stocks"
    if feature in UNSIGNED:
        return f"{v * 100:.1f}%"
    if feature in PCT:
        return f"{v * 100:+.1f}%"
    if feature == "days_since_high":
        return f"{v * 12:.0f} months"
    if feature in ("vol_surge", "vol_ratio"):
        return f"{v:.2f}x"
    if feature == "up_days_10":
        return f"{v:.0f} of 10"
    return f"{v:.2f}" if abs(v) < 10 else f"{v:.0f}"


def why(model, features: list[str], rows: pd.DataFrame, top: int = 4, against: int = 2) -> list[list[dict]]:
    """Per-stock contribution of each feature to the model's score (LightGBM's own SHAP values, in log-odds).
    Returns, for every row, the strongest reasons for and against, with the stock's actual value."""
    X = rows[features].astype(float)
    contrib = model.booster_.predict(X, pred_contrib=True)[:, :-1]  # last column is the base value
    out = []
    for i in range(len(X)):
        c = contrib[i]
        order = np.argsort(c)
        pos = [j for j in order[::-1] if c[j] > 0][:top]
        neg = [j for j in order if c[j] < 0][:against]
        out.append([{"feature": features[j], "label": LABELS.get(features[j], features[j]),
                     "value": _fmt(features[j], float(X.iat[i, j])), "effect": round(float(c[j]), 3)}
                    for j in pos + neg])
    return out


def risk(levels: dict, official_event: dict | None, research: dict) -> dict:
    """Low / Medium / High from how far the stop is (a wide stop means a volatile stock) and known event risk."""
    stop_pct = float(levels.get("stop_pct") or 0)
    reasons = [f"stop is {stop_pct:.1f}% below the entry"]
    level = "low" if stop_pct < 5 else "medium" if stop_pct < 9 else "high"
    if official_event:
        level = "high"
        reasons.append(f"{official_event.get('purpose', 'corporate event')} on {official_event.get('date')}")
    if research.get("news_sentiment") == "negative" or research.get("red_flags"):
        level = "high"
        reasons.append("negative news or a red flag")
    return {"level": level, "reasons": reasons}


def next_session(day: str) -> str:
    """The first weekday after `day` (NSE holidays are not known here): the session a pick is meant for."""
    d = dt.date.fromisoformat(day) + dt.timedelta(days=1)
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d.isoformat()


def is_expired(valid_until: str | None, now: dt.datetime) -> bool:
    """A pick is bought at the open of `valid_until`; after that session has started the entry is gone."""
    if not valid_until:
        return False
    start = dt.datetime.combine(dt.date.fromisoformat(valid_until), dt.time(9, 15))
    return now >= start


def cross_check(symbol: str, day: str, close: float, bhav: pd.DataFrame | None, tolerance: float = 0.01) -> dict:
    """Yahoo's close against NSE's official bhavcopy close for the same day."""
    if bhav is None or symbol not in bhav.index:
        return {"status": "unavailable"}
    nse = float(bhav.loc[symbol, "close"])
    diff = close / nse - 1 if nse else 0.0
    return {"status": "ok" if abs(diff) <= tolerance else "mismatch", "nse_close": round(nse, 2),
            "yahoo_close": round(float(close), 2), "diff_pct": round(diff * 100, 2)}
