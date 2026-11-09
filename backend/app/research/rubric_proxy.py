"""Rule-based copy of the agent rubric's technical components, computable on any historical date.

It mirrors the agent scorecard (setup structure, momentum, volume/flow, risk-reward) without news,
fundamentals or LLM judgement, so it can be backtested over decades without look-ahead.
"""
import numpy as np
import pandas as pd


def setup_type(d: pd.DataFrame, side: str = "long") -> pd.Series:
    c, e20, e50, e200 = d.close, d.ema20, d.ema50, d.ema200
    if side == "short":
        return pd.Series(np.select([(c < e50) & (e50 < e200), (c > e20) & (d.rsi14 > 60)],
                                   ["trend", "fade"], "breakdown"), index=d.index)
    return pd.Series(np.select(
        [(c > e50) & (e50 > e200) & (d.dist_52w_high > -0.08), (c > e200) & (c < e20), (c < e50) & (d.rsi14 < 45)],
        ["trend", "pullback", "bounce"], "reversal"), index=d.index)


RUBRIC_MAX = {"structure": 20, "momentum": 15, "volume": 15, "risk_reward": 10}


def proxy_score(d: pd.DataFrame, side: str, stop_atr: float) -> pd.Series:
    """0-100 score: the four components below, rescaled from their 60-point total."""
    return proxy_components(d, side, stop_atr).sum(axis=1) / sum(RUBRIC_MAX.values()) * 100


def proxy_components(d: pd.DataFrame, side: str, stop_atr: float) -> pd.DataFrame:
    """Points per component (structure 0-20, momentum 0-15, volume 0-15, risk_reward 0-10)."""
    st = setup_type(d, side)
    sgn = 1 if side == "long" else -1
    # setup structure (0-1), judged for the setup type
    trend = ((sgn * d.dist_ema50 > 0).astype(float) + (sgn * d.dist_ema200 > 0) + (sgn * d.ema20_slope > 0)
             + (d.adx14 > 20)) / 4
    pull = ((d.close > d.ema200) & (d.dist_ema50 > -0.04) & d.rsi14.between(38, 55)).astype(float)
    exhaustion = (d.ret_5 > d.ret_20 / 4).astype(float)  # fall is slowing
    off_low = (d.close > d.low_20 * 1.02).astype(float)
    bounce = ((d.rsi14 < 38).astype(float) + exhaustion + off_low + (d.bb_pos < 0.25)) / 4
    fade = ((d.rsi14 > 65).astype(float) + (d.bb_pos > 0.8) + (d.dist_ema200 < 0)) / 3
    structure = np.select(
        [st == "trend", st == "pullback", st == "bounce", st == "fade"],
        [trend, pull, bounce, fade], (trend + bounce) / 2)
    # momentum: cross-sectional rank of relative strength in the trade direction
    mom = (d.xr_rs_60 + d.xr_ret_20) / 2
    mom = mom if side == "long" else 1 - mom
    # volume / flow
    vol = ((d.vol_surge > 1.0).astype(float) + ((d.up_days_10 >= 5) if side == "long" else (d.up_days_10 <= 5))) / 2
    # risk-reward: stop sits beyond the 20-day swing low (high for shorts is approximated by the stop distance)
    stop = d.close - sgn * stop_atr * d.atr
    rr = ((stop < d.low_20) if side == "long" else (d.atr / d.close < 0.04)).astype(float)
    return pd.DataFrame({
        "structure": 20 * pd.Series(structure, index=d.index).astype(float),
        "momentum": 15 * mom.astype(float),
        "volume": 15 * vol.astype(float),
        "risk_reward": 10 * rr.astype(float),
    }, index=d.index)
