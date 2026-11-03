"""Historical test of the LLM judgement on anonymised past signals.

The model sees no ticker, company, sector or date, and prices are rebased to 100, so it cannot recall what
happened next. Only the technical rubric components are scored (news and fundamentals are not available
as-of-date). Results show whether the LLM's conviction separates winners from losers.
"""
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from ..agents.llm import extract_json, run_agent
from .experiment import OUT, SCORES, load_dataset
from .history import load_all

log = logging.getLogger(__name__)

SYSTEM = """You are a professional swing trader. You will see an anonymised chart snapshot of a stock
(name, sector and dates are hidden; prices are rebased so the latest close = 100). A quant screener already
ranked it in the top 2% for a {side} swing trade held up to 20 trading days, with the stop and target given.
Score ONLY what the data shows, using the full range of each scale:
  - structure (0-20): does the price structure support this setup type?
  - momentum (0-15): strength of the move in the trade direction, RSI regime
  - volume_flow (0-15): volume behaviour around the recent move
  - risk_reward (0-10): is the stop beyond a logical level and is there room to the target?
Reply with ONLY JSON: {{"structure": n, "momentum": n, "volume_flow": n, "risk_reward": n, "reason": "<one sentence>"}}"""


def _snapshot(prices: dict, row, side: str, target: float, stop: float) -> dict:
    df = prices[row.symbol]
    hist = df[df.index <= row.date].tail(60)
    scale = 100 / float(hist.close.iloc[-1])
    vol = hist.volume / hist.volume.tail(50).mean()
    sgn = 1 if side == "long" else -1
    atr = float(row.atr) * scale
    return {
        "setup_type": row.setup, "direction": side,
        "last_60_closes": [round(float(x) * scale, 1) for x in hist.close],
        "last_60_relative_volume": [round(float(x), 2) for x in vol],
        "rsi14": round(float(row.rsi14), 1), "adx14": round(float(row.adx14), 1),
        "ema20": round(float(row.ema20) * scale, 1), "ema50": round(float(row.ema50) * scale, 1),
        "ema200": round(float(row.ema200) * scale, 1),
        "pct_from_52w_high": round(float(row.dist_52w_high) * 100, 1),
        "relative_strength_vs_index_60d_pct": round(float(row.rs_60) * 100, 1),
        "entry": 100.0, "target": round(100 + sgn * target * atr, 1), "stop": round(100 - sgn * stop * atr, 1),
    }


def run(side: str = "long", n: int = 200, target: float = 2.0, stop: float = 2.0, seed: int = 7) -> pd.DataFrame:
    from .rubric_proxy import setup_type

    data = load_dataset()
    scores = pd.read_parquet(str(SCORES).format(side=side))
    df = scores[scores.pct >= 0.98].merge(data, on=["date", "symbol"])
    k = f"{side}_t{target:g}_s{stop:g}"
    df = df[(df.date >= "2008-01-01") & df[f"r_{k}"].notna()]
    df["setup"] = setup_type(df, side).values
    sample = df.sample(n=min(n, len(df)), random_state=seed).rename(columns={f"r_{k}": "out_ret"})
    prices = load_all()

    def one(row):
        snap = _snapshot(prices, row, side, target, stop)
        msgs = [{"role": "user", "content": [{"text": json.dumps(snap)}]}]
        text = ""
        for ev in run_agent(SYSTEM.format(side=side.upper()), msgs, [], {}, max_steps=1, max_tokens=400):
            text = ev.get("text", "")
        try:
            j = extract_json(text)
            conv = sum(float(j.get(c, 0)) for c in ("structure", "momentum", "volume_flow", "risk_reward")) / 60 * 100
        except Exception:
            j, conv = {}, np.nan
        return {"date": row.date, "symbol": row.symbol, "setup": row.setup, "conviction": conv,
                "ret": float(row.out_ret), "win": float(row.out_ret) > 0,
                "reason": j.get("reason", "")}

    with ThreadPoolExecutor(6) as ex:
        res = pd.DataFrame(list(ex.map(one, sample.itertuples())))
    res.to_csv(OUT / f"llm_backtest_{side}.csv", index=False)
    return res


def summarize(res: pd.DataFrame) -> dict:
    from sklearn.metrics import roc_auc_score

    r = res.dropna(subset=["conviction"])
    out = {"n": len(r), "base_win_rate": round(float(r.win.mean()), 3), "base_avg_ret": round(float(r.ret.mean()), 4),
           "auc": round(float(roc_auc_score(r.win, r.conviction)), 3) if r.win.nunique() > 1 else None}
    for th in (50, 60, 65, 70, 75):
        s = r[r.conviction >= th]
        if len(s) >= 15:
            out[f"conv>={th}"] = {"n": len(s), "win_rate": round(float(s.win.mean()), 3), "avg_ret": round(float(s.ret.mean()), 4)}
    return out


if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO)
    side = sys.argv[1] if len(sys.argv) > 1 else "long"
    t, s = float(sys.argv[2]), float(sys.argv[3])
    print(json.dumps(summarize(run(side, target=t, stop=s)), indent=2))
