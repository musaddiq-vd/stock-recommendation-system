"""Exit rules and a graded market gate for the live Buy rule (survivorship-free, next-open entries, 0.3% cost).

Uses the walk-forward scores written by model_lab (column p<N> / pct<N> of the chosen training recipe).

    python -m app.research.trade_lab        # recipe 0 (live)
    python -m app.research.trade_lab 2      # recipe number 2 from model_lab

Pick on pre-2016, confirm on 2016+.
"""
import json
import logging
import sys

import numpy as np
import pandas as pd

from ..features import _atr
from .exits import managed_outcomes
from .experiment import portfolio
from .pit import load_prices
from .v2 import COST_REAL, HORIZON, OUT, SPLIT, STOP, TARGET

log = logging.getLogger(__name__)

# name: (target ATR, stop ATR, max days, extra rule)
EXITS = {
    "live 3.5/2.5, 20d": (TARGET, STOP, HORIZON, {}),
    "3.5/2.5, 15d": (TARGET, STOP, 15, {}),
    "3.5/2.5, 10d": (TARGET, STOP, 10, {}),
    "3.0/2.0, 20d": (3.0, 2.0, HORIZON, {}),
    "3.0/2.5, 20d": (3.0, 2.5, HORIZON, {}),
    "4.0/2.5, 20d": (4.0, 2.5, HORIZON, {}),
    "3.5/2.0, 20d": (TARGET, 2.0, HORIZON, {}),
    "live + sell on day 5 if below entry": (TARGET, STOP, HORIZON, {"time_stop": (5, 0.0)}),
    "live + sell on day 10 if below entry": (TARGET, STOP, HORIZON, {"time_stop": (10, 0.0)}),
    "live + sell on day 10 if below +1 ATR": (TARGET, STOP, HORIZON, {"time_stop": (10, 1.0)}),
    "live + sell on day 15 if below entry": (TARGET, STOP, HORIZON, {"time_stop": (15, 0.0)}),
}


def _gates(top: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Market-gate variants. size_ is the fraction of a normal slot."""
    nifty, b = top.mkt_above_200, top.breadth200
    both, one = nifty & (b > 0.5), nifty ^ (b > 0.5)
    return {
        "no gate": top,
        "live gate (Nifty & breadth > 50%)": top[both],
        "graded: full when both, half when one": top[both | one].assign(size_=np.where(both[both | one], 1.0, 0.5)),
        "graded: full when both, half when Nifty only": top[nifty].assign(size_=np.where(both[nifty], 1.0, 0.5)),
        "graded: full when both, half when breadth 40-50%": top[nifty & (b > 0.4)].assign(
            size_=np.where(both[nifty & (b > 0.4)], 1.0, 0.5)),
        "stricter: Nifty & breadth > 60%": top[nifty & (b > 0.6)],
    }


def _row(name, sig, rcol, dcol, calendar) -> list[dict]:
    rows = []
    sig = sig.dropna(subset=[rcol])
    for period, part in (("pre-2016", sig[sig.date < SPLIT]), ("2016+", sig[sig.date >= SPLIT]), ("all", sig)):
        r = portfolio(part, rcol, dcol, cost=COST_REAL, calendar=calendar)
        if r:
            rows.append({"variant": name, "period": period, "trades": r["trades"], "win_rate": r["win_rate"],
                         "profit_factor": r["profit_factor"], "avg_trade": r["avg_trade"], "cagr": r["cagr"],
                         "max_dd": r["max_dd"], "avg_days": round(float(part[dcol].mean()), 1)})
    return rows


def run(recipe: int = 0, source: str = "pit") -> dict:
    df = pd.read_parquet(OUT / f"model_lab_{source}.parquet")
    calendar = sorted(df.date.unique())
    top = df[df[f"pct{recipe}"] >= 0.98].assign(prob=lambda x: x[f"p{recipe}"])
    gate_rows = []
    for name, sig in _gates(top).items():
        gate_rows += _row(name, sig, "r_open", "d_open", calendar)

    live = top[top.mkt_above_200 & (top.breadth200 > 0.5)]
    prices = load_prices(set(live.symbol))
    extra = []
    for s, p in prices.items():
        atr = _atr(p)
        out = pd.DataFrame(index=p.index)
        for i, (t, st, hz, kw) in enumerate(EXITS.values()):
            out[f"r{i}"], out[f"d{i}"] = managed_outcomes(p.open, p.high, p.low, p.close, atr, t, st, hz, **kw)
        extra.append(out.assign(symbol=s).rename_axis("date").reset_index())
    live = live.merge(pd.concat(extra), on=["date", "symbol"], how="left")
    exit_rows = []
    for i, name in enumerate(EXITS):
        exit_rows += _row(name, live, f"r{i}", f"d{i}", calendar)
    out = {"recipe": recipe, "gates": gate_rows, "exits": exit_rows}
    (OUT / f"trade_lab_{source}.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    res = run(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
    pd.set_option("display.width", 220)
    for part in ("gates", "exits"):
        print(f"\n== {part} ==")
        print(pd.DataFrame(res[part]).to_string(index=False))
