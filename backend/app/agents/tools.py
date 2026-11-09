"""Tools the agents can call: market data, fundamentals, web search, news."""
import logging
import urllib.parse

import feedparser
import yfinance as yf

from ..data.events import get_corporate_events
from ..data.prices import benchmark, load, yf_ticker
from ..features import compute, trade_levels

log = logging.getLogger(__name__)


def get_technicals(symbol: str) -> dict:
    df = load(symbol)
    if df is None:
        return {"error": f"No price data for {symbol}"}
    f = compute(df, benchmark())
    r = f.iloc[-1]
    closes = f.close.tail(30)
    return {
        "symbol": symbol, "date": str(f.index[-1].date()), "close": round(float(r.close), 2),
        "rsi14": round(float(r.rsi14), 1), "adx14": round(float(r.adx14), 1),
        "ema20": round(float(r.ema20), 2), "ema50": round(float(r.ema50), 2), "ema200": round(float(r.ema200), 2),
        "pct_from_52w_high": round(float(r.dist_52w_high) * 100, 1),
        "return_20d_pct": round(float(r.ret_20) * 100, 1), "return_60d_pct": round(float(r.ret_60) * 100, 1),
        "rel_strength_vs_nifty_60d_pct": round(float(r.rs_60) * 100, 1),
        "volume_surge_5d_vs_50d": round(float(r.vol_surge), 2), "atr_pct": round(float(r.atr_pct) * 100, 2),
        "last_30_closes": [round(float(x), 1) for x in closes],
        "trade_levels": trade_levels(r),
    }


def get_fundamentals(symbol: str) -> dict:
    t = yf.Ticker(yf_ticker(symbol))
    info = t.info or {}
    keys = ["longName", "sector", "industry", "marketCap", "trailingPE", "forwardPE", "priceToBook",
            "returnOnEquity", "debtToEquity", "revenueGrowth", "earningsGrowth", "profitMargins",
            "heldPercentInstitutions", "recommendationKey", "targetMeanPrice", "dividendYield"]
    out = {k: info.get(k) for k in keys if info.get(k) is not None}
    try:
        cal = t.calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            out["next_earnings_dates"] = [str(d) for d in cal["Earnings Date"]]
    except Exception:
        pass
    return out


def web_search(query: str, max_results: int = 6) -> list[dict]:
    from ddgs import DDGS

    try:
        res = DDGS().text(query, region="in-en", max_results=max_results)
        out = [{"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")} for r in res]
        out = [r for r in out if r["url"] or r["snippet"]]
        return out or [{"note": "no results, try a different query"}]
    except Exception as e:
        return [{"error": f"search failed: {e}"}]


def get_news(query: str, max_results: int = 10) -> list[dict]:
    """Google News RSS (India edition) + DuckDuckGo news, newest first."""
    out = []
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": f"{query} when:30d", "hl": "en-IN", "gl": "IN", "ceid": "IN:en"})
    try:
        for e in feedparser.parse(url).entries[:max_results]:
            out.append({"title": e.title, "published": e.get("published"), "source": e.get("source", {}).get("title"),
                        "url": e.link})
    except Exception as e:
        log.warning("rss failed: %s", e)
    if len(out) < 3:
        try:
            from ddgs import DDGS

            for r in DDGS().news(query, region="in-en", max_results=max_results):
                out.append({"title": r.get("title"), "published": r.get("date"), "source": r.get("source"),
                            "url": r.get("url"), "snippet": r.get("body")})
        except Exception:
            pass
    return out[:max_results]


SPECS = [
    {"name": "get_technicals", "description": "Latest technical indicators, recent closes and ATR-based swing trade levels for an NSE symbol (e.g. RELIANCE).",
     "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "get_fundamentals", "description": "Valuation, growth, profitability, analyst view and next earnings date for an NSE symbol.",
     "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
    {"name": "get_news", "description": "Recent (30 days) Indian news headlines for a company or topic.",
     "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "web_search", "description": "General web search (India region) for deeper research: results, orders, management commentary, sector trends.",
     "schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_corporate_events", "description": "OFFICIAL NSE calendar of announced board meetings (results, dividends, fund raising) for a symbol. Prefer this over web search for result dates.",
     "schema": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}},
]
HANDLERS = {"get_technicals": get_technicals, "get_fundamentals": get_fundamentals,
            "get_news": get_news, "web_search": web_search, "get_corporate_events": get_corporate_events}
