"use client";
import { useEffect, useRef, useState } from "react";
import { createChart, ColorType, LineStyle } from "lightweight-charts";
import { get, Pick, post, TIER_STYLE } from "@/lib/api";
import { inr, positionSize, Settings } from "@/lib/sizing";
import { GATE_HINT } from "./PickCard";

type Candle = { time: string; open: number; high: number; low: number; close: number; volume: number };
// rule-based rubric, shown for context only (it did not improve the survivorship-free backtest)
const RUBRIC_MAX: Record<string, number> = { structure: 20, momentum: 15, volume: 15, risk_reward: 10 };

export default function StockDrawer({ pick, onClose, settings, scanId, onPaper }: {
  pick: Pick; onClose: () => void; settings: Settings | null; scanId?: number; onPaper: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [fund, setFund] = useState<Record<string, unknown>>({});
  const size = positionSize(pick, settings);
  const [qty, setQty] = useState(size?.qty || 1);
  const [paperMsg, setPaperMsg] = useState("");

  async function paperTrade() {
    setPaperMsg("");
    try {
      const q = Math.max(1, Math.floor(qty));
      // the server opens at the current price, so a trade taken days after the scan stays honest
      const t = await post<{ entry: number; opened_on: string }>("/api/paper", {
        symbol: pick.symbol, qty: q, target: pick.levels.target, stop_loss: pick.levels.stop_loss, scan_id: scanId,
      });
      setPaperMsg(`Paper trade open: ${q} shares at ₹${t.entry} (${t.opened_on}) ✓`);
      onPaper();
    } catch (e) {
      setPaperMsg(`⚠️ ${e instanceof Error ? e.message : e}`);
    }
  }

  useEffect(() => {
    let chart: ReturnType<typeof createChart> | null = null;
    let cancelled = false;
    get<{ candles: Candle[]; fundamentals: Record<string, unknown> }>(`/api/stock/${pick.symbol}`).then((d) => {
      if (cancelled || !ref.current) return;
      setFund(d.fundamentals || {});
      chart = createChart(ref.current, {
        layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#a1a1aa" },
        grid: { vertLines: { color: "#27272a" }, horzLines: { color: "#27272a" } },
        rightPriceScale: { borderColor: "#3f3f46" },
        timeScale: { borderColor: "#3f3f46" },
        autoSize: true,
      });
      const s = chart.addCandlestickSeries({
        upColor: "#10b981", downColor: "#ef4444", borderVisible: false,
        wickUpColor: "#10b981", wickDownColor: "#ef4444",
      });
      s.setData(d.candles);
      const L = pick.levels;
      s.createPriceLine({ price: L.target, color: "#10b981", lineStyle: LineStyle.Dashed, title: "Target", lineWidth: 1 });
      s.createPriceLine({ price: L.entry, color: "#a1a1aa", lineStyle: LineStyle.Dotted, title: "Entry", lineWidth: 1 });
      s.createPriceLine({ price: L.stop_loss, color: "#ef4444", lineStyle: LineStyle.Dashed, title: "Stop", lineWidth: 1 });
      chart.timeScale().fitContent();
    });
    return () => {
      cancelled = true;
      chart?.remove();
    };
  }, [pick]);

  const r = pick.research;
  const t = TIER_STYLE[pick.tier];
  const f = (k: string, pct = false) => {
    const v = fund[k] as number | undefined;
    if (v == null) return "–";
    return pct ? `${(v * 100).toFixed(1)}%` : typeof v === "number" ? v.toLocaleString("en-IN", { maximumFractionDigits: 2 }) : String(v);
  };

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/60" onClick={onClose}>
      <div className="w-full max-w-3xl h-full overflow-y-auto bg-zinc-950 border-l border-zinc-800 p-6" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-2xl font-bold text-white">{pick.symbol}</h2>
              <span className={`text-xs px-2 py-0.5 rounded ${t.badge}`}>{t.label}</span>
            </div>
            <div className="text-sm text-zinc-400">{pick.name} · {pick.sector} · data as of {pick.as_of}</div>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-white text-2xl leading-none">×</button>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-4 text-sm">
          <Stat k="Final score" v={pick.score.toFixed(0)} />
          <Stat k="Market gate" v={pick.gates?.market === undefined ? "–" : pick.gates.market ? "open" : "closed"} />
          <Stat k="ML rank" v={`Top ${Math.max(1, 100 - pick.ml_percentile)}%`} />
          <Stat k="Target-hit rate*" v={`${(pick.hist_win_rate * 100).toFixed(0)}%`} />
        </div>

        <div ref={ref} className="mt-4 w-full h-[320px]" />

        <div className="grid grid-cols-3 gap-2 mt-2 text-sm">
          <Stat k="Entry (last close)" v={`₹${pick.levels.entry}`} />
          <Stat k={`Target (+${pick.levels.target_pct}%)`} v={`₹${pick.levels.target}`} cls="text-emerald-400" />
          <Stat k={`Stop (-${pick.levels.stop_pct}%)`} v={`₹${pick.levels.stop_loss}`} cls="text-red-400" />
        </div>

        <div className="mt-4 rounded-xl border border-zinc-800 bg-zinc-900/60 p-4">
          <div className="text-sm font-semibold text-zinc-200">Position size & paper trade</div>
          {pick.tier !== "strong" && (
            <p className="text-xs text-amber-300 mt-1">
              Not a Buy call ({pick.tier}): the size below is only for a practice paper trade.
            </p>
          )}
          {size && (
            <p className="text-xs text-zinc-400 mt-1">
              Capital {inr(+settings!.capital)} · risk {settings!.risk_pct}% = max loss {inr(size.riskAmt)} per trade.{" "}
              {size.qty > 0 ? (
                <>{pick.tier === "strong" ? "Suggested" : "Size"}: <b className="text-white">{size.qty} shares</b>, invest {inr(size.invest)},
                  stop hit = <span className="text-red-400">-{inr(size.maxLoss)}</span>, target = <span className="text-emerald-400">+{inr(size.profit)}</span></>
              ) : <span className="text-amber-300">{size.note}</span>}
            </p>
          )}
          <div className="flex flex-wrap items-center gap-2 mt-3">
            <label className="text-xs text-zinc-400">Qty (shares)</label>
            <input type="number" min={1} step={1} value={qty}
              onChange={(e) => setQty(+e.target.value)}
              className="w-28 rounded-lg bg-zinc-950 border border-zinc-700 px-2 py-1 text-sm text-white" />
            <button onClick={paperTrade} className="rounded-lg bg-sky-700 hover:bg-sky-600 px-4 py-1.5 text-sm font-semibold text-white">
              Paper trade (buy)
            </button>
            <span className="text-xs text-emerald-400">{paperMsg}</span>
          </div>
        </div>

        {pick.rubric_parts && (
          <Section title={`Price-structure rubric: ${pick.rubric_score}/100 (context only, not a gate)`}>
            <div className="space-y-1.5">
              {Object.entries(pick.rubric_parts).map(([k, v]) => {
                const max = RUBRIC_MAX[k] ?? 15;
                const f = v / max;
                return (
                  <div key={k} className="grid grid-cols-[110px_1fr_48px] items-center gap-2 text-xs">
                    <span className="text-zinc-400">{k.replace("_", " ")}</span>
                    <div className="h-2 rounded bg-zinc-800 overflow-hidden">
                      <div className={`h-full ${f >= 0.65 ? "bg-emerald-500" : f >= 0.45 ? "bg-amber-500" : "bg-red-500"}`} style={{ width: `${f * 100}%` }} />
                    </div>
                    <span className="text-right text-zinc-300">{v}/{max}</span>
                  </div>
                );
              })}
            </div>
          </Section>
        )}

        {pick.gates && (
          <Section title="Gates (all must pass for Buy)">
            <ul className="text-xs space-y-1">
              {Object.entries(pick.gates).map(([g, ok]) => (
                <li key={g} className={ok ? "text-emerald-300" : "text-zinc-500"}>{ok ? "✓" : "✗"} <b>{g}</b>: {GATE_HINT[g]}</li>
              ))}
            </ul>
          </Section>
        )}

        <Section title="Research agent (news, filings, events)">
          <p className="text-zinc-300">{r.summary}</p>
          <p className="text-zinc-400 text-xs mt-2">
            News sentiment: <b className="text-zinc-200">{r.news_sentiment}</b> · Event risk: <b className="text-zinc-200">{r.event_risk}</b>
          </p>
          {r.red_flags && r.red_flags.length > 0 && (
            <div className="mt-2 rounded-lg border border-red-900 bg-red-950/40 px-3 py-2 text-xs text-red-200">
              🚩 Red flags: {r.red_flags.join(" · ")}
            </div>
          )}
        </Section>
        <div className="grid sm:grid-cols-2 gap-4">
          <Section title="Bull case"><List items={r.bull_case} cls="text-emerald-300" /></Section>
          <Section title="Bear case"><List items={r.bear_case} cls="text-red-300" /></Section>
        </div>
        <Section title="Fundamentals">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
            <Stat k="P/E" v={f("trailingPE")} /><Stat k="P/B" v={f("priceToBook")} />
            <Stat k="ROE" v={f("returnOnEquity", true)} /><Stat k="Debt/Eq" v={f("debtToEquity")} />
            <Stat k="Rev growth" v={f("revenueGrowth", true)} /><Stat k="EPS growth" v={f("earningsGrowth", true)} />
            <Stat k="Analyst" v={f("recommendationKey")} /><Stat k="Analyst target" v={f("targetMeanPrice")} />
          </div>
        </Section>

        <Section title="Technicals">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-sm">
            {Object.entries(pick.technicals)
              .filter(([k]) => !["symbol", "date"].includes(k))
              // older scans stored nested groups; flatten them into tiles
              .flatMap(([k, v]): [string, unknown][] =>
                v && typeof v === "object" ? Object.entries(v as Record<string, unknown>) : [[k, v]])
              .map(([k, v]) => (
                <Stat key={k} k={String(k).replaceAll("_", " ")} v={v == null ? "–" : String(v)} />
              ))}
          </div>
        </Section>

        {r.steps && r.steps.length > 0 && (
          <Section title={`Agent research trail (${r.steps.length} steps)`}>
            <ol className="text-xs text-zinc-400 space-y-1 list-decimal ml-4">
              {r.steps.map((s, i) => (
                <li key={i}><span className="text-zinc-200">{s.tool}</span> {Object.values(s.input).join(", ")}</li>
              ))}
            </ol>
          </Section>
        )}
        {r.sources && r.sources.length > 0 && (
          <Section title="Sources">
            <ul className="text-xs space-y-1">
              {r.sources.map((u) => (
                <li key={u}><a href={u} target="_blank" className="text-sky-400 hover:underline break-all">{u}</a></li>
              ))}
            </ul>
          </Section>
        )}
        <p className="text-[11px] text-zinc-500 mt-6">
          * Share of similar ML signals in the last 12 months that reached the full target before the stop within 20 days,
          bought at the next open. The rest ended at the stop or at the 20-day limit (often still in profit).
          For education only, not financial advice.
        </p>
      </div>
    </div>
  );
}

function Stat({ k, v, cls = "text-zinc-100" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2">
      <div className="text-[10px] uppercase text-zinc-500 truncate">{k}</div>
      <div className={`font-medium ${cls}`}>{v}</div>
    </div>
  );
}
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-5">
      <h3 className="text-sm font-semibold text-zinc-200 mb-2">{title}</h3>
      {children}
    </div>
  );
}
function List({ items, cls }: { items: string[]; cls: string }) {
  return <ul className={`text-sm space-y-1 list-disc ml-4 ${cls}`}>{(items || []).map((x, i) => <li key={i}>{x}</li>)}</ul>;
}
