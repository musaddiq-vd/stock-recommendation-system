"use client";
import { useEffect, useRef, useState } from "react";
import { ColorType, createChart } from "lightweight-charts";
import { get } from "@/lib/api";

interface Stats { trades: number; win_rate: number; avg_trade: number; profit_factor: number; cagr: number; max_dd: number; losing_years: number; years: number }
interface Variant extends Stats { name: string; yearly: Record<string, number>; holdout: Stats | null; equity: { date: string; value: number }[] }
interface LlmBucket { n: number; win_rate: number; avg_ret: number }
interface Research {
  from: string; to: string; split: string; target_atr: number; stop_atr: number; breadth_min: number;
  survivorship_free?: boolean; symbols?: number; cost?: number; variants: Variant[];
  llm_test?: { n: number; base_win_rate: number; auc: number | null } & Record<string, LlmBucket | number | null>;
}

const pct = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
const COLORS = ["#71717a", "#38bdf8", "#a3e635", "#10b981"];

/** 22-year walk-forward results: each year scored by a model trained only on earlier years. */
export default function LongTermBacktest() {
  const [r, setR] = useState<Research | null>(null);
  const [err, setErr] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    get<Research>("/api/research").then(setR).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    if (!r || !ref.current) return;
    const chart = createChart(ref.current, {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#a1a1aa" },
      grid: { vertLines: { color: "#27272a" }, horzLines: { color: "#27272a" } },
      rightPriceScale: { borderColor: "#3f3f46", mode: 1 }, // log scale: decades of compounding stay readable
      timeScale: { borderColor: "#3f3f46" }, autoSize: true,
    });
    r.variants.forEach((v, i) => {
      const s = chart.addLineSeries({ color: COLORS[i], lineWidth: i === r.variants.length - 1 ? 2 : 1, title: v.name.split(" (")[0] });
      s.setData(v.equity.map((e) => ({ time: e.date, value: e.value * 100 })));
    });
    chart.timeScale().fitContent();
    return () => chart.remove();
  }, [r]);

  if (err) return <div className="text-xs text-zinc-500">22-year research not available yet.</div>;
  if (!r) return <div className="text-xs text-zinc-500">Loading 22-year walk-forward…</div>;
  const live = r.variants[r.variants.length - 1];
  const years = Object.keys(live.yearly);

  return (
    <div className="rounded-xl border border-emerald-900/60 bg-emerald-950/10 p-4 space-y-4">
      <div>
        <div className="text-sm font-semibold text-white">22-year walk-forward ({r.from.slice(0, 4)} → {r.to.slice(0, 4)})</div>
        <p className="text-xs text-zinc-400 mt-1">
          Every year is traded by a model trained only on earlier years. Live setting: target {r.target_atr}× ATR, stop {r.stop_atr}× ATR,
          top 2% of the ML ranking, only when Nifty and more than {Math.round(r.breadth_min * 100)}% of stocks are above their 200-day
          averages (chosen on pre-2016 data among six market filters).
          Bought at the next day&apos;s open with gap fills, {((r.cost ?? 0.003) * 100).toFixed(1)}% cost per trade.
          {r.survivorship_free && <> Universe: the {r.symbols} stocks that were in the top 500 at the time, including later-delisted ones.</>}
          {" "}Rules were chosen on data
          before {r.split.slice(0, 4)}; the &quot;since {r.split.slice(0, 4)}&quot; column is the out-of-sample check.
        </p>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="text-zinc-500">
            <tr>
              <th className="text-left py-1">Strategy</th><th className="text-right">Trades</th><th className="text-right">Win rate</th>
              <th className="text-right">Profit factor</th><th className="text-right">Avg trade</th><th className="text-right">Max drawdown</th>
              <th className="text-right">Losing years</th><th className="text-right">Win rate since {r.split.slice(0, 4)}</th>
            </tr>
          </thead>
          <tbody>
            {r.variants.map((v, i) => (
              <tr key={v.name} className={`border-t border-zinc-800 ${i === r.variants.length - 1 ? "text-emerald-300 font-semibold" : "text-zinc-300"}`}>
                <td className="py-1.5"><span className="inline-block h-2 w-2 rounded-full mr-2" style={{ background: COLORS[i] }} />{v.name}</td>
                <td className="text-right">{v.trades}</td>
                <td className="text-right">{pct(v.win_rate, 1).replace("+", "")}</td>
                <td className="text-right">{v.profit_factor}</td>
                <td className="text-right">{pct(v.avg_trade, 2)}</td>
                <td className="text-right">{pct(v.max_dd, 0)}</td>
                <td className="text-right">{v.losing_years}/{v.years}</td>
                <td className="text-right">{v.holdout ? `${pct(v.holdout.win_rate, 1).replace("+", "")} · PF ${v.holdout.profit_factor}` : "–"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <div className="text-xs text-zinc-400 mb-1">Equity, log scale (start = 100)</div>
        <div ref={ref} className="h-[260px]" />
      </div>

      <div>
        <div className="text-xs text-zinc-400 mb-1">Live setting, return by year</div>
        <div className="grid grid-cols-4 sm:grid-cols-8 lg:grid-cols-11 gap-1">
          {years.map((y) => {
            const v = live.yearly[y];
            const a = 0.15 + Math.min(1, Math.abs(v) / 0.3) * 0.6;
            return (
              <div key={y} className="rounded px-1 py-1 text-center" style={{ background: v >= 0 ? `rgba(16,185,129,${a})` : `rgba(239,68,68,${a})` }}>
                <div className="text-[10px] text-zinc-200/80">{y}</div>
                <div className="text-xs font-semibold text-white">{pct(v, 0)}</div>
              </div>
            );
          })}
        </div>
      </div>

      {r.llm_test && (
        <div className="text-xs text-zinc-400">
          <b className="text-zinc-200">LLM judgement test</b> on {r.llm_test.n} anonymised past signals (no name, date or real prices):
          base win rate {pct(r.llm_test.base_win_rate as number, 0).replace("+", "")}, AUC {String(r.llm_test.auc)}.{" "}
          {Object.entries(r.llm_test).filter(([k]) => k.startsWith("conv")).map(([k, v]) => {
            const b = v as LlmBucket;
            return <span key={k} className="mr-3">{k.replace("conv>=", "conviction ≥ ")}: {(b.win_rate * 100).toFixed(0)}% win ({b.n})</span>;
          })}
        </div>
      )}

      <p className="text-[11px] text-zinc-500">
        Caveats: the stock list is today&apos;s Nifty 500, so delisted losers are missing (survivorship bias), which flatters every row,
        including the random baseline. Compare strategies with each other, not with absolute returns. 0.2% cost per trade included.
      </p>
    </div>
  );
}
