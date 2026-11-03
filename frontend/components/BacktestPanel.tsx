"use client";
import { useEffect, useRef, useState } from "react";
import { ColorType, createChart } from "lightweight-charts";
import { get } from "@/lib/api";

interface Portfolio {
  threshold: number; slots: number; cost: number; total_return: number; max_drawdown: number; trades: number;
  win_rate: number | null; avg_trade: number | null; nifty_return: number | null;
  equity: { date: string; equity: number; nifty: number | null; drawdown: number }[];
  monthly: { month: string; ret: number }[];
}
interface Side {
  auc: number; test_from: string; test_to: string; trained_at: string; feature_set?: string;
  comparison?: Record<string, { auc: number; quality: number; by_threshold: { threshold: number; trades: number; win_rate: number; avg_ret: number }[] }>;
  portfolio?: Portfolio;
  backtest: { base_win_rate: number; by_threshold: { threshold: number; trades: number; win_rate: number; avg_ret: number; avg_days: number }[] };
  top_features: [string, number][];
}

const pct = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);

export default function BacktestPanel() {
  const [data, setData] = useState<{ long?: Side; short?: Side } | null>(null);
  const [side, setSide] = useState<"long" | "short">("long");
  useEffect(() => {
    get<{ long?: Side; short?: Side }>("/api/backtest").then(setData).catch(() => setData({}));
  }, []);
  const s = data?.[side];

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {(["long", "short"] as const).map((x) => (
          <button key={x} onClick={() => setSide(x)} className={`rounded-lg px-3 py-1.5 text-sm ${side === x ? (x === "long" ? "bg-emerald-700" : "bg-rose-700") + " text-white" : "text-zinc-400 border border-zinc-800"}`}>
            {x === "long" ? "Long (buy) model" : "Short model"}
          </button>
        ))}
      </div>
      {!data ? <div className="text-sm text-zinc-500">Loading…</div> : !s ? <div className="text-sm text-zinc-500">Model report not found.</div> : !s.portfolio ? (
        <div className="text-sm text-zinc-500">Portfolio backtest not available yet. It appears here after the next model retrain.</div>
      ) : (
        <>
          <p className="text-xs text-zinc-400">
            Out-of-sample test: <b>{s.test_from} → {s.test_to}</b> (data the model never saw during training). Portfolio: max {s.portfolio.slots} positions,
            har trade {Math.round(100 / s.portfolio.slots)}% capital, ML probability ≥ {s.portfolio.threshold} (top 2% signals), {(s.portfolio.cost * 100).toFixed(1)}% cost per trade.
            The AI agent filter is not included.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-2">
            <Box k="Strategy return" v={pct(s.portfolio.total_return)} good={s.portfolio.total_return >= 0} />
            <Box k="Nifty same period" v={pct(s.portfolio.nifty_return)} good={(s.portfolio.nifty_return ?? 0) >= 0} />
            <Box k="Max drawdown" v={pct(s.portfolio.max_drawdown)} good={false} />
            <Box k="Trades" v={String(s.portfolio.trades)} />
            <Box k="Win rate (after cost)" v={pct(s.portfolio.win_rate, 0).replace("+", "")} />
            <Box k="Avg per trade" v={pct(s.portfolio.avg_trade, 2)} good={(s.portfolio.avg_trade ?? 0) >= 0} />
          </div>
          <Charts p={s.portfolio} short={side === "short"} />
          <Monthly rows={s.portfolio.monthly} />
          {s.comparison && (
            <div className="rounded-xl border border-zinc-800 p-4 text-xs">
              <div className="font-semibold text-zinc-200 mb-2">Feature set comparison (same test period) · chosen: <span className="text-emerald-400">{s.feature_set}</span></div>
              <table className="w-full">
                <thead className="text-zinc-500"><tr><th className="text-left py-1">Set</th><th className="text-right">AUC</th><th className="text-right">Avg trade (≥100 trades buckets)</th><th className="text-right">Top-signal win rate</th></tr></thead>
                <tbody>
                  {Object.entries(s.comparison).map(([k, v]) => {
                    const top = [...v.by_threshold].reverse().find((r) => r.trades >= 100);
                    return (
                      <tr key={k} className="border-t border-zinc-800">
                        <td className="py-1 text-zinc-200">{k === "base" ? "Price/volume only" : "+ Delivery %, OI, PCR"}</td>
                        <td className="text-right">{v.auc}</td>
                        <td className="text-right">{pct(v.quality, 2)}</td>
                        <td className="text-right">{top ? `${(top.win_rate * 100).toFixed(0)}% (${top.trades})` : "–"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
          <div className="text-xs text-zinc-500">
            Top features: {s.top_features.slice(0, 8).map(([f]) => f).join(", ")} · trained {s.trained_at}. Past performance does not guarantee future returns.
          </div>
        </>
      )}
    </div>
  );
}

function Charts({ p, short }: { p: Portfolio; short: boolean }) {
  const eqRef = useRef<HTMLDivElement>(null);
  const ddRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!eqRef.current || !ddRef.current) return;
    const opts = {
      layout: { background: { type: ColorType.Solid, color: "transparent" }, textColor: "#a1a1aa" },
      grid: { vertLines: { color: "#27272a" }, horzLines: { color: "#27272a" } },
      rightPriceScale: { borderColor: "#3f3f46" }, timeScale: { borderColor: "#3f3f46" }, autoSize: true,
    };
    const c1 = createChart(eqRef.current, opts);
    const strat = c1.addLineSeries({ color: short ? "#f43f5e" : "#10b981", lineWidth: 2, title: "Strategy" });
    strat.setData(p.equity.map((e) => ({ time: e.date, value: e.equity * 100 })));
    const nifty = c1.addLineSeries({ color: "#a1a1aa", lineWidth: 1, title: "Nifty" });
    nifty.setData(p.equity.filter((e) => e.nifty != null).map((e) => ({ time: e.date, value: (e.nifty as number) * 100 })));
    c1.timeScale().fitContent();
    const c2 = createChart(ddRef.current, opts);
    const dd = c2.addAreaSeries({ lineColor: "#ef4444", topColor: "rgba(239,68,68,0.05)", bottomColor: "rgba(239,68,68,0.4)", lineWidth: 1 });
    dd.setData(p.equity.map((e) => ({ time: e.date, value: e.drawdown * 100 })));
    c2.timeScale().fitContent();
    return () => {
      c1.remove();
      c2.remove();
    };
  }, [p, short]);
  return (
    <div className="grid gap-3">
      <div className="rounded-xl border border-zinc-800 p-3">
        <div className="text-xs text-zinc-400 mb-1">Equity curve (start = 100) vs Nifty</div>
        <div ref={eqRef} className="h-[260px]" />
      </div>
      <div className="rounded-xl border border-zinc-800 p-3">
        <div className="text-xs text-zinc-400 mb-1">Drawdown % (fall from peak)</div>
        <div ref={ddRef} className="h-[140px]" />
      </div>
    </div>
  );
}

function Monthly({ rows }: { rows: { month: string; ret: number }[] }) {
  return (
    <div className="rounded-xl border border-zinc-800 p-3">
      <div className="text-xs text-zinc-400 mb-2">Monthly returns</div>
      <div className="grid grid-cols-3 sm:grid-cols-6 lg:grid-cols-7 gap-1.5">
        {rows.map((r) => {
          const a = 0.15 + Math.min(1, Math.abs(r.ret) / 0.06) * 0.6;
          return (
            <div key={r.month} className="rounded-md px-2 py-1.5 text-center" style={{ background: r.ret >= 0 ? `rgba(16,185,129,${a})` : `rgba(239,68,68,${a})` }}>
              <div className="text-[10px] text-zinc-200/80">{r.month}</div>
              <div className="text-sm font-semibold text-white">{pct(r.ret)}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Box({ k, v, good }: { k: string; v: string; good?: boolean }) {
  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2">
      <div className="text-[10px] uppercase text-zinc-500">{k}</div>
      <div className={`font-semibold ${good === undefined ? "text-zinc-100" : good ? "text-emerald-400" : "text-red-400"}`}>{v}</div>
    </div>
  );
}
