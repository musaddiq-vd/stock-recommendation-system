"use client";
import { useEffect, useRef, useState } from "react";
import { ColorType, createChart } from "lightweight-charts";
import { get } from "@/lib/api";
import LongTermBacktest from "./LongTermBacktest";

interface Portfolio {
  threshold: number; slots: number; cost: number; total_return: number; max_drawdown: number; trades: number;
  win_rate: number | null; avg_trade: number | null; nifty_return: number | null;
  equity: { date: string; equity: number; nifty: number | null; drawdown: number }[];
  monthly: { month: string; ret: number }[];
}
interface Report {
  auc: number; test_from: string; test_to: string; trained_at: string; portfolio?: Portfolio;
  top_features: [string, number][];
}
interface TierRecord { closed: number; open: number; win_rate: number | null; target_rate: number | null; avg_ret: number | null }
interface Track {
  version: string;
  earlier_version_picks: number;
  tiers: Record<string, TierRecord>;
  agent_check?: Partial<Record<"cleared" | "vetoed", TierRecord>>;
  recent: { symbol: string; as_of: string; tier: string; outcome: string; outcome_ret: number | null; outcome_days: number | null }[];
}

const pct = (v: number | null | undefined, d = 1) => (v == null ? "–" : `${v > 0 ? "+" : ""}${(v * 100).toFixed(d)}%`);
const TIER_ORDER = ["strong", "watch", "avoid"];
const TIER_NAME: Record<string, string> = { strong: "Buy", good: "Good (older rules)", watch: "Watch", avoid: "Avoid" };

export default function BacktestPanel() {
  const [s, setS] = useState<Report | null | undefined>(undefined);
  const [track, setTrack] = useState<Track | null>(null);
  useEffect(() => {
    get<Report>("/api/backtest").then(setS).catch(() => setS(null));
    get<Track>("/api/performance").then(setTrack).catch(() => setTrack(null));
  }, []);

  return (
    <div className="space-y-4">
      <TrackRecord t={track} />
      <LongTermBacktest />
      <div className="text-sm font-semibold text-zinc-200 pt-2">Latest model: last 12 months out-of-sample</div>
      {s === undefined ? <div className="text-sm text-zinc-500">Loading…</div> : !s ? <div className="text-sm text-zinc-500">Model report not found.</div> : !s.portfolio ? (
        <div className="text-sm text-zinc-500">Portfolio backtest not available yet. It appears here after the next model retrain.</div>
      ) : (
        <>
          <p className="text-xs text-zinc-400">
            Out-of-sample test: <b>{s.test_from} → {s.test_to}</b> (data the model never saw during training). Portfolio: max {s.portfolio.slots} positions,
            {" "}{Math.round(100 / s.portfolio.slots)}% of capital per trade, ML probability ≥ {s.portfolio.threshold} (top 2% signals), {(s.portfolio.cost * 100).toFixed(1)}% cost per trade.
            Today&apos;s Nifty 500 only (survivors) and no breadth or agent gates; the 22-year section above is the honest test.
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-6 gap-2">
            <Box k="Strategy return" v={pct(s.portfolio.total_return)} good={s.portfolio.total_return >= 0} />
            <Box k="Nifty same period" v={pct(s.portfolio.nifty_return)} good={(s.portfolio.nifty_return ?? 0) >= 0} />
            <Box k="Max drawdown" v={pct(s.portfolio.max_drawdown)} good={false} />
            <Box k="Trades" v={String(s.portfolio.trades)} />
            <Box k="Profitable (after cost)" v={pct(s.portfolio.win_rate, 0).replace("+", "")} />
            <Box k="Avg per trade" v={pct(s.portfolio.avg_trade, 2)} good={(s.portfolio.avg_trade ?? 0) >= 0} />
          </div>
          <Charts p={s.portfolio} />
          <Monthly rows={s.portfolio.monthly} />
          <div className="text-xs text-zinc-500">
            Top features: {s.top_features.slice(0, 8).map(([f]) => f).join(", ")} · trained {s.trained_at}. Past performance does not guarantee future returns.
          </div>
        </>
      )}
    </div>
  );
}

/** What actually happened to past recommendations (no backtest, no assumptions). */
function TrackRecord({ t }: { t: Track | null }) {
  if (!t) return null;
  const tiers = TIER_ORDER.filter((k) => t.tiers[k]);
  return (
    <div className="rounded-xl border border-sky-900/60 bg-sky-950/10 p-4 space-y-3">
      <div>
        <div className="text-sm font-semibold text-white">Live track record</div>
        <p className="text-xs text-zinc-400 mt-1">
          Real outcomes of this app&apos;s own recommendations, checked against market prices: target hit, stop hit, or closed at the
          20-day limit. Each stock-day counts once. A trade is profitable if it closed above entry (before costs).
          Only picks made with the current scoring rules (v{t.version}) are counted
          {t.earlier_version_picks > 0 && <> · {t.earlier_version_picks} picks from earlier rule sets are excluded</>}.
        </p>
      </div>
      {tiers.length === 0 ? (
        <div className="text-xs text-zinc-500">No recommendations tracked yet.</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {tiers.map((k) => {
            const r = t.tiers[k];
            return (
              <div key={k} className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-xs">
                <div className="text-zinc-300 font-semibold">{TIER_NAME[k] || k}</div>
                <div className="text-zinc-400 mt-1">{r.closed} closed · {r.open} still open</div>
                <div className="text-zinc-200 mt-1">
                  {r.closed ? <>Profitable {(r.win_rate! * 100).toFixed(0)}% · avg {pct(r.avg_ret, 1)}</> : "Waiting for results"}
                </div>
              </div>
            );
          })}
        </div>
      )}
      {t.agent_check && (t.agent_check.cleared || t.agent_check.vetoed) && (
        <div className="rounded-lg border border-zinc-800 p-3 text-xs">
          <div className="text-zinc-200 font-semibold">Agent veto check</div>
          <div className="text-zinc-500 mt-0.5">
            Picks that passed the ML and breadth gates: does the agent&apos;s veto actually avoid losers?
            The veto earns its place only if vetoed picks do clearly worse than cleared ones over many trades.
          </div>
          <div className="grid grid-cols-2 gap-2 mt-2">
            {(["cleared", "vetoed"] as const).map((k) => {
              const r = t.agent_check?.[k];
              return (
                <div key={k} className="rounded bg-zinc-900 px-2 py-1.5">
                  <div className="text-zinc-300">{k === "cleared" ? "Agent cleared" : "Agent vetoed"}</div>
                  <div className="text-zinc-400">
                    {r ? <>{r.closed} closed · {r.open} open{r.closed ? <> · profitable {(r.win_rate! * 100).toFixed(0)}% · avg {pct(r.avg_ret, 1)}</> : null}</> : "none yet"}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
      {t.recent.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="text-zinc-500">
              <tr><th className="text-left py-1">Stock</th><th className="text-left">Recommended</th><th className="text-left">Tier</th><th className="text-left">Outcome</th><th className="text-right">Return</th><th className="text-right">Days</th></tr>
            </thead>
            <tbody>
              {t.recent.map((r) => (
                <tr key={`${r.symbol}-${r.as_of}`} className="border-t border-zinc-800 text-zinc-300">
                  <td className="py-1 text-white">{r.symbol}</td><td>{r.as_of}</td><td>{TIER_NAME[r.tier] || r.tier}</td>
                  <td className={r.outcome === "target" ? "text-emerald-400" : r.outcome === "stop" ? "text-red-400" : ""}>{r.outcome}</td>
                  <td className={`text-right ${(r.outcome_ret ?? 0) >= 0 ? "text-emerald-400" : "text-red-400"}`}>{pct(r.outcome_ret, 1)}</td>
                  <td className="text-right">{r.outcome_days ?? "–"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {t.recent.length > 0 && t.recent.length < 30 && (
        <div className="text-[11px] text-zinc-500">Too few closed trades to judge yet; treat these numbers as early evidence only.</div>
      )}
    </div>
  );
}

function Charts({ p }: { p: Portfolio }) {
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
    const strat = c1.addLineSeries({ color: "#10b981", lineWidth: 2, title: "Strategy" });
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
  }, [p]);
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
