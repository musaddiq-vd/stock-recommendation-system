"use client";
import { Pick, TIER_STYLE, TierStat } from "@/lib/api";
import { inr, positionSize, Settings } from "@/lib/sizing";

export const GATE_HINT: Record<string, string> = {
  ml_top: "In the top 2% of the daily ML ranking of ~480 stocks",
  market: "Nifty above its 200-day average and more than half of stocks above theirs (worst drawdown -66% → -23% in the honest test)",
  regime: "Nifty above its 200-day average (older scans)",
  rubric: "Rule-based rubric score ≥ 70 (older scans)",
  agent: "Research agent found no red flags, no negative news and no results inside the holding window",
};

const OUTCOME_STYLE: Record<string, string> = {
  target: "text-emerald-400", stop: "text-red-400", timeout: "text-zinc-300", open: "text-sky-300",
};

export default function PickCard({ p, onOpen, settings, stat }: {
  p: Pick; onOpen: () => void; settings: Settings | null; stat?: TierStat;
}) {
  const s = TIER_STYLE[p.tier];
  const l = p.levels;
  const size = positionSize(p, settings);
  return (
    <button
      onClick={onOpen}
      className={`text-left rounded-xl border p-4 transition hover:-translate-y-0.5 hover:brightness-110 ${s.card}`}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="font-semibold text-lg text-white">{p.symbol}</span>
            <span className={`text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded ${s.badge}`}>{s.label}</span>
            {p.official_event && (
              <span title={p.official_event.desc} className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-600/50 whitespace-nowrap">
                📅 {p.official_event.purpose.split("/")[0].replace("Financial ", "")} {p.official_event.date.slice(5)}
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-400 truncate">{p.name}</div>
          <div className="text-[11px] text-zinc-500 mt-0.5">
            {p.sector} · {p.indices.filter((i) => i !== "NIFTY500").join(", ") || "NIFTY500"} · prices as of {p.as_of}
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className={`text-2xl font-bold ${p.tier === "strong" || p.tier === "good" ? "text-emerald-400" : p.tier === "avoid" ? "text-red-400" : "text-zinc-300"}`}>
            {p.score.toFixed(0)}
          </div>
          <div className="text-[10px] text-zinc-500">score</div>
        </div>
      </div>

      {stat && (
        <div className="mt-2 text-[11px] text-emerald-200/90" title="22-year walk-forward result of picks in this tier, after 0.2% costs">
          22-yr test incl. delisted stocks: {(stat.win_rate * 100).toFixed(0)}% trades profitable · profit factor {stat.profit_factor}{" "}
          · worst drawdown {(stat.max_dd * 100).toFixed(0)}%
        </div>
      )}

      <div className="grid grid-cols-3 gap-2 mt-3 text-center">
        <Lvl k="Entry" v={l.entry} />
        <Lvl k="Target" v={l.target} sub={`+${l.target_pct}%`} cls="text-emerald-400" />
        <Lvl k="Stop" v={l.stop_loss} sub={`-${l.stop_pct}%`} cls="text-red-400" />
      </div>
      {size && (
        <div className="mt-2 rounded-md bg-black/30 px-2 py-1.5 text-[11px] text-zinc-300">
          {size.qty > 0 ? (
            <>
              <b className="text-white">{size.qty} shares</b> · invest {inr(size.invest)} · max loss{" "}
              <span className="text-red-400">{inr(size.maxLoss)}</span> · target <span className="text-emerald-400">+{inr(size.profit)}</span>
            </>
          ) : (
            <span className="text-amber-300">{size.note}</span>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 mt-3 text-[11px] text-zinc-400">
        <span>ML top {Math.max(1, 100 - p.ml_percentile)}%</span>
        {p.gates && (
          <span className="flex gap-1">
            {Object.entries(p.gates).map(([g, ok]) => (
              <span key={g} title={GATE_HINT[g]} className={`rounded px-1 ${ok ? "bg-emerald-900/60 text-emerald-300" : "bg-zinc-800 text-zinc-500 line-through"}`}>
                {g}
              </span>
            ))}
          </span>
        )}
        <span>News: {p.research.news_sentiment}</span>
        <span>Hold up to {p.hold_days.split("-").pop()}d</span>
      </div>
      <p className="text-xs text-zinc-300 mt-2 line-clamp-3">{p.research.summary}</p>
      {p.outcome && (
        <div className="mt-2 text-[11px] text-zinc-400">
          Result so far: <b className={OUTCOME_STYLE[p.outcome] || ""}>{p.outcome}</b>{" "}
          {p.outcome_ret != null && `(${p.outcome_ret > 0 ? "+" : ""}${(p.outcome_ret * 100).toFixed(1)}%)`}
        </div>
      )}
    </button>
  );
}

function Lvl({ k, v, sub, cls = "text-zinc-100" }: { k: string; v: number; sub?: string; cls?: string }) {
  return (
    <div className="rounded-md bg-black/30 py-1.5">
      <div className="text-[10px] text-zinc-500">{k}</div>
      <div className={`text-sm font-medium ${cls}`}>₹{v.toLocaleString("en-IN")}</div>
      {sub && <div className={`text-[10px] ${cls}`}>{sub}</div>}
    </div>
  );
}
