"use client";
import { Pick, TIER_STYLE } from "@/lib/api";
import { inr, positionSize, Settings } from "@/lib/sizing";

export default function PickCard({ p, onOpen, settings }: { p: Pick; onOpen: () => void; settings: Settings | null }) {
  const s = TIER_STYLE[p.tier];
  const l = p.levels;
  const short = p.side === "short";
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
            {short && <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-rose-600 text-white whitespace-nowrap">Short ↓</span>}
            {p.official_event && (
              <span title={p.official_event.desc} className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-600/50 whitespace-nowrap">
                📅 {p.official_event.purpose.split("/")[0].replace("Financial ", "")} {p.official_event.date.slice(5)}
              </span>
            )}
          </div>
          <div className="text-xs text-zinc-400 truncate">{p.name}</div>
          <div className="text-[11px] text-zinc-500 mt-0.5">
            {p.sector} · {p.indices.filter((i) => i !== "NIFTY500").join(", ") || "NIFTY500"}
          </div>
        </div>
        <div className="text-right shrink-0">
          <div className={`text-2xl font-bold ${p.tier === "strong" || p.tier === "good" ? "text-emerald-400" : p.tier === "avoid" ? "text-red-400" : "text-zinc-300"}`}>
            {p.score.toFixed(0)}
          </div>
          <div className="text-[10px] text-zinc-500">score</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 mt-3 text-center">
        <Lvl k={short ? "Sell at" : "Entry"} v={l.entry} />
        <Lvl k="Target" v={l.target} sub={`${short ? "-" : "+"}${l.target_pct}%`} cls="text-emerald-400" />
        <Lvl k="Stop" v={l.stop_loss} sub={`${short ? "+" : "-"}${l.stop_pct}%`} cls="text-red-400" />
      </div>
      {short && p.lot_size && (
        <div className="mt-2 text-[11px] text-rose-300">
          Futures lot: {p.lot_size} shares · 1 lot ≈ ₹{Math.round(p.lot_size * l.entry).toLocaleString("en-IN")} exposure
        </div>
      )}
      {size && (
        <div className="mt-2 rounded-md bg-black/30 px-2 py-1.5 text-[11px] text-zinc-300">
          {size.qty > 0 ? (
            <>
              <b className="text-white">{size.lots ? `${size.lots} lot (${size.qty} sh)` : `${size.qty} shares`}</b>
              {" · "}{short ? "margin ~" : "invest "}{inr(size.invest)} · max loss <span className="text-red-400">{inr(size.maxLoss)}</span>
              {" · "}target <span className="text-emerald-400">+{inr(size.profit)}</span>
            </>
          ) : (
            <span className="text-amber-300">{size.note}</span>
          )}
        </div>
      )}

      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-3 text-[11px] text-zinc-400">
        <span>ML top {Math.max(1, 100 - p.ml_percentile)}%</span>
        <span>Agent: <b className="text-zinc-200">{p.research.verdict}</b> ({p.research.conviction})</span>
        <span>Hold {p.hold_days}d</span>
        {p.research.setup && <span className="text-zinc-300">{p.research.setup}</span>}
        <span>News: {p.research.news_sentiment}</span>
      </div>
      <p className="text-xs text-zinc-300 mt-2 line-clamp-3">{p.research.summary}</p>
      {p.outcome && p.outcome !== "open" && (
        <div className="mt-2 text-[11px] text-zinc-400">
          Result: <b>{p.outcome}</b> {p.outcome_ret != null && `(${(p.outcome_ret * 100).toFixed(1)}%)`}
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
