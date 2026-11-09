"use client";
import { useEffect, useMemo, useState } from "react";
import { get } from "@/lib/api";

type Quadrant = "leading" | "weakening" | "lagging" | "improving";
interface Member { symbol: string; ret_1d: number; ret_5d: number; ret_20d: number; ret_60d: number; flow_20d: number; turnover_cr: number }
interface Sector {
  sector: string; stocks: number; ret_1d: number; ret_5d: number; ret_20d: number; ret_60d: number; rs_20d: number;
  breadth: number; flow_20d: number; flow_5d: number; turnover_surge: number; deliv_surge: number | null;
  turnover_cr: number; leaders: string[]; laggards: string[]; members?: Member[];
  rotation?: { x: number; y: number }[]; quadrant?: Quadrant; prev_quadrant?: Quadrant;
}
interface Heat { index: string; as_of: string; sectors: Sector[] }

type Col = { key: keyof Sector; label: string; scale: number; center?: number; fmt: (v: number) => string; hint: string };
const pct = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const flowFmt = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(0)}`;
const COLS: Col[] = [
  { key: "flow_20d", label: "Money flow 20D", scale: 40, fmt: flowFmt, hint: "Up-day turnover minus down-day turnover (−100…+100). + = money flowing in" },
  { key: "flow_5d", label: "Flow 5D", scale: 60, fmt: flowFmt, hint: "Same, last 5 sessions" },
  { key: "ret_1d", label: "1D", scale: 2, fmt: pct, hint: "Median return" },
  { key: "ret_5d", label: "1W", scale: 4, fmt: pct, hint: "Median return" },
  { key: "ret_20d", label: "1M", scale: 8, fmt: pct, hint: "Median return" },
  { key: "ret_60d", label: "3M", scale: 15, fmt: pct, hint: "Median return" },
  { key: "rs_20d", label: "vs Nifty 1M", scale: 6, fmt: pct, hint: "Median relative strength" },
  { key: "breadth", label: "% > 50 EMA", scale: 50, center: 50, fmt: (v) => `${v.toFixed(0)}%`, hint: "Breadth" },
  { key: "turnover_surge", label: "Turnover ×", scale: 0.6, center: 1, fmt: (v) => `${v.toFixed(2)}×`, hint: "5D vs 50D avg turnover" },
  { key: "deliv_surge", label: "Delivery ×", scale: 0.6, center: 1, fmt: (v) => `${v.toFixed(2)}×`, hint: "5D vs 50D delivery qty" },
];
// Metrics for the tile view; each one exists for sectors and for their member stocks
const TILE_METRICS: { key: "ret_1d" | "ret_5d" | "ret_20d" | "ret_60d" | "flow_20d"; label: string; scale: number; stockScale: number; fmt: (v: number) => string }[] = [
  { key: "ret_1d", label: "1D", scale: 2, stockScale: 3, fmt: pct },
  { key: "ret_5d", label: "1W", scale: 4, stockScale: 6, fmt: pct },
  { key: "ret_20d", label: "1M", scale: 8, stockScale: 12, fmt: pct },
  { key: "ret_60d", label: "3M", scale: 15, stockScale: 20, fmt: pct },
  { key: "flow_20d", label: "Money flow", scale: 40, stockScale: 50, fmt: flowFmt },
];

const Q_STYLE: Record<Quadrant, { label: string; color: string; bg: string; hint: string }> = {
  leading: { label: "Leading", color: "#34d399", bg: "rgba(16,185,129,0.07)", hint: "Stronger than the market and still gaining" },
  weakening: { label: "Weakening", color: "#fbbf24", bg: "rgba(245,158,11,0.07)", hint: "Stronger than the market but losing pace" },
  lagging: { label: "Lagging", color: "#f87171", bg: "rgba(239,68,68,0.07)", hint: "Weaker than the market and still slipping" },
  improving: { label: "Improving", color: "#60a5fa", bg: "rgba(59,130,246,0.07)", hint: "Weaker than the market but catching up" },
};

function heat(v: number | null | undefined, scale: number, center = 0) {
  if (v == null) return "transparent";
  const x = Math.max(-1, Math.min(1, (v - center) / scale));
  const a = 0.12 + Math.abs(x) * 0.6;
  return x >= 0 ? `rgba(16,185,129,${a})` : `rgba(239,68,68,${a})`;
}

type View = "tiles" | "rotation" | "table";

export default function SectorHeatmap({ index }: { index: string }) {
  const [data, setData] = useState<Heat | null>(null);
  const [err, setErr] = useState("");
  const [view, setView] = useState<View>("tiles");
  const [metric, setMetric] = useState(2);
  const [open, setOpen] = useState<string | null>(null);
  useEffect(() => {
    setData(null);
    setErr("");
    setOpen(null);
    get<Heat>(`/api/sectors?index=${index}`).then(setData).catch((e) => setErr(String(e)));
  }, [index]);

  if (err) return <div className="text-red-400 text-sm">{err}</div>;
  if (!data) return <div className="text-sm text-zinc-500 animate-pulse">Calculating sector data…</div>;
  if (!data.sectors.length) return <div className="text-sm text-zinc-500">No sector data yet.</div>;
  const real = data.sectors.filter((s) => s.stocks >= 3);
  const inflow = real.filter((s) => s.flow_20d > 10).slice(0, 3).map((s) => s.sector);
  const outflow = real.filter((s) => s.flow_20d < -10).slice(-3).map((s) => s.sector);
  const moved = real.filter((s) => s.quadrant && s.prev_quadrant && s.quadrant !== s.prev_quadrant
    && (s.quadrant === "leading" || s.quadrant === "improving"));

  return (
    <div className="space-y-3">
      <div className="text-sm text-zinc-300 space-y-1">
        <div>
          <b className="text-white">{data.index}</b> sectors · as of {data.as_of}.{" "}
          {inflow.length > 0 && <>Money flowing in: <b className="text-emerald-400">{inflow.join(", ")}</b>. </>}
          {outflow.length > 0 && <>Flowing out: <b className="text-red-400">{outflow.join(", ")}</b>.</>}
        </div>
        {moved.length > 0 && (
          <div className="text-xs text-zinc-400">
            Rotation in the last 5 weeks:{" "}
            {moved.map((s, i) => (
              <span key={s.sector}>
                {i > 0 && " · "}
                <b className="text-zinc-200">{s.sector}</b> moved into{" "}
                <b style={{ color: Q_STYLE[s.quadrant!].color }}>{Q_STYLE[s.quadrant!].label}</b>
              </span>
            ))}
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <div className="flex rounded-lg border border-zinc-800 p-0.5 text-xs">
          {(["tiles", "rotation", "table"] as View[]).map((v) => (
            <button key={v} onClick={() => setView(v)}
              className={`px-3 py-1 rounded-md capitalize ${view === v ? "bg-zinc-700 text-white" : "text-zinc-400 hover:text-zinc-200"}`}>
              {v === "rotation" ? "Rotation map" : v}
            </button>
          ))}
        </div>
        {view === "tiles" && (
          <div className="flex flex-wrap gap-1 text-xs">
            {TILE_METRICS.map((m, i) => (
              <button key={m.key} onClick={() => setMetric(i)}
                className={`px-2 py-1 rounded-md border ${metric === i ? "border-emerald-600 text-emerald-300" : "border-zinc-800 text-zinc-400"}`}>
                {m.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {view === "tiles" && <Tiles sectors={real} small={data.sectors.filter((s) => s.stocks < 3)} metric={metric} open={open} setOpen={setOpen} />}
      {view === "rotation" && <RotationMap sectors={real} />}
      {view === "table" && <Table sectors={data.sectors} />}
    </div>
  );
}

function Tiles({ sectors, small, metric, open, setOpen }: {
  sectors: Sector[]; small: Sector[]; metric: number; open: string | null; setOpen: (s: string | null) => void;
}) {
  const m = TILE_METRICS[metric];
  const total = sectors.reduce((a, s) => a + s.turnover_cr, 0) || 1;
  const sel = sectors.find((s) => s.sector === open);
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-1">
        {sectors.map((s) => {
          const share = s.turnover_cr / total;
          const v = s[m.key];
          return (
            <button key={s.sector} onClick={() => setOpen(open === s.sector ? null : s.sector)}
              title={`${s.sector}: ${s.stocks} stocks, ₹${Math.round(s.turnover_cr).toLocaleString("en-IN")} cr/day`}
              className={`rounded-lg p-2 text-left min-w-[96px] min-h-[64px] transition hover:brightness-125 ${open === s.sector ? "ring-2 ring-white/70" : ""}`}
              style={{ flexGrow: Math.max(1, Math.round(share * 100)), flexBasis: `${Math.max(96, Math.sqrt(share) * 520)}px`, background: heat(v, m.scale) }}>
              <div className="text-xs font-semibold text-white truncate">{s.sector}</div>
              <div className="text-base font-bold tabular-nums text-white">{m.fmt(v)}</div>
              {s.quadrant && <div className="text-[10px]" style={{ color: Q_STYLE[s.quadrant].color }}>● {Q_STYLE[s.quadrant].label}</div>}
            </button>
          );
        })}
      </div>
      {sel && <SectorDetail s={sel} metric={metric} />}
      {!sel && <p className="text-[11px] text-zinc-500">Tile size = daily turnover. Click a sector to see its stocks. {small.length > 0 && `${small.length} sectors with fewer than 3 stocks are only in the table.`}</p>}
    </div>
  );
}

function SectorDetail({ s, metric }: { s: Sector; metric: number }) {
  const m = TILE_METRICS[metric];
  const members = s.members ?? [];
  const total = members.reduce((a, x) => a + x.turnover_cr, 0) || 1;
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-3 space-y-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div className="text-sm font-semibold text-white">{s.sector} <span className="text-xs font-normal text-zinc-400">· {s.stocks} stocks · {s.breadth.toFixed(0)}% above 50-day EMA</span></div>
        {s.quadrant && <div className="text-xs" style={{ color: Q_STYLE[s.quadrant].color }}>{Q_STYLE[s.quadrant].label}: {Q_STYLE[s.quadrant].hint}</div>}
      </div>
      <div className="flex flex-wrap gap-1">
        {members.map((x) => (
          <div key={x.symbol} className="rounded-md px-2 py-1 min-w-[78px]"
            style={{ flexGrow: Math.max(1, Math.round((x.turnover_cr / total) * 60)), background: heat(x[m.key], m.stockScale) }}
            title={`₹${x.turnover_cr.toLocaleString("en-IN")} cr/day`}>
            <div className="text-[11px] font-semibold text-white truncate">{x.symbol}</div>
            <div className="text-xs tabular-nums text-white">{m.fmt(x[m.key])}</div>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-zinc-500">Largest {members.length} stocks by turnover; size = turnover. Colors show a move, not a recommendation.</p>
    </div>
  );
}

function RotationMap({ sectors }: { sectors: Sector[] }) {
  const [hover, setHover] = useState<string | null>(null);
  const pts = useMemo(() => sectors.filter((s) => s.rotation?.length), [sectors]);
  if (!pts.length) return <div className="text-sm text-zinc-500">Not enough history for the rotation map.</div>;
  // Each axis is scaled to today's points so one sector's long trail cannot squeeze the rest; trails are clipped.
  const latest = pts.map((s) => s.rotation![s.rotation!.length - 1]);
  const spanX = Math.max(1.5, ...latest.map((p) => Math.abs(p.x - 100))) * 1.25;
  const spanY = Math.max(1, ...latest.map((p) => Math.abs(p.y - 100))) * 1.25;
  const W = 640, H = 440, P = 28;
  const sx = (x: number) => P + ((x - (100 - spanX)) / (2 * spanX)) * (W - 2 * P);
  const sy = (y: number) => H - P - ((y - (100 - spanY)) / (2 * spanY)) * (H - 2 * P);
  // Nudge labels apart so clustered sectors stay readable
  const placed: { x: number; y: number }[] = [];
  const labelY = new Map<string, number>();
  [...pts].sort((a, b) => sy(latest[pts.indexOf(a)].y) - sy(latest[pts.indexOf(b)].y)).forEach((s) => {
    const p = latest[pts.indexOf(s)];
    const x = sx(p.x), y0 = sy(p.y) + 4;
    const y = [0, 12, -12, 24, -24, 36, -36].map((d) => y0 + d)
      .find((y) => !placed.some((q) => Math.abs(q.y - y) < 11 && Math.abs(q.x - x) < 120)) ?? y0;
    placed.push({ x, y });
    labelY.set(s.sector, y);
  });
  const short = (n: string) => (n.length > 22 ? `${n.slice(0, 20)}…` : n);
  const cx = sx(100), cy = sy(100);
  const quads: [Quadrant, number, number, number, number][] = [
    ["improving", P, P, cx - P, cy - P], ["leading", cx, P, W - P - cx, cy - P],
    ["lagging", P, cy, cx - P, H - P - cy], ["weakening", cx, cy, W - P - cx, H - P - cy],
  ];
  return (
    <div className="space-y-2">
      <div className="rounded-xl border border-zinc-800 bg-zinc-950 p-1 max-w-3xl">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto" role="img" aria-label="Sector rotation map">
          {quads.map(([q, x, y, w, h]) => (
            <g key={q}>
              <rect x={x} y={y} width={w} height={h} fill={Q_STYLE[q].bg} />
              <text x={q === "improving" || q === "lagging" ? x + 8 : x + w - 8} y={q === "improving" || q === "leading" ? y + 16 : y + h - 8}
                textAnchor={q === "improving" || q === "lagging" ? "start" : "end"} fontSize="12" fontWeight="600" fill={Q_STYLE[q].color} opacity={0.8}>
                {Q_STYLE[q].label}
              </text>
            </g>
          ))}
          <line x1={cx} x2={cx} y1={P} y2={H - P} stroke="#3f3f46" />
          <line x1={P} x2={W - P} y1={cy} y2={cy} stroke="#3f3f46" />
          <text x={W - P} y={H - 8} textAnchor="end" fontSize="10" fill="#71717a">Relative strength vs the market →</text>
          <text x={10} y={P - 10} fontSize="10" fill="#71717a">↑ Momentum of that strength</text>
          <clipPath id="rrg-plot"><rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} /></clipPath>
          {pts.map((s) => {
            const tr = s.rotation!;
            const last = tr[tr.length - 1];
            const col = Q_STYLE[s.quadrant ?? "lagging"].color;
            const dim = hover && hover !== s.sector;
            return (
              <g key={s.sector} opacity={dim ? 0.15 : 1} onMouseEnter={() => setHover(s.sector)} onMouseLeave={() => setHover(null)}
                onClick={() => setHover(hover === s.sector ? null : s.sector)} style={{ cursor: "pointer" }}>
                <title>{`${s.sector}: ${Q_STYLE[s.quadrant ?? "lagging"].label}`}</title>
                <g clipPath="url(#rrg-plot)">
                  <polyline points={tr.map((p) => `${sx(p.x)},${sy(p.y)}`).join(" ")} fill="none" stroke={col} strokeWidth={1.5} opacity={0.6} />
                  {tr.slice(0, -1).map((p, i) => <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={2} fill={col} opacity={0.5} />)}
                </g>
                <circle cx={sx(last.x)} cy={sy(last.y)} r={5.5} fill={col} stroke="#09090b" strokeWidth={1.5} />
                <text x={sx(last.x) > W - 150 ? sx(last.x) - 8 : sx(last.x) + 8} y={labelY.get(s.sector)}
                  textAnchor={sx(last.x) > W - 150 ? "end" : "start"} fontSize="11" fill="#e4e4e7">{short(s.sector)}</text>
              </g>
            );
          })}
        </svg>
      </div>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        {(["leading", "weakening", "lagging", "improving"] as Quadrant[]).map((q) => (
          <div key={q} className="rounded-lg border border-zinc-800 p-2">
            <div className="font-semibold" style={{ color: Q_STYLE[q].color }}>{Q_STYLE[q].label}</div>
            <div className="text-zinc-500">{Q_STYLE[q].hint}</div>
            <div className="text-zinc-300 mt-1">{pts.filter((s) => s.quadrant === q).map((s) => s.sector).join(", ") || "–"}</div>
          </div>
        ))}
      </div>
      <p className="text-[11px] text-zinc-500">
        Each line is a sector&apos;s path over the last 5 weeks (big dot = today). Sectors usually rotate clockwise:
        Improving → Leading → Weakening → Lagging. &ldquo;Market&rdquo; = all stocks in the index, equal weight. Hover or tap to highlight one.
      </p>
    </div>
  );
}

function Table({ sectors }: { sectors: Sector[] }) {
  return (
    <div className="space-y-2">
      <div className="overflow-x-auto rounded-xl border border-zinc-800">
        <table className="w-full text-xs">
          <thead className="bg-zinc-900 text-zinc-400">
            <tr>
              <th className="px-3 py-2 text-left sticky left-0 bg-zinc-900">Sector</th>
              {COLS.map((c) => <th key={c.key} title={c.hint} className="px-2 py-2 text-right font-medium whitespace-nowrap cursor-help">{c.label}</th>)}
              <th className="px-3 py-2 text-left">Leaders (1M)</th>
            </tr>
          </thead>
          <tbody>
            {sectors.map((s) => (
              <tr key={s.sector} className="border-t border-zinc-800/70">
                <td className="px-3 py-1.5 sticky left-0 bg-zinc-950 whitespace-nowrap">
                  <div className="text-zinc-100">{s.sector}</div>
                  <div className="text-[10px] text-zinc-500">{s.stocks} stocks · ₹{Math.round(s.turnover_cr).toLocaleString("en-IN")} cr/day</div>
                </td>
                {COLS.map((c) => {
                  const v = s[c.key] as number | null;
                  return (
                    <td key={c.key} className="px-2 py-1.5 text-right tabular-nums text-zinc-100" style={{ background: heat(v, c.scale, c.center) }}>
                      {v == null ? "–" : c.fmt(v)}
                    </td>
                  );
                })}
                <td className="px-3 py-1.5 text-zinc-400 whitespace-nowrap">{s.leaders.join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-zinc-500">
        Money flow = turnover on up days minus turnover on down days over 20 sessions (turnover-weighted). Green = buying, red = selling. Hover a column header for details.
      </p>
    </div>
  );
}
