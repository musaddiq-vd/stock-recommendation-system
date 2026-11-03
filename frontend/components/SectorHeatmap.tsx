"use client";
import { useEffect, useState } from "react";
import { get } from "@/lib/api";

interface Sector {
  sector: string; stocks: number; ret_1d: number; ret_5d: number; ret_20d: number; ret_60d: number; rs_20d: number;
  breadth: number; flow_20d: number; flow_5d: number; turnover_surge: number; deliv_surge: number | null;
  turnover_cr: number; leaders: string[]; laggards: string[];
}
interface Heat { index: string; as_of: string; sectors: Sector[] }

type Col = { key: keyof Sector; label: string; scale: number; center?: number; fmt: (v: number) => string; hint: string };
const pct = (v: number) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`;
const COLS: Col[] = [
  { key: "flow_20d", label: "Money flow 20D", scale: 40, fmt: (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}`, hint: "Up-day turnover minus down-day turnover (−100…+100). + = money flowing in" },
  { key: "flow_5d", label: "Flow 5D", scale: 60, fmt: (v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}`, hint: "Same, last 5 sessions" },
  { key: "ret_1d", label: "1D", scale: 2, fmt: pct, hint: "Median return" },
  { key: "ret_5d", label: "1W", scale: 4, fmt: pct, hint: "Median return" },
  { key: "ret_20d", label: "1M", scale: 8, fmt: pct, hint: "Median return" },
  { key: "ret_60d", label: "3M", scale: 15, fmt: pct, hint: "Median return" },
  { key: "rs_20d", label: "vs Nifty 1M", scale: 6, fmt: pct, hint: "Median relative strength" },
  { key: "breadth", label: "% > 50 EMA", scale: 50, center: 50, fmt: (v) => `${v.toFixed(0)}%`, hint: "Breadth" },
  { key: "turnover_surge", label: "Turnover ×", scale: 0.6, center: 1, fmt: (v) => `${v.toFixed(2)}×`, hint: "5D vs 50D avg turnover" },
  { key: "deliv_surge", label: "Delivery ×", scale: 0.6, center: 1, fmt: (v) => `${v.toFixed(2)}×`, hint: "5D vs 50D delivery qty" },
];

function cellBg(v: number | null, c: Col) {
  if (v == null) return "transparent";
  const x = Math.max(-1, Math.min(1, (v - (c.center ?? 0)) / c.scale));
  const a = 0.12 + Math.abs(x) * 0.6;
  return x >= 0 ? `rgba(16,185,129,${a})` : `rgba(239,68,68,${a})`;
}

export default function SectorHeatmap({ index }: { index: string }) {
  const [data, setData] = useState<Heat | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    setData(null);
    get<Heat>(`/api/sectors?index=${index}`).then(setData).catch((e) => setErr(String(e)));
  }, [index]);

  if (err) return <div className="text-red-400 text-sm">{err}</div>;
  if (!data) return <div className="text-sm text-zinc-500 animate-pulse">Calculating sector data…</div>;
  const real = data.sectors.filter((s) => s.stocks >= 3);
  const inflow = real.filter((s) => s.flow_20d > 10).slice(0, 3).map((s) => s.sector);
  const outflow = real.filter((s) => s.flow_20d < -10).slice(-3).map((s) => s.sector);
  return (
    <div className="space-y-3">
      <div className="text-sm text-zinc-300">
        <b className="text-white">{data.index}</b> sectors · as of {data.as_of}.{" "}
        {inflow.length > 0 && <>Money flowing in: <b className="text-emerald-400">{inflow.join(", ")}</b>. </>}
        {outflow.length > 0 && <>Flowing out: <b className="text-red-400">{outflow.join(", ")}</b>.</>}
      </div>
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
            {data.sectors.map((s) => (
              <tr key={s.sector} className="border-t border-zinc-800/70">
                <td className="px-3 py-1.5 sticky left-0 bg-zinc-950 whitespace-nowrap">
                  <div className="text-zinc-100">{s.sector}</div>
                  <div className="text-[10px] text-zinc-500">{s.stocks} stocks · ₹{Math.round(s.turnover_cr).toLocaleString("en-IN")} cr/day</div>
                </td>
                {COLS.map((c) => {
                  const v = s[c.key] as number | null;
                  return (
                    <td key={c.key} className="px-2 py-1.5 text-right tabular-nums text-zinc-100" style={{ background: cellBg(v, c) }}>
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
