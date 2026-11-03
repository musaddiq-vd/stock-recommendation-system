"use client";
import { useCallback, useEffect, useState } from "react";
import { API, get } from "@/lib/api";
import { inr } from "@/lib/sizing";

interface Trade {
  id: number; symbol: string; side: string; qty: number; lots: number | null; entry: number; target: number;
  stop_loss: number; status: string; opened_on: string; exit_price: number | null; exit_date: string | null;
  pnl: number | null; last_price: number | null; unrealized?: number; unrealized_pct?: number;
}
interface Portfolio {
  capital: number; realized: number; unrealized: number; equity: number; return_pct: number;
  open: number; closed: number; win_rate: number | null; trades: Trade[];
}

const STATUS: Record<string, string> = {
  open: "text-sky-300", target: "text-emerald-400", stop: "text-red-400", timeout: "text-zinc-300", closed: "text-zinc-300",
};

export default function PaperPanel({ refreshKey }: { refreshKey: number }) {
  const [pf, setPf] = useState<Portfolio | null>(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    setBusy(true);
    try {
      setPf(await get<Portfolio>("/api/paper"));
    } finally {
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    load();
  }, [load, refreshKey]);

  async function act(id: number, how: "close" | "delete") {
    await fetch(`${API}/api/paper/${id}${how === "close" ? "/close" : ""}`, { method: how === "close" ? "POST" : "DELETE" });
    load();
  }

  if (!pf) return <div className="text-sm text-zinc-500">Loading paper portfolio…</div>;
  const pnl = pf.realized + pf.unrealized;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
        <Box k="Virtual equity" v={inr(pf.equity)} />
        <Box k="Total P&L" v={`${inr(pnl)} (${pf.return_pct > 0 ? "+" : ""}${pf.return_pct}%)`} cls={pnl >= 0 ? "text-emerald-400" : "text-red-400"} />
        <Box k="Realized" v={inr(pf.realized)} cls={pf.realized >= 0 ? "text-emerald-400" : "text-red-400"} />
        <Box k="Unrealized" v={inr(pf.unrealized)} cls={pf.unrealized >= 0 ? "text-emerald-400" : "text-red-400"} />
        <Box k="Win rate (closed)" v={pf.win_rate == null ? "–" : `${(pf.win_rate * 100).toFixed(0)}% of ${pf.closed}`} />
      </div>
      <div className="flex items-center justify-between">
        <div className="text-xs text-zinc-500">
          Trades auto-close at target, stop or the 20-day timeout (checked every 15 min during market hours). Prices are ~15 min delayed.
        </div>
        <button onClick={load} disabled={busy} className="text-xs rounded border border-zinc-700 px-2 py-1 text-zinc-300">{busy ? "Refreshing…" : "Refresh"}</button>
      </div>
      {pf.trades.length === 0 ? (
        <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center text-sm text-zinc-500">
          No paper trades yet. Open any pick and press <b className="text-zinc-300">Paper trade</b>.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-zinc-800">
          <table className="w-full text-sm">
            <thead className="bg-zinc-900 text-[11px] uppercase text-zinc-500">
              <tr>{["Stock", "Side", "Qty", "Entry", "Target / Stop", "Now / Exit", "P&L", "Status", ""].map((h) => <th key={h} className="px-3 py-2 text-left font-medium">{h}</th>)}</tr>
            </thead>
            <tbody>
              {pf.trades.map((t) => {
                const open = t.status === "open";
                const val = open ? t.unrealized ?? 0 : t.pnl ?? 0;
                return (
                  <tr key={t.id} className="border-t border-zinc-800">
                    <td className="px-3 py-2 font-medium text-white">{t.symbol}<div className="text-[10px] text-zinc-500">{t.opened_on}</div></td>
                    <td className={`px-3 py-2 ${t.side === "short" ? "text-rose-300" : "text-emerald-300"}`}>{t.side}</td>
                    <td className="px-3 py-2">{t.qty}{t.lots ? <span className="text-zinc-500"> ({t.lots} lot)</span> : null}</td>
                    <td className="px-3 py-2">₹{t.entry}</td>
                    <td className="px-3 py-2 text-xs"><span className="text-emerald-400">₹{t.target}</span> / <span className="text-red-400">₹{t.stop_loss}</span></td>
                    <td className="px-3 py-2">₹{open ? t.last_price ?? "–" : t.exit_price}{!open && <div className="text-[10px] text-zinc-500">{t.exit_date}</div>}</td>
                    <td className={`px-3 py-2 font-medium ${val >= 0 ? "text-emerald-400" : "text-red-400"}`}>
                      {inr(val)}{open && t.unrealized_pct != null && <div className="text-[10px]">{t.unrealized_pct}%</div>}
                    </td>
                    <td className={`px-3 py-2 uppercase text-xs ${STATUS[t.status]}`}>{t.status}</td>
                    <td className="px-3 py-2 text-right whitespace-nowrap">
                      {open && <button onClick={() => act(t.id, "close")} className="text-xs text-zinc-300 hover:text-white mr-2">Close</button>}
                      <button onClick={() => act(t.id, "delete")} className="text-xs text-zinc-500 hover:text-red-400">Delete</button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function Box({ k, v, cls = "text-zinc-100" }: { k: string; v: string; cls?: string }) {
  return (
    <div className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2">
      <div className="text-[10px] uppercase text-zinc-500">{k}</div>
      <div className={`font-semibold ${cls}`}>{v}</div>
    </div>
  );
}
