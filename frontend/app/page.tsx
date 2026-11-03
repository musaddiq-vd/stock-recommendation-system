"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import PaperPanel from "@/components/PaperPanel";
import BacktestPanel from "@/components/BacktestPanel";
import SectorHeatmap from "@/components/SectorHeatmap";
import PickCard from "@/components/PickCard";
import SettingsPanel from "@/components/SettingsPanel";
import StockDrawer from "@/components/StockDrawer";
import { get, Market, Pick, post, Scan, Tier } from "@/lib/api";
import { Settings } from "@/lib/sizing";

const UNIVERSES = ["NIFTY50", "NIFTY100", "NIFTY500"];
type ModelReport = { auc: number; test_from: string; test_to: string; backtest: { base_win_rate: number; by_threshold: { threshold: number; trades: number; win_rate: number; avg_ret: number; avg_days: number }[] } };

export default function Home() {
  const [universe, setUniverse] = useState("NIFTY100");
  const [sector, setSector] = useState("");
  const [mode, setMode] = useState<"news" | "technical">("news");
  const [topN, setTopN] = useState(12);
  const [sectors, setSectors] = useState<string[]>([]);
  const [market, setMarket] = useState<Market | null>(null);
  const [scan, setScan] = useState<Scan | null>(null);
  const [scanId, setScanId] = useState<number | null>(null);
  const [open, setOpen] = useState<Pick | null>(null);
  const [tierFilter, setTierFilter] = useState<Tier | "all">("all");
  const [report, setReport] = useState<ModelReport | null>(null);
  const [shortReport, setShortReport] = useState<ModelReport | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [tab, setTab] = useState<"picks" | "sectors" | "backtest" | "paper">("picks");
  const [paperKey, setPaperKey] = useState(0);
  const [err, setErr] = useState("");

  useEffect(() => {
    get<{ sectors: string[] }>("/api/universe").then((d) => setSectors(d.sectors)).catch((e) => setErr(String(e)));
    get<Market>("/api/market").then(setMarket).catch(() => {});
    get<ModelReport>("/api/model").then(setReport).catch(() => {});
    get<ModelReport>("/api/model?side=short").then(setShortReport).catch(() => {});
    get<Settings>("/api/settings").then(setSettings).catch(() => {});
    get<Scan>("/api/scans/latest").then((s) => s?.id && setScan(s)).catch(() => {});
  }, []);

  const poll = useCallback(async (id: number) => {
    const s = await get<Scan>(`/api/scans/${id}`);
    setScan(s);
    if (s.status === "running") setTimeout(() => poll(id), 2500);
    else setScanId(null);
  }, []);

  async function runScan() {
    setErr("");
    try {
      const { scan_id } = await post<{ scan_id: number }>("/api/scan", { universe, sector: sector || null, mode, top_n: topN });
      setScanId(scan_id);
      poll(scan_id);
    } catch (e) {
      setErr(String(e));
    }
  }

  const picks = useMemo(
    () => (scan?.picks || []).filter((p) => tierFilter === "all" || p.tier === tierFilter),
    [scan, tierFilter],
  );
  const longPicks = picks.filter((p) => p.side !== "short");
  const shortPicks = picks.filter((p) => p.side === "short");
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    (scan?.picks || []).forEach((p) => (c[p.tier] = (c[p.tier] || 0) + 1));
    return c;
  }, [scan]);
  const running = scan?.status === "running" || scanId !== null;
  const best = report?.backtest.by_threshold.find((r) => r.threshold === 0.4);
  const shortBest = shortReport?.backtest.by_threshold.find((r) => r.threshold === 0.4);

  return (
    <div className="flex h-screen overflow-hidden">
      <main className="flex-1 overflow-y-auto">
        <header className="sticky top-0 z-10 bg-zinc-950/90 backdrop-blur border-b border-zinc-800 px-6 py-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h1 className="text-xl font-bold text-white">AI Swing Trader <span className="text-emerald-400">· NSE</span></h1>
              <p className="text-xs text-zinc-500">ML screener + AI research agents · 15–20 day swing trades</p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {market && (
                <div className={`rounded-lg border px-3 py-1.5 text-xs ${market.regime === "bullish" ? "border-emerald-700 text-emerald-300" : market.regime === "bearish" ? "border-red-800 text-red-300" : "border-zinc-700 text-zinc-300"}`}>
                  NIFTY {market.nifty.toLocaleString("en-IN")} · 20d {market.ret_20d_pct > 0 ? "+" : ""}{market.ret_20d_pct}% ·
                  market <b className="uppercase">{market.regime}</b> · {market.date}
                </div>
              )}
              {market?.flows?.fii != null && (
                <div className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300">
                  {market.flows.date}: FII <Flow v={market.flows.fii} /> · DII <Flow v={market.flows.dii ?? 0} /> cr
                </div>
              )}
              <button onClick={() => setShowSettings(true)} className="rounded-lg border border-zinc-700 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-900">
                ⚙ Settings{settings?.telegram_configured ? " · 🔔" : ""}
              </button>
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-3 mt-4">
            <Field label="Universe">
              <div className="flex rounded-lg border border-zinc-800 overflow-hidden">
                {UNIVERSES.map((u) => (
                  <button key={u} onClick={() => setUniverse(u)} className={`px-3 py-1.5 text-sm ${universe === u ? "bg-emerald-600 text-white" : "text-zinc-400 hover:bg-zinc-900"}`}>
                    {u.replace("NIFTY", "Nifty ")}
                  </button>
                ))}
              </div>
            </Field>
            <Field label="Sector tag">
              <select value={sector} onChange={(e) => setSector(e.target.value)} className="rounded-lg bg-zinc-900 border border-zinc-800 px-2 py-1.5 text-sm">
                <option value="">All sectors</option>
                {sectors.map((s) => <option key={s}>{s}</option>)}
              </select>
            </Field>
            <Field label="Research mode">
              <div className="flex rounded-lg border border-zinc-800 overflow-hidden">
                <button onClick={() => setMode("news")} className={`px-3 py-1.5 text-sm ${mode === "news" ? "bg-sky-700 text-white" : "text-zinc-400 hover:bg-zinc-900"}`}>Technical + News/Web</button>
                <button onClick={() => setMode("technical")} className={`px-3 py-1.5 text-sm ${mode === "technical" ? "bg-sky-700 text-white" : "text-zinc-400 hover:bg-zinc-900"}`}>Technical only</button>
              </div>
            </Field>
            <Field label={`Stocks to research: ${topN}`}>
              <input type="range" min={3} max={30} value={topN} onChange={(e) => setTopN(+e.target.value)} className="accent-emerald-500 w-32" />
            </Field>
            <button onClick={runScan} disabled={running} className="rounded-lg bg-emerald-600 hover:bg-emerald-500 px-5 py-2 text-sm font-semibold text-white disabled:opacity-50">
              {running ? "Scanning…" : "Run AI Scan"}
            </button>
          </div>
          {err && <div className="mt-2 text-xs text-red-400">{err}</div>}
        </header>

        <div className="px-6 pt-4 flex gap-2 overflow-x-auto whitespace-nowrap">
          {(["picks", "sectors", "backtest", "paper"] as const).map((t) => (
            <button key={t} onClick={() => setTab(t)} className={`rounded-lg px-4 py-1.5 text-sm ${tab === t ? "bg-zinc-800 text-white" : "text-zinc-400 hover:bg-zinc-900"}`}>
              {{ picks: "AI Picks", sectors: "Sector Heatmap", backtest: "Backtest", paper: "Paper Portfolio" }[t]}
            </button>
          ))}
        </div>
        {tab !== "picks" ? (
          <div className="px-6 py-5">
            {tab === "paper" && <PaperPanel refreshKey={paperKey} />}
            {tab === "sectors" && <SectorHeatmap index={universe} />}
            {tab === "backtest" && <BacktestPanel />}
          </div>
        ) : (
        <div className="px-6 py-5 space-y-5">
          {market?.regime === "bearish" && (
            <div className="rounded-lg border border-red-900 bg-red-950/30 px-4 py-2 text-sm text-red-200">
              ⚠ Nifty is below its 200-day average (bearish regime). Scores are penalised; use smaller position sizes.
            </div>
          )}

          {scan && (
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="text-zinc-400">
                Scan #{scan.id} · {scan.universe}{scan.sector ? ` · ${scan.sector}` : ""} · {scan.mode === "news" ? "Technical + News" : "Technical"} · {scan.created_at.replace("T", " ")}
              </span>
              <span className="flex-1" />
              {(["all", "strong", "good", "watch", "avoid"] as const).map((t) => (
                <button key={t} onClick={() => setTierFilter(t)} className={`rounded-full px-3 py-1 text-xs border ${tierFilter === t ? "border-emerald-500 text-emerald-300" : "border-zinc-800 text-zinc-400"}`}>
                  {t === "all" ? `All (${scan.picks.length})` : `${t} (${counts[t] || 0})`}
                </button>
              ))}
            </div>
          )}

          {running && scan && (
            <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
              <div className="text-sm font-medium text-white mb-2 flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" /> Agents researching…
              </div>
              <div className="max-h-56 overflow-y-auto font-mono text-[11px] text-zinc-400 space-y-0.5">
                {[...scan.progress].reverse().map((p, i) => <div key={i}><span className="text-zinc-600">{p.t}</span> {p.msg}</div>)}
              </div>
            </div>
          )}
          {scan?.status === "error" && <div className="text-red-400 text-sm">Scan failed: {scan.error}</div>}

          {picks.length > 0 ? (
            <>
              {longPicks.length > 0 && (
                <section>
                  <h2 className="text-sm font-semibold text-zinc-200 mb-3">Buy (long) ideas · {longPicks.length}</h2>
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {longPicks.map((p) => <PickCard key={`L-${p.symbol}`} p={p} settings={settings} onOpen={() => setOpen(p)} />)}
                  </div>
                </section>
              )}
              {shortPicks.length > 0 && (
                <section className="pt-2">
                  <h2 className="text-sm font-semibold text-rose-300">Short-sell ideas (market {scan?.market?.regime}) · {shortPicks.length}</h2>
                  <p className="text-xs text-zinc-500 mb-3">
                    F&O stocks only: a 15–20 day short needs stock futures (cash-market shorts are intraday only). Losses are not capped, so always use the stop-loss.
                    {shortReport && <> Backtest: random short {(shortReport.backtest.base_win_rate * 100).toFixed(0)}% vs ML top signals {shortBest ? (shortBest.win_rate * 100).toFixed(0) : "–"}% win rate.</>}
                  </p>
                  <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
                    {shortPicks.map((p) => <PickCard key={`S-${p.symbol}`} p={p} settings={settings} onOpen={() => setOpen(p)} />)}
                  </div>
                </section>
              )}
            </>
          ) : (
            !running && (
              <div className="rounded-xl border border-dashed border-zinc-800 p-10 text-center text-zinc-500">
                Pick a universe and sector, then press <b className="text-zinc-300">Run AI Scan</b>. The agents will research every shortlisted stock in depth.
              </div>
            )
          )}

          {report && (
            <div className="rounded-xl border border-zinc-800 p-4 text-xs text-zinc-400">
              <div className="font-semibold text-zinc-200 mb-1">Model backtest (out-of-sample {report.test_from} → {report.test_to})</div>
              Random pick win rate: {(report.backtest.base_win_rate * 100).toFixed(0)}% ·
              {best && <> ML top ~1% signals: win rate {(best.win_rate * 100).toFixed(0)}%, avg {(best.avg_ret * 100).toFixed(1)}% per trade over {best.avg_days} days ({best.trades} trades)</>}
              <div className="mt-2 flex flex-wrap gap-3">
                <Legend cls="bg-emerald-500" t="High probability (score ≥ 85, agent says buy)" />
                <Legend cls="bg-emerald-800" t="Good (score ≥ 72)" />
                <Legend cls="bg-zinc-700" t="Watch" />
                <Legend cls="bg-red-800" t="Avoid" />
              </div>
              <div className="mt-2 text-zinc-500">Educational tool, not financial advice. Past backtest results do not guarantee future returns.</div>
            </div>
          )}
        </div>
        )}
      </main>

      <aside className={`${chatOpen ? "flex fixed inset-0 z-30" : "hidden"} lg:static lg:flex w-full lg:w-[400px] border-l border-zinc-800 bg-zinc-950`}>
        <ChatPanel />
      </aside>
      <button
        onClick={() => setChatOpen((o) => !o)}
        className={`lg:hidden fixed right-4 z-30 rounded-full bg-emerald-600 text-sm font-semibold text-white shadow-lg ${chatOpen ? "top-3 px-3 py-1.5" : "bottom-4 px-4 py-3"}`}
      >
        {chatOpen ? "✕ Close" : "💬 Chat"}
      </button>

      {open && (
        <StockDrawer pick={open} onClose={() => setOpen(null)} settings={settings} scanId={scan?.id}
          onPaper={() => setPaperKey((k) => k + 1)} />
      )}
      {showSettings && settings && (
        <SettingsPanel s={settings} onSaved={setSettings} onClose={() => setShowSettings(false)} />
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</span>
      {children}
    </label>
  );
}
function Legend({ cls, t }: { cls: string; t: string }) {
  return <span className="flex items-center gap-1.5"><span className={`h-2.5 w-2.5 rounded-sm ${cls}`} />{t}</span>;
}
function Flow({ v }: { v: number }) {
  return <b className={v >= 0 ? "text-emerald-400" : "text-red-400"}>{v >= 0 ? "+" : ""}{Math.round(v).toLocaleString("en-IN")}</b>;
}
