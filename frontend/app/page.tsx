"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import ChatPanel from "@/components/ChatPanel";
import PaperPanel from "@/components/PaperPanel";
import BacktestPanel from "@/components/BacktestPanel";
import SectorHeatmap from "@/components/SectorHeatmap";
import PickCard from "@/components/PickCard";
import SettingsPanel from "@/components/SettingsPanel";
import StockDrawer from "@/components/StockDrawer";
import Login from "@/components/Login";
import { AUTH_EVENT, get, getToken, Market, Pick, post, Scan, setToken, Tier, TierStats, tierStatsFrom } from "@/lib/api";
import { Settings } from "@/lib/sizing";

const UNIVERSES = ["NIFTY50", "NIFTY100", "NIFTY500"];

/** Shows the login screen when the API asks for one, otherwise the dashboard. */
export default function Home() {
  const [state, setState] = useState<"checking" | "login" | "ok">("checking");
  const [authRequired, setAuthRequired] = useState(false);

  useEffect(() => {
    get<{ auth_required: boolean }>("/api/auth/status")
      .then((s) => {
        setAuthRequired(s.auth_required);
        setState(s.auth_required && !getToken() ? "login" : "ok");
      })
      .catch(() => setState("ok")); // API down: the dashboard shows its own errors
    const onAuth = () => setState("login");
    window.addEventListener(AUTH_EVENT, onAuth);
    return () => window.removeEventListener(AUTH_EVENT, onAuth);
  }, []);

  if (state === "checking") return <div className="min-h-screen flex items-center justify-center text-sm text-zinc-500">Loading…</div>;
  if (state === "login") return <Login onDone={() => setState("ok")} />;
  return <Dashboard authRequired={authRequired} />;
}

function Dashboard({ authRequired }: { authRequired: boolean }) {
  const [universe, setUniverse] = useState("NIFTY100");
  const [sector, setSector] = useState("");
  const [topN, setTopN] = useState(12);
  const [sectors, setSectors] = useState<string[]>([]);
  const [market, setMarket] = useState<Market | null>(null);
  const [scan, setScan] = useState<Scan | null>(null);
  const [scanId, setScanId] = useState<number | null>(null);
  const [open, setOpen] = useState<Pick | null>(null);
  const [tierFilter, setTierFilter] = useState<Tier | "all">("all");
  const [tierStats, setTierStats] = useState<TierStats>({});
  const [chatOpen, setChatOpen] = useState(false);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [tab, setTab] = useState<"picks" | "sectors" | "backtest" | "paper">("picks");
  const [paperKey, setPaperKey] = useState(0);
  const [err, setErr] = useState("");

  useEffect(() => {
    get<{ sectors: string[] }>("/api/universe").then((d) => setSectors(d.sectors)).catch((e) => setErr(String(e)));
    get<Market>("/api/market").then(setMarket).catch(() => {});
    get<{ variants: [] }>("/api/research").then((r) => setTierStats(tierStatsFrom(r))).catch(() => {});
    get<Settings>("/api/settings").then(setSettings).catch(() => {});
  }, []);

  // Polls a running scan; network blips are retried with backoff instead of leaving the button stuck.
  const poll = useCallback(async (id: number, failures = 0) => {
    try {
      const s = await get<Scan>(`/api/scans/${id}`);
      setScan(s);
      if (s.status === "running") setTimeout(() => poll(id), 2500);
      else setScanId(null);
    } catch (e) {
      if (failures >= 8) {
        setErr(`Lost track of scan #${id}: ${e instanceof Error ? e.message : e}`);
        setScanId(null);
      } else setTimeout(() => poll(id, failures + 1), Math.min(30000, 2500 * 2 ** failures));
    }
  }, []);

  useEffect(() => {
    // pick up a scan that is still running (e.g. the scheduled daily scan) so the page shows its progress
    get<Scan>("/api/scans/latest?include_running=true")
      .then((s) => {
        if (!s?.id) return;
        if (s.status === "running") {
          setScanId(s.id);
          poll(s.id);
        } else if (s.status === "done") setScan(s);
        else get<Scan>("/api/scans/latest").then((d) => d?.id && setScan(d)).catch(() => {});
      })
      .catch(() => {});
  }, [poll]);

  async function runScan() {
    setErr("");
    try {
      const { scan_id } = await post<{ scan_id: number }>("/api/scan", { universe, sector: sector || null, top_n: topN });
      setScanId(scan_id);
      poll(scan_id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  }

  const picks = useMemo(
    () => (scan?.picks || []).filter((p) => p.side !== "short" && (tierFilter === "all" || p.tier === tierFilter)),
    [scan, tierFilter],
  );
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    (scan?.picks || []).filter((p) => p.side !== "short").forEach((p) => (c[p.tier] = (c[p.tier] || 0) + 1));
    return c;
  }, [scan]);
  const running = scan?.status === "running" || scanId !== null;

  return (
    <div className="flex h-screen overflow-hidden">
      <main className="flex-1 min-w-0 overflow-y-auto">
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
              {authRequired && (
                <button
                  onClick={() => {
                    setToken("");
                    window.dispatchEvent(new Event(AUTH_EVENT));
                  }}
                  className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-900"
                >
                  Sign out
                </button>
              )}
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
              ⚠ Nifty is below its 200-day average (bearish regime). Be extra careful with position sizes.
            </div>
          )}

          {scan && (
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="text-zinc-400">
                Scan #{scan.id} · {scan.universe}{scan.sector ? ` · ${scan.sector}` : ""} · {scan.created_at.replace("T", " ")}
                {scan.market?.breadth200 != null && (
                  <> · Nifty {scan.market.above_ema200 ? "above" : "below"} 200-day avg · breadth {Math.round(scan.market.breadth200 * 100)}%{" "}
                    {scan.market.above_ema200 && scan.market.breadth200 > 0.5 ? "✓ market gate open" : "(market gate closed: no Buy calls)"}</>
                )}
              </span>
              <span className="flex-1" />
              {(["all", "strong", "watch", "avoid"] as const).map((t) => (
                <button key={t} onClick={() => setTierFilter(t)} className={`rounded-full px-3 py-1 text-xs border ${tierFilter === t ? "border-emerald-500 text-emerald-300" : "border-zinc-800 text-zinc-400"}`}>
                  {t === "all" ? `All (${Object.values(counts).reduce((a, b) => a + b, 0)})` : `${t === "strong" ? "buy" : t} (${counts[t] || 0})`}
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
            <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
              {picks.map((p) => (
                <PickCard key={p.symbol} p={p} settings={settings} stat={tierStats[p.tier]} onOpen={() => setOpen(p)} />
              ))}
            </div>
          ) : (
            !running && (
              <div className="rounded-xl border border-dashed border-zinc-800 p-10 text-center text-zinc-500">
                Pick a universe and sector, then press <b className="text-zinc-300">Run AI Scan</b>. The research agent checks news, filings and result dates for every shortlisted stock.
              </div>
            )
          )}

          <div className="rounded-xl border border-zinc-800 p-4 text-xs text-zinc-400">
            <div className="flex flex-wrap gap-3">
              <Legend cls="bg-emerald-500" t="Buy: top 2% ML rank, Nifty and > 50% of stocks above their 200-day averages, agent clear" />
              <Legend cls="bg-zinc-700" t="Watch" />
              <Legend cls="bg-red-800" t="Avoid: confirmed red flag" />
            </div>
            {tierStats.strong && (
              <div className="mt-2">
                In a 22-year test that includes delisted stocks, Buy-rule trades closed in profit {(tierStats.strong.win_rate * 100).toFixed(0)}% of the
                time with a profit factor of {tierStats.strong.profit_factor}: a modest edge. Expect many losing trades; size every position with
                the stop-loss in mind.
              </div>
            )}
            <div className="mt-2 text-zinc-500">Educational tool, not financial advice. Past results do not guarantee future returns.</div>
          </div>
        </div>
        )}
      </main>

      <aside className={`${chatOpen ? "flex fixed inset-0 z-30" : "hidden"} lg:static lg:flex w-full lg:w-[400px] lg:shrink-0 min-w-0 overflow-hidden border-l border-zinc-800 bg-zinc-950`}>
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
  const n = Math.round(v) || 0; // avoid "-0" for tiny outflows
  return <b className={n >= 0 ? "text-emerald-400" : "text-red-400"}>{n > 0 ? "+" : ""}{n.toLocaleString("en-IN")}</b>;
}
