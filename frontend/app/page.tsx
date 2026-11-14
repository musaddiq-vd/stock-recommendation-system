"use client";
import { useCallback, useEffect, useMemo, useState } from "react";
import ChatDock from "@/components/ChatDock";
import PaperPanel from "@/components/PaperPanel";
import BacktestPanel from "@/components/BacktestPanel";
import SectorHeatmap from "@/components/SectorHeatmap";
import PickCard from "@/components/PickCard";
import SettingsPanel from "@/components/SettingsPanel";
import StockDrawer from "@/components/StockDrawer";
import Login from "@/components/Login";
import { Icon, MobileNav, NavRail, Tab, TAB_META, TABS } from "@/components/Nav";
import { AUTH_EVENT, get, getToken, Market, Pick, post, Scan, setToken, Tier, TierStats, tierStatsFrom } from "@/lib/api";
import { Settings } from "@/lib/sizing";

const UNIVERSES = ["NIFTY50", "NIFTY100", "NIFTY500"];
const FILTERS: { key: Tier | "all"; label: string }[] = [
  { key: "all", label: "All" }, { key: "strong", label: "Buy" }, { key: "watch", label: "Watch" }, { key: "avoid", label: "Avoid" },
];

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

function tabFromHash(): Tab | null {
  const h = window.location.hash.slice(1);
  return (TABS as readonly string[]).includes(h) ? (h as Tab) : null;
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
  const [query, setQuery] = useState("");
  const [scanOpen, setScanOpen] = useState(false);
  const [heatIndex, setHeatIndex] = useState("NIFTY500"); // its own choice: browsing sectors must not change the next scan
  const [tierStats, setTierStats] = useState<TierStats>({});
  const [settings, setSettings] = useState<Settings | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [tab, setTab] = useState<Tab>("picks");
  const [paperKey, setPaperKey] = useState(0);
  const [err, setErr] = useState("");

  // The page lives in the URL (#picks, #sectors, ...), so refresh, back / forward and shared links keep it.
  useEffect(() => {
    const read = () => setTab(tabFromHash() ?? "picks");
    read();
    window.addEventListener("hashchange", read);
    return () => window.removeEventListener("hashchange", read);
  }, []);
  const goTab = useCallback((t: Tab) => {
    setTab(t);
    if (tabFromHash() !== t) window.history.pushState(null, "", `#${t}`);
  }, []);
  useEffect(() => {
    document.title = `${TAB_META[tab].title} · AI Swing Trader`;
  }, [tab]);

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

  const longs = useMemo(() => (scan?.picks || []).filter((p) => p.side !== "short"), [scan]);
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: longs.length };
    longs.forEach((p) => (c[p.tier] = (c[p.tier] || 0) + 1));
    return c;
  }, [longs]);
  const picks = useMemo(() => {
    const q = query.trim().toLowerCase();
    return longs.filter((p) => (tierFilter === "all" || p.tier === tierFilter)
      && (!q || `${p.symbol} ${p.name} ${p.sector}`.toLowerCase().includes(q)));
  }, [longs, tierFilter, query]);
  const running = scan?.status === "running" || scanId !== null;
  const topPick = scan?.status === "done" ? longs.find((p) => p.top_pick) : undefined;
  const sm = scan?.market;
  const gateOpen = sm?.breadth200 != null ? sm.above_ema200 && sm.breadth200 > 0.5 : null;
  const closeDrawer = useCallback(() => setOpen(null), []);
  const closeSettings = useCallback(() => setShowSettings(false), []);
  const signOut = authRequired
    ? () => {
        setToken("");
        window.dispatchEvent(new Event(AUTH_EVENT));
      }
    : undefined;

  return (
    <div className="flex h-dvh overflow-hidden">
      <NavRail tab={tab} onTab={goTab} onSettings={() => setShowSettings(true)} onSignOut={signOut}
        buyCount={counts.strong || 0} alertsOn={!!settings?.telegram_configured} />

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="z-10 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-zinc-800 bg-zinc-950 px-4 py-3 lg:px-6">
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-lg font-semibold text-white">
              <span className="lg:hidden">AI Swing Trader <span className="text-emerald-400">· NSE</span></span>
              <span className="hidden lg:inline">{TAB_META[tab].title}</span>
            </h1>
            <p className="hidden truncate text-xs text-zinc-500 lg:block">{TAB_META[tab].hint}</p>
          </div>
          <div className="flex items-center gap-2 lg:hidden">
            <button onClick={() => setShowSettings(true)} aria-label="Settings" className="rounded-lg border border-zinc-800 p-2 text-zinc-300 hover:bg-zinc-900">
              <Icon name="settings" className="h-4 w-4" />
            </button>
            {signOut && (
              <button onClick={signOut} aria-label="Sign out" className="rounded-lg border border-zinc-800 p-2 text-zinc-400 hover:bg-zinc-900">
                <Icon name="signout" className="h-4 w-4" />
              </button>
            )}
          </div>
          <div className="flex w-full items-center gap-2 overflow-x-auto whitespace-nowrap text-xs [scrollbar-width:none] lg:w-auto">
            {market && (
              <>
                <Chip>
                  <span className="text-zinc-500">NIFTY</span> <b className="text-zinc-100">{market.nifty.toLocaleString("en-IN")}</b>{" "}
                  <span className={market.ret_20d_pct >= 0 ? "text-emerald-400" : "text-red-400"}>
                    {market.ret_20d_pct > 0 ? "+" : ""}{market.ret_20d_pct}% <span className="text-zinc-500">20d</span>
                  </span>
                </Chip>
                <Chip tone={market.regime === "bullish" ? "good" : market.regime === "bearish" ? "bad" : undefined}>
                  <span className="capitalize">{market.regime}</span>
                </Chip>
              </>
            )}
            {gateOpen != null && (
              <Chip tone={gateOpen ? "good" : "warn"} title="Buy calls need Nifty and more than half of the stocks above their 200-day averages">
                <span className={`mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle ${gateOpen ? "bg-emerald-400" : "bg-amber-400"}`} />
                Market gate {gateOpen ? "open" : "closed"}
              </Chip>
            )}
            {market?.flows?.fii != null && (
              <Chip title={`Net institutional flows on ${market.flows.date}, in crore rupees`}>
                <span className="text-zinc-500">FII</span> <Flow v={market.flows.fii} /> <span className="text-zinc-500">· DII</span> <Flow v={market.flows.dii ?? 0} />
              </Chip>
            )}
            {market && <span className="text-zinc-600">as of {market.date}</span>}
          </div>
        </header>

        <main className="@container flex-1 overflow-y-auto pb-20 lg:pb-0">
          <div className="mx-auto w-full max-w-[1500px] space-y-5 px-4 py-5 lg:px-6">
            {err && <div className="rounded-lg border border-red-900 bg-red-950/30 px-4 py-2 text-sm text-red-200">{err}</div>}

            {tab === "paper" && <PaperPanel refreshKey={paperKey} />}
            {tab === "backtest" && <BacktestPanel />}
            {tab === "sectors" && (
              <>
                <div className="flex flex-wrap items-center gap-3">
                  <span className="text-xs uppercase tracking-wide text-zinc-500">Index</span>
                  <Segmented value={heatIndex} onChange={setHeatIndex} options={UNIVERSES.map((u) => ({ key: u, label: u.replace("NIFTY", "Nifty ") }))} />
                </div>
                <SectorHeatmap index={heatIndex} />
              </>
            )}

            {tab === "picks" && (
              <>
                {market?.regime === "bearish" && (
                  <div className="rounded-lg border border-red-900 bg-red-950/30 px-4 py-2 text-sm text-red-200">
                    Nifty is below its 200-day average (bearish regime). Be extra careful with position sizes.
                  </div>
                )}

                <section aria-label="New scan" className="rounded-xl border border-zinc-800 bg-zinc-900/40 p-4">
                  {/* phones: the controls fold away so the picks come first */}
                  <button type="button" onClick={() => setScanOpen((o) => !o)} aria-expanded={scanOpen}
                    className="flex w-full items-center justify-between text-sm font-medium text-zinc-200 lg:hidden">
                    <span>{running ? "Scanning…" : "New scan"}</span>
                    <span className="text-xs text-zinc-500">{universe.replace("NIFTY", "Nifty ")}{sector ? ` · ${sector}` : ""} · {scanOpen ? "hide" : "change"}</span>
                  </button>
                  <div className={`${scanOpen ? "mt-3 flex" : "hidden"} flex-wrap items-end gap-x-4 gap-y-3 lg:mt-0 lg:flex`}>
                    <Field label="Universe">
                      <Segmented value={universe} onChange={setUniverse} options={UNIVERSES.map((u) => ({ key: u, label: u.replace("NIFTY", "Nifty ") }))} />
                    </Field>
                    <Field label="Sector">
                      <select value={sector} onChange={(e) => setSector(e.target.value)}
                        className="h-9 rounded-lg border border-zinc-800 bg-zinc-950 px-2 text-sm text-zinc-200">
                        <option value="">All sectors</option>
                        {sectors.map((s) => <option key={s}>{s}</option>)}
                      </select>
                    </Field>
                    <Field label={`Stocks to research: ${topN}`}>
                      <input type="range" min={3} max={30} value={topN} onChange={(e) => setTopN(+e.target.value)} className="h-9 w-36 accent-emerald-500" />
                    </Field>
                    <button onClick={runScan} disabled={running}
                      className="h-9 rounded-lg bg-emerald-600 px-5 text-sm font-semibold text-white hover:bg-emerald-500 disabled:opacity-50">
                      {running ? "Scanning…" : "Run AI scan"}
                    </button>
                    <p className="basis-full text-xs text-zinc-500 @4xl:ml-auto @4xl:basis-auto @4xl:text-right">
                      A scan also runs by itself every trading day after the market closes.
                    </p>
                  </div>
                </section>

                {scan && scan.status !== "running" && (
                  <div className="grid grid-cols-2 gap-3 @3xl:grid-cols-4">
                    <Tile label="Market gate" tone={gateOpen ? "good" : gateOpen === false ? "warn" : undefined}
                      value={gateOpen == null ? "–" : gateOpen ? "Open" : "Closed"}
                      sub={sm?.breadth200 != null
                        ? `Nifty ${sm.above_ema200 ? "above" : "below"} 200-day avg · breadth ${Math.round(sm.breadth200 * 100)}%`
                        : "Not measured in this scan"} />
                    <Tile label="Buy calls" value={String(counts.strong || 0)} tone={counts.strong ? "good" : undefined}
                      sub={gateOpen === false ? "None while the gate is closed" : `of ${counts.all} stocks researched`} />
                    <Tile label="Buy rule, 22-year test"
                      value={tierStats.strong ? `${(tierStats.strong.win_rate * 100).toFixed(0)}% profitable` : "–"}
                      sub={tierStats.strong ? `profit factor ${tierStats.strong.profit_factor} · worst drawdown ${(tierStats.strong.max_dd * 100).toFixed(0)}%` : "Run the research report"} />
                    <Tile label="Latest scan" value={`#${scan.id} · ${scan.universe.replace("NIFTY", "Nifty ")}`}
                      sub={`${scan.created_at.replace("T", " ").slice(0, 16)}${scan.sector ? ` · ${scan.sector}` : ""}`} />
                  </div>
                )}

                {running && scan && (
                  <div className="rounded-xl border border-zinc-800 bg-zinc-900/50 p-4">
                    <div className="mb-2 flex items-center gap-2 text-sm font-medium text-white">
                      <span className="h-2 w-2 animate-ping rounded-full bg-emerald-400" /> Agents researching…
                    </div>
                    <div className="max-h-56 space-y-0.5 overflow-y-auto font-mono text-[11px] text-zinc-400">
                      {[...scan.progress].reverse().map((p, i) => <div key={i}><span className="text-zinc-600">{p.t}</span> {p.msg}</div>)}
                    </div>
                  </div>
                )}
                {scan?.status === "error" && (
                  <div className="rounded-lg border border-red-900 bg-red-950/30 px-4 py-2 text-sm text-red-200">Scan failed: {scan.error}</div>
                )}

                {topPick && <TopPick p={topPick} onOpen={() => setOpen(topPick)} />}

                {longs.length > 0 && (
                  <div className="flex flex-wrap items-center gap-3">
                    <Segmented value={tierFilter} onChange={setTierFilter}
                      options={FILTERS.map((f) => ({ key: f.key, label: `${f.label} ${counts[f.key] || 0}` }))} />
                    <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search symbol, name or sector" aria-label="Search picks"
                      className="h-9 w-full rounded-lg border border-zinc-800 bg-zinc-950 px-3 text-sm text-zinc-200 placeholder:text-zinc-600 @xl:ml-auto @xl:w-64" />
                  </div>
                )}

                {picks.length > 0 ? (
                  <div className="grid gap-4 @2xl:grid-cols-2 @6xl:grid-cols-3">
                    {picks.map((p) => (
                      <PickCard key={p.symbol} p={p} settings={settings} stat={tierStats[p.tier]} onOpen={() => setOpen(p)} />
                    ))}
                  </div>
                ) : (
                  !running && (
                    <div className="rounded-xl border border-dashed border-zinc-800 p-10 text-center text-sm text-zinc-500">
                      {longs.length > 0 ? (
                        <>
                          No picks match this filter.{" "}
                          <button className="text-emerald-400 hover:underline" onClick={() => { setTierFilter("all"); setQuery(""); }}>Show all</button>
                        </>
                      ) : (
                        <>
                          No scan yet. Choose a universe and press <b className="text-zinc-300">Run AI scan</b>: the research agent checks news, filings
                          and result dates for every shortlisted stock.
                        </>
                      )}
                    </div>
                  )
                )}

                <footer className="rounded-xl border border-zinc-800 p-4 text-xs text-zinc-400">
                  <div className="flex flex-wrap gap-x-4 gap-y-1.5">
                    <Legend cls="bg-emerald-500" t="Buy: top 2% ML rank, Nifty and > 50% of stocks above their 200-day averages, agent clear" />
                    <Legend cls="bg-zinc-600" t="Watch: one of the checks failed" />
                    <Legend cls="bg-red-700" t="Avoid: confirmed red flag" />
                  </div>
                  <p className="mt-2 text-zinc-500">
                    The edge is modest: expect about half of the trades to lose, and size every position with the stop-loss in mind. Educational tool,
                    not financial advice. Past results do not guarantee future returns.
                  </p>
                </footer>
              </>
            )}
          </div>
        </main>
      </div>

      <ChatDock />
      <MobileNav tab={tab} onTab={goTab} buyCount={counts.strong || 0} />

      {open && (
        <StockDrawer pick={open} onClose={closeDrawer} settings={settings} scanId={scan?.id}
          onPaper={() => setPaperKey((k) => k + 1)} />
      )}
      {showSettings && settings && <SettingsPanel s={settings} onSaved={setSettings} onClose={closeSettings} />}
    </div>
  );
}

/** The day's single best stock, with everything needed to judge it at a glance. */
function TopPick({ p, onOpen }: { p: Pick; onOpen: () => void }) {
  const buy = p.tier === "strong";
  const l = p.levels;
  return (
    <button onClick={onOpen}
      className={`group w-full rounded-xl border p-4 text-left transition-colors @3xl:p-5 ${buy
        ? "border-emerald-600/70 bg-gradient-to-br from-emerald-950/50 to-zinc-950 hover:border-emerald-500"
        : "border-zinc-700 bg-zinc-900/40 hover:border-zinc-500"}`}>
      <div className="grid gap-4 @5xl:grid-cols-[minmax(0,1fr)_minmax(0,1.25fr)] @5xl:items-center @5xl:gap-8">
        <div className="min-w-0">
          <div className="text-[11px] font-medium uppercase tracking-wider text-zinc-400">Top pick · {p.as_of}</div>
          <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1">
            <span className="text-2xl font-bold text-white">{p.symbol}</span>
            <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${buy ? "bg-emerald-500 text-zinc-950" : "bg-zinc-700 text-zinc-100"}`}>
              {buy ? "Buy" : "Watch"}
            </span>
            <span className="text-xs text-zinc-400">ML top {Math.max(1, 100 - p.ml_percentile)}% today</span>
          </div>
          <div className="mt-0.5 truncate text-sm text-zinc-400">{p.name} · {p.sector}</div>
          <p className="mt-2 text-xs text-zinc-500">
            {buy
              ? "Highest-ranked stock that passed all three checks."
              : `Highest-ranked stock the research agent cleared, but not a Buy: ${
                  p.gates && !p.gates.market ? "the market gate is closed" : "it is not in the top 2% of the ML ranking"
                }. Shown for research only.`}
            <span className="ml-1 text-emerald-400 group-hover:underline">Open details</span>
          </p>
        </div>
        <dl className="grid grid-cols-2 gap-2 text-sm @2xl:grid-cols-4">
          <Num k="Entry" v={`₹${l.entry}`} />
          <Num k="Target" v={`₹${l.target}`} sub={`+${l.target_pct}%`} cls="text-emerald-400" />
          <Num k="Stop" v={`₹${l.stop_loss}`} sub={`-${l.stop_pct}%`} cls="text-red-400" />
          <Num k="Chance of profit" v={p.odds ? `${(p.odds.win * 100).toFixed(0)}%` : "–"}
            sub={p.odds ? `~${Math.round(p.odds.avg_days)} days avg` : undefined} />
        </dl>
      </div>
    </button>
  );
}

function Num({ k, v, sub, cls = "text-white" }: { k: string; v: string; sub?: string; cls?: string }) {
  return (
    <div className="rounded-lg bg-black/30 px-3 py-2">
      <dt className="text-[10px] uppercase tracking-wide text-zinc-500">{k}</dt>
      <dd className={`text-base font-semibold ${cls}`}>{v}</dd>
      <dd className="h-4 text-[11px] text-zinc-500">{sub}</dd>
    </div>
  );
}

type Tone = "good" | "warn" | "bad";
const TONE: Record<Tone, string> = {
  good: "border-emerald-800/80 text-emerald-300", warn: "border-amber-800/80 text-amber-300", bad: "border-red-900 text-red-300",
};

function Chip({ children, tone, title }: { children: React.ReactNode; tone?: Tone; title?: string }) {
  return <span title={title} className={`rounded-full border px-2.5 py-1 ${tone ? TONE[tone] : "border-zinc-800 text-zinc-300"}`}>{children}</span>;
}

function Tile({ label, value, sub, tone }: { label: string; value: string; sub: string; tone?: Tone }) {
  return (
    <div className="rounded-xl border border-zinc-800 bg-zinc-900/40 px-4 py-3">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={`mt-0.5 text-lg font-semibold ${tone === "good" ? "text-emerald-400" : tone === "warn" ? "text-amber-400" : "text-white"}`}>{value}</div>
      <div className="mt-0.5 text-[11px] leading-snug text-zinc-500">{sub}</div>
    </div>
  );
}

function Segmented<T extends string>({ value, onChange, options }: {
  value: T; onChange: (v: T) => void; options: { key: T; label: string }[];
}) {
  return (
    <div className="inline-flex h-9 rounded-lg border border-zinc-800 bg-zinc-950 p-0.5">
      {options.map((o) => (
        <button key={o.key} type="button" onClick={() => onChange(o.key)} aria-pressed={value === o.key}
          className={`rounded-md px-3 text-sm transition-colors ${value === o.key ? "bg-emerald-600 font-medium text-white" : "text-zinc-400 hover:text-zinc-200"}`}>
          {o.label}
        </button>
      ))}
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
