"use client";

export const TABS = ["picks", "sectors", "backtest", "paper"] as const;
export type Tab = (typeof TABS)[number];

export const TAB_META: Record<Tab, { label: string; short: string; title: string; hint: string }> = {
  picks: { label: "AI Picks", short: "Picks", title: "AI Picks", hint: "Candidates from the latest scan" },
  sectors: { label: "Sectors", short: "Sectors", title: "Sector Heatmap", hint: "Where money is flowing, sector by sector" },
  backtest: { label: "Backtest", short: "Backtest", title: "Backtest", hint: "Live track record and the 22-year walk-forward" },
  paper: { label: "Portfolio", short: "Portfolio", title: "Paper Portfolio", hint: "Virtual trades opened from picks" },
};

const PATHS: Record<Tab | "settings" | "signout", string[]> = {
  picks: ["M12 3l2.4 5.6L20 9.5l-4.3 3.9L17 19l-5-2.9L7 19l1.3-5.6L4 9.5l5.6-.9L12 3z"],
  sectors: ["M4 4h7v7H4z", "M13 4h7v4h-7z", "M13 10h7v10h-7z", "M4 13h7v7H4z"],
  backtest: ["M4 19V5", "M4 19h16", "M7 15l4-5 3 3 5-7"],
  paper: ["M4 8h16v11H4z", "M9 8V5h6v3", "M4 13h16"],
  settings: ["M12 9a3 3 0 100 6 3 3 0 000-6z",
    "M19 12a7 7 0 00-.1-1.3l2-1.5-2-3.4-2.3.9a7 7 0 00-2.2-1.3L14 3h-4l-.4 2.4a7 7 0 00-2.2 1.3l-2.3-.9-2 3.4 2 1.5A7 7 0 005 12c0 .4 0 .9.1 1.3l-2 1.5 2 3.4 2.3-.9a7 7 0 002.2 1.3L10 21h4l.4-2.4a7 7 0 002.2-1.3l2.3.9 2-3.4-2-1.5c.1-.4.1-.9.1-1.3z"],
  signout: ["M10 4H5v16h5", "M15 8l4 4-4 4", "M19 12H9"],
};

export function Icon({ name, className = "h-5 w-5" }: { name: keyof typeof PATHS; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.7} strokeLinecap="round" strokeLinejoin="round"
      className={className} aria-hidden="true">
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  );
}

const FOCUS = "outline-none focus-visible:ring-2 focus-visible:ring-emerald-500";

/** Desktop: a slim rail on the left with the four pages on top and settings at the bottom. */
export function NavRail({ tab, onTab, onSettings, onSignOut, buyCount, alertsOn }: {
  tab: Tab; onTab: (t: Tab) => void; onSettings: () => void; onSignOut?: () => void; buyCount: number; alertsOn: boolean;
}) {
  return (
    <nav aria-label="Main" className="hidden lg:flex w-[76px] shrink-0 flex-col items-center border-r border-zinc-800 bg-zinc-950 py-3">
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-600 text-sm font-bold text-white" title="AI Swing Trader · NSE">
        AI
      </div>
      <div className="flex flex-1 flex-col gap-1">
        {TABS.map((t) => (
          <button key={t} onClick={() => onTab(t)} aria-current={tab === t ? "page" : undefined} title={TAB_META[t].title}
            className={`relative flex w-16 flex-col items-center gap-1 rounded-lg px-1 py-2 text-[10px] font-medium transition-colors ${FOCUS} ${
              tab === t ? "bg-emerald-500/10 text-emerald-300" : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200"}`}>
            {tab === t && <span className="absolute -left-1.5 top-2 bottom-2 w-0.5 rounded bg-emerald-400" />}
            <Icon name={t} />
            {TAB_META[t].label}
            {t === "picks" && buyCount > 0 && (
              <span className="absolute right-1.5 top-1 min-w-4 rounded-full bg-emerald-500 px-1 text-center text-[9px] font-bold leading-4 text-zinc-950"
                title={`${buyCount} Buy calls in the latest scan`}>
                {buyCount}
              </span>
            )}
          </button>
        ))}
      </div>
      <button onClick={onSettings} title="Capital, risk and Telegram alerts"
        className={`relative flex w-16 flex-col items-center gap-1 rounded-lg px-1 py-2 text-[10px] font-medium text-zinc-400 hover:bg-zinc-900 hover:text-zinc-200 ${FOCUS}`}>
        <Icon name="settings" />
        Settings
        {alertsOn && <span className="absolute right-3 top-1.5 h-2 w-2 rounded-full bg-emerald-400" title="Telegram alerts are on" />}
      </button>
      {onSignOut && (
        <button onClick={onSignOut} title="Sign out"
          className={`flex w-16 flex-col items-center gap-1 rounded-lg px-1 py-2 text-[10px] font-medium text-zinc-500 hover:bg-zinc-900 hover:text-zinc-200 ${FOCUS}`}>
          <Icon name="signout" />
          Sign out
        </button>
      )}
    </nav>
  );
}

/** Phone and tablet: the same pages as a bar at the bottom of the screen, within thumb reach. */
export function MobileNav({ tab, onTab, buyCount }: { tab: Tab; onTab: (t: Tab) => void; buyCount: number }) {
  return (
    <nav aria-label="Main" className="lg:hidden fixed inset-x-0 bottom-0 z-20 grid grid-cols-4 border-t border-zinc-800 bg-zinc-950/95 backdrop-blur pb-[env(safe-area-inset-bottom)]">
      {TABS.map((t) => (
        <button key={t} onClick={() => onTab(t)} aria-current={tab === t ? "page" : undefined}
          className={`relative flex flex-col items-center gap-0.5 py-2 text-[11px] font-medium ${FOCUS} ${tab === t ? "text-emerald-300" : "text-zinc-400"}`}>
          {tab === t && <span className="absolute top-0 h-0.5 w-10 rounded-b bg-emerald-400" />}
          <span className="relative">
            <Icon name={t} />
            {t === "picks" && buyCount > 0 && (
              <span className="absolute -right-3 -top-1 min-w-4 rounded-full bg-emerald-500 px-1 text-center text-[9px] font-bold leading-4 text-zinc-950">{buyCount}</span>
            )}
          </span>
          {TAB_META[t].short}
        </button>
      ))}
    </nav>
  );
}
