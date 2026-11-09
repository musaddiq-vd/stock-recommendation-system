export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Tier = "strong" | "good" | "watch" | "avoid";

/** What the research agent found. It can only hold a pick back; it does not score the chart. */
export interface Research {
  verdict: string; // buy | watch | avoid
  summary: string;
  bull_case: string[];
  bear_case: string[];
  news_sentiment: string;
  event_risk: string;
  red_flags?: string[];
  sources?: string[];
  steps?: { tool: string; input: Record<string, unknown> }[];
}

export interface Pick {
  symbol: string;
  name: string;
  sector: string;
  indices: string[];
  side?: string; // older scans may contain short picks; the dashboard hides them
  official_event?: { purpose: string; date: string; desc: string } | null;
  rubric_score?: number;
  rubric_parts?: Record<string, number>;
  gates?: { regime: boolean; rubric: boolean; agent?: boolean };
  as_of: string;
  ml_prob: number;
  ml_percentile: number;
  hist_win_rate: number;
  hist_avg_ret: number;
  score: number;
  tier: Tier;
  hold_days: string;
  levels: { entry: number; target: number; stop_loss: number; target_pct: number; stop_pct: number };
  technicals: Record<string, number | string>;
  research: Research;
  outcome?: string | null;
  outcome_ret?: number | null;
}

export interface Market {
  nifty: number;
  ret_20d_pct: number;
  above_ema200: boolean;
  regime: "bullish" | "neutral" | "bearish";
  date: string;
  flows?: { date?: string; fii?: number; dii?: number; fii_5d?: number; dii_5d?: number; days_tracked?: number };
}

export interface Scan {
  id: number;
  created_at: string;
  universe: string;
  sector: string | null;
  mode: string;
  status: "running" | "done" | "error";
  progress: { t: string; msg: string }[];
  market: Market | null;
  error: string | null;
  picks: Pick[];
}

// ---------- session ----------
const TOKEN_KEY = "swing.token";
export const AUTH_EVENT = "swing:auth-required";

export function getToken(): string {
  try {
    return localStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function setToken(token: string) {
  try {
    if (token) localStorage.setItem(TOKEN_KEY, token);
    else localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* storage unavailable (private mode): the session lasts until reload */
  }
}

/** fetch() against the API with the session token; a 401 clears it and asks the page to show the login. */
export async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const r = await fetch(`${API}${path}`, { cache: "no-store", ...init, headers });
  if (r.status === 401 && !path.startsWith("/api/auth/")) {
    setToken("");
    window.dispatchEvent(new Event(AUTH_EVENT));
  }
  return r;
}

/** Readable error text from a FastAPI error body ({"detail": "..."} or validation errors). */
async function errorText(r: Response): Promise<string> {
  try {
    const j = await r.json();
    if (typeof j.detail === "string") return j.detail;
    if (Array.isArray(j.detail)) return j.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `Request failed (${r.status})`;
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const r = await apiFetch(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) throw new Error(await errorText(r));
  return r.json();
}

export const get = <T,>(path: string) => send<T>(path, "GET");
export const post = <T,>(path: string, body: unknown) => send<T>(path, "POST", body);
export const put = <T,>(path: string, body: unknown) => send<T>(path, "PUT", body);
export const del = <T,>(path: string) => send<T>(path, "DELETE");

export const TIER_STYLE: Record<Tier, { card: string; badge: string; label: string }> = {
  strong: {
    card: "border-emerald-500 bg-emerald-950/60 shadow-[0_0_24px_-6px_rgba(16,185,129,0.6)]",
    badge: "bg-emerald-500 text-black",
    label: "Top tier",
  },
  good: { card: "border-emerald-700/70 bg-emerald-950/25", badge: "bg-emerald-800 text-emerald-100", label: "Good" },
  watch: { card: "border-zinc-700 bg-zinc-900/60", badge: "bg-zinc-700 text-zinc-200", label: "Watch" },
  avoid: { card: "border-red-900/70 bg-red-950/20", badge: "bg-red-800 text-red-100", label: "Avoid" },
};

/** 22-year walk-forward result that backs each tier (from /api/research). */
export interface TierStat { win_rate: number; profit_factor: number; max_dd: number; avg_trade: number; trades: number }
export type TierStats = Partial<Record<Tier, TierStat>>;

export function tierStatsFrom(research: { variants?: (TierStat & { name: string })[] } | null): TierStats {
  const v = research?.variants || [];
  const live = v.find((x) => x.name.includes("live setting"));
  const good = v.find((x) => x.name.includes("Good tier"));
  return { ...(live ? { strong: live } : {}), ...(good ? { good } : {}) };
}
