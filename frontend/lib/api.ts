export const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type Tier = "strong" | "good" | "watch" | "avoid";

export interface Research {
  verdict: "buy" | "short" | "watch" | "avoid";
  conviction: number;
  scores?: Record<string, number>;
  score_notes?: Record<string, string>;
  red_flags?: string[];
  setup?: string;
  summary: string;
  bull_case: string[];
  bear_case: string[];
  news_sentiment: string;
  event_risk: string;
  key_levels?: string;
  sources?: string[];
  steps?: { tool: string; input: Record<string, unknown> }[];
}

export interface Pick {
  symbol: string;
  name: string;
  sector: string;
  indices: string[];
  side?: "long" | "short";
  lot_size?: number | null;
  official_event?: { purpose: string; date: string; desc: string } | null;
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

export async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${API}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json();
}

export const TIER_STYLE: Record<Tier, { card: string; badge: string; label: string }> = {
  strong: {
    card: "border-emerald-500 bg-emerald-950/60 shadow-[0_0_24px_-6px_rgba(16,185,129,0.6)]",
    badge: "bg-emerald-500 text-black",
    label: "High probability",
  },
  good: { card: "border-emerald-700/70 bg-emerald-950/25", badge: "bg-emerald-800 text-emerald-100", label: "Good" },
  watch: { card: "border-zinc-700 bg-zinc-900/60", badge: "bg-zinc-700 text-zinc-200", label: "Watch" },
  avoid: { card: "border-red-900/70 bg-red-950/20", badge: "bg-red-800 text-red-100", label: "Avoid" },
};
