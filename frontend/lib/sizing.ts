import { Pick } from "./api";

export interface Settings {
  capital: string;
  risk_pct: string;
  alerts_enabled: string;
  telegram_chat_id: string;
  telegram_configured: boolean;
  telegram_token_hint: string;
}

export interface Size {
  qty: number; // shares
  invest: number; // capital used
  maxLoss: number; // loss if the stop hits
  profit: number; // gain if the target hits
  riskAmt: number; // capital x risk %
  note?: string;
}

/** Risk-based sizing: lose at most risk% of capital if the stop hits, never invest more than the capital. */
export function positionSize(p: Pick, s: Settings | null): Size | null {
  if (!s) return null;
  const capital = +s.capital;
  const riskAmt = (capital * +s.risk_pct) / 100;
  const { entry, target, stop_loss } = p.levels;
  const perShareRisk = entry - stop_loss;
  const perShareGain = target - entry;
  if (!capital || perShareRisk <= 0 || entry <= 0) return null;
  const qty = Math.min(Math.floor(riskAmt / perShareRisk), Math.floor(capital / entry));
  if (qty < 1) return { qty: 0, invest: 0, maxLoss: 0, profit: 0, riskAmt, note: "Capital is too small for even 1 share." };
  return {
    qty, invest: Math.round(qty * entry), maxLoss: Math.round(qty * perShareRisk),
    profit: Math.round(qty * perShareGain), riskAmt,
  };
}

export const inr = (n: number) => `₹${Math.round(n).toLocaleString("en-IN")}`;
