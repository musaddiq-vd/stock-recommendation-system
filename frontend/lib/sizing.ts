import { Pick } from "./api";

export interface Settings {
  capital: string;
  risk_pct: string;
  margin_pct: string;
  alerts_enabled: string;
  telegram_chat_id: string;
  telegram_configured: boolean;
  telegram_token_hint: string;
}

export interface Size {
  qty: number; // shares
  lots: number | null; // futures lots (shorts)
  invest: number; // capital used (cash for longs, approx margin for shorts)
  exposure: number;
  maxLoss: number;
  profit: number;
  riskAmt: number;
  note?: string;
}

/** Risk-based sizing: lose at most risk% of capital if the stop hits. */
export function positionSize(p: Pick, s: Settings | null): Size | null {
  if (!s) return null;
  const capital = +s.capital, riskAmt = (capital * +s.risk_pct) / 100;
  const { entry, target, stop_loss } = p.levels;
  const perShareRisk = Math.abs(entry - stop_loss);
  const perShareGain = Math.abs(target - entry);
  if (!capital || !perShareRisk) return null;

  if (p.side === "short") {
    const lot = p.lot_size || 1;
    const marginPerLot = (lot * entry * +s.margin_pct) / 100;
    let lots = Math.floor(riskAmt / (lot * perShareRisk));
    lots = Math.min(lots, Math.floor(capital / marginPerLot));
    if (lots < 1) {
      const need = Math.ceil(((lot * perShareRisk) / capital) * 1000) / 10;
      return { qty: 0, lots: 0, invest: 0, exposure: 0, maxLoss: 0, profit: 0, riskAmt,
        note: `Risk for 1 lot is ₹${Math.round(lot * perShareRisk).toLocaleString("en-IN")} (${need}% of capital). Increase risk % or capital.` };
    }
    const qty = lots * lot;
    return { qty, lots, invest: Math.round(lots * marginPerLot), exposure: Math.round(qty * entry),
      maxLoss: Math.round(qty * perShareRisk), profit: Math.round(qty * perShareGain), riskAmt };
  }

  const qty = Math.min(Math.floor(riskAmt / perShareRisk), Math.floor(capital / entry));
  if (qty < 1) return { qty: 0, lots: null, invest: 0, exposure: 0, maxLoss: 0, profit: 0, riskAmt, note: "Capital is too small for even 1 share." };
  return { qty, lots: null, invest: Math.round(qty * entry), exposure: Math.round(qty * entry),
    maxLoss: Math.round(qty * perShareRisk), profit: Math.round(qty * perShareGain), riskAmt };
}

export const inr = (n: number) => `₹${Math.round(n).toLocaleString("en-IN")}`;
