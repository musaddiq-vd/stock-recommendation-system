"use client";
import { useState } from "react";
import { post, put } from "@/lib/api";
import { Settings } from "@/lib/sizing";

export default function SettingsPanel({ s, onSaved, onClose }: { s: Settings; onSaved: (s: Settings) => void; onClose: () => void }) {
  const [capital, setCapital] = useState(s.capital);
  const [risk, setRisk] = useState(s.risk_pct);
  const [margin, setMargin] = useState(s.margin_pct);
  const [alertsOn, setAlertsOn] = useState(s.alerts_enabled === "1");
  const [token, setToken] = useState("");
  const [chatId, setChatId] = useState(s.telegram_chat_id);
  const [msg, setMsg] = useState("");

  async function save() {
    setMsg("");
    const body: Record<string, unknown> = { capital: +capital, risk_pct: +risk, margin_pct: +margin, alerts_enabled: alertsOn };
    if (token.trim()) body.telegram_bot_token = token.trim();
    if (chatId.trim()) body.telegram_chat_id = chatId.trim();
    let ns: Settings;
    try {
      ns = await put<Settings>("/api/settings", body);
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
      return;
    }
    setChatId(ns.telegram_chat_id);
    setToken("");
    onSaved(ns);
    setMsg("Saved ✓");
  }

  async function test() {
    setMsg("Sending…");
    try {
      await post("/api/alerts/test", {});
      setMsg("Test message sent ✓ Check Telegram.");
    } catch (e) {
      setMsg(String(e).replace(/^Error: \d+ /, ""));
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-lg rounded-xl border border-zinc-800 bg-zinc-950 p-5 max-h-[90vh] overflow-y-auto" onClick={(e) => e.stopPropagation()}>
        <div className="flex justify-between items-center">
          <h2 className="text-lg font-semibold text-white">Settings</h2>
          <button onClick={onClose} className="text-zinc-400 text-2xl leading-none">×</button>
        </div>

        <h3 className="mt-4 text-sm font-semibold text-zinc-200">Position sizing</h3>
        <div className="grid grid-cols-3 gap-3 mt-2">
          <Num label="Capital (₹)" v={capital} set={setCapital} />
          <Num label="Risk per trade %" v={risk} set={setRisk} step="0.25" />
          <Num label="F&O margin %" v={margin} set={setMargin} />
        </div>
        <p className="text-[11px] text-zinc-500 mt-1">
          Risk % = the most you lose (as % of capital) if the stop-loss hits. 1% is common. F&O margin is approximate, confirm with your broker.
        </p>

        <h3 className="mt-5 text-sm font-semibold text-zinc-200">Telegram alerts</h3>
        <ol className="text-xs text-zinc-400 list-decimal ml-4 mt-1 space-y-0.5">
          <li>Open <b>@BotFather</b> in Telegram → <code>/newbot</code> → copy the bot token.</li>
          <li>Open your new bot and send <b>/start</b>.</li>
          <li>Paste the token here and Save (the chat ID is detected automatically), then press Send test alert.</li>
        </ol>
        <div className="grid gap-2 mt-2">
          <input value={token} onChange={(e) => setToken(e.target.value)} placeholder={s.telegram_token_hint ? `Saved token ${s.telegram_token_hint} (paste a new one to replace)` : "Bot token"}
            className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm" />
          <input value={chatId} onChange={(e) => setChatId(e.target.value)} placeholder="Chat ID (auto)"
            className="rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm" />
          <label className="flex items-center gap-2 text-sm text-zinc-300">
            <input type="checkbox" checked={alertsOn} onChange={(e) => setAlertsOn(e.target.checked)} className="accent-emerald-500" />
            Alerts on: naye green picks, target/stop hit, paper trade exits
          </label>
        </div>
        <p className="text-[11px] text-zinc-500 mt-1">WhatsApp needs a paid Business API, so alerts use Telegram (free).</p>

        <div className="flex items-center gap-2 mt-5">
          <button onClick={save} className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white">Save</button>
          <button onClick={test} className="rounded-lg border border-zinc-700 px-4 py-2 text-sm text-zinc-200">Send test alert</button>
          <span className="text-xs text-zinc-400">{msg}</span>
        </div>
      </div>
    </div>
  );
}

function Num({ label, v, set, step = "1" }: { label: string; v: string; set: (x: string) => void; step?: string }) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] uppercase text-zinc-500">{label}</span>
      <input type="number" step={step} value={v} onChange={(e) => set(e.target.value)}
        className="rounded-lg bg-zinc-900 border border-zinc-800 px-2 py-1.5 text-sm text-white" />
    </label>
  );
}
