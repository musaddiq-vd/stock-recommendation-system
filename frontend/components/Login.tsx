"use client";
import { useState } from "react";
import { post, setToken } from "@/lib/api";

export default function Login({ onDone }: { onDone: () => void }) {
  const [password, setPassword] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      const r = await post<{ token: string }>("/api/auth/login", { password });
      setToken(r.token);
      onDone();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <form onSubmit={submit} className="w-full max-w-sm rounded-xl border border-zinc-800 bg-zinc-950 p-6 space-y-4">
        <div>
          <h1 className="text-xl font-bold text-white">AI Swing Trader <span className="text-emerald-400">· NSE</span></h1>
          <p className="text-xs text-zinc-500 mt-1">Enter the dashboard password to continue.</p>
        </div>
        <input
          type="password"
          autoFocus
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="w-full rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-600"
        />
        {err && <div className="text-xs text-red-400">{err}</div>}
        <button disabled={busy || !password} className="w-full rounded-lg bg-emerald-600 py-2 text-sm font-semibold text-white disabled:opacity-50">
          {busy ? "Signing in…" : "Sign in"}
        </button>
      </form>
    </div>
  );
}
