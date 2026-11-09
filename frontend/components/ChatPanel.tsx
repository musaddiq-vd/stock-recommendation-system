"use client";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiFetch } from "@/lib/api";

// Compact dark styling for agent answers: tables scroll sideways instead of overflowing the panel
const MD: Components = {
  table: ({ children }) => (
    <div className="my-2 overflow-x-auto rounded-lg border border-zinc-700">
      <table className="w-full text-xs">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-zinc-800 text-zinc-300">{children}</thead>,
  th: ({ children }) => <th className="px-2 py-1.5 text-left font-semibold whitespace-nowrap">{children}</th>,
  td: ({ children }) => <td className="px-2 py-1.5 border-t border-zinc-800 whitespace-nowrap tabular-nums">{children}</td>,
  h1: ({ children }) => <h3 className="mt-2 mb-1 text-sm font-semibold text-white">{children}</h3>,
  h2: ({ children }) => <h3 className="mt-2 mb-1 text-sm font-semibold text-white">{children}</h3>,
  h3: ({ children }) => <h4 className="mt-2 mb-1 text-sm font-semibold text-white">{children}</h4>,
  p: ({ children }) => <p className="my-1.5 leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="my-1.5 ml-4 list-disc space-y-1">{children}</ul>,
  ol: ({ children }) => <ol className="my-1.5 ml-4 list-decimal space-y-1">{children}</ol>,
  strong: ({ children }) => <strong className="font-semibold text-white">{children}</strong>,
  hr: () => <hr className="my-2 border-zinc-700" />,
  a: ({ children, href }) => (
    <a href={href} target="_blank" rel="noopener noreferrer" className="text-sky-400 hover:underline">{children}</a>
  ),
  blockquote: ({ children }) => <blockquote className="my-1.5 border-l-2 border-amber-600 pl-2 text-zinc-400">{children}</blockquote>,
};

type Msg = { role: "user" | "assistant"; content: string; tools?: string[]; error?: boolean };

const SUGGESTIONS = [
  "Any Buy calls today? Why or why not?",
  "I have ₹1,00,000. How should I size today's picks?",
  "Any good swing trade in the banking sector?",
  "What is the 15-20 day outlook for RELIANCE?",
];
const STORE = "chat-history-v1";
const KEEP = 40; // messages remembered in this browser
const SEND = 20; // messages sent to the server (it accepts up to 60 of 8,000 characters)
const MAX_INPUT = 2000;

// Friendly names for the agent's tool calls
const TOOL_LABEL: Record<string, string> = {
  get_latest_picks: "Read the latest scan", get_technicals: "Technicals", get_fundamentals: "Fundamentals",
  get_news: "News", web_search: "Web search", get_corporate_events: "NSE event calendar",
};
function toolLabel(name: string, input: Record<string, unknown>) {
  const arg = Object.entries(input || {})
    .map(([k, v]) => (k === "capital" ? `₹${Number(v).toLocaleString("en-IN")}` : String(v)))
    .filter(Boolean).join(", ");
  return `${TOOL_LABEL[name] ?? name}${arg ? ` · ${arg}` : ""}`;
}

function loadHistory(): Msg[] {
  try {
    const m = JSON.parse(localStorage.getItem(STORE) || "[]");
    // drop an answer that was still streaming when the page closed
    return Array.isArray(m) ? m.filter((x: Msg) => x && x.content) : [];
  } catch {
    return [];
  }
}

export default function ChatPanel({ onHide }: { onHide?: () => void }) {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);
  const stick = useRef(true); // follow new text only while the user is at the bottom

  useEffect(() => setMsgs(loadHistory()), []);
  useEffect(() => {
    if (busy) return; // save finished conversations only
    try {
      localStorage.setItem(STORE, JSON.stringify(msgs.slice(-KEEP)));
    } catch {
      /* storage blocked: history just is not remembered */
    }
  }, [msgs, busy]);
  useEffect(() => {
    const el = scroller.current;
    if (el && stick.current) el.scrollTop = el.scrollHeight;
  }, [msgs]);
  useEffect(() => {
    const el = box.current; // grow the input with its text, up to ~6 lines
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 140)}px`;
  }, [input]);

  function onScroll() {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
  }

  function patchLast(f: (m: Msg) => Msg) {
    setMsgs((m) => (m.length ? [...m.slice(0, -1), f({ ...m[m.length - 1] })] : m));
  }

  async function send(text: string, base: Msg[] = msgs) {
    const q = text.trim().slice(0, MAX_INPUT);
    if (!q || busy) return;
    const history: Msg[] = [...base, { role: "user", content: q }];
    setMsgs([...history, { role: "assistant", content: "", tools: [] }]);
    setInput("");
    setBusy(true);
    stick.current = true;
    const ctl = new AbortController();
    abort.current = ctl;
    try {
      const payload = history.filter((m) => !m.error && m.content).slice(-SEND)
        .map(({ role, content }) => ({ role, content: content.slice(0, 8000) }));
      const r = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: payload }),
        signal: ctl.signal,
      });
      if (!r.ok || !r.body) throw new Error(r.status === 401 ? "Please sign in again." : `Chat failed (${r.status}).`);
      const reader = r.body.getReader();
      const dec = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += dec.decode(value, { stream: true });
        const parts = buf.split("\n\n");
        buf = parts.pop() || "";
        for (const p of parts) {
          const data = p.replace(/^data: /, "").trim();
          if (!data || data === "[DONE]") continue;
          let ev: { type: string; name?: string; input?: Record<string, unknown>; text?: string };
          try {
            ev = JSON.parse(data);
          } catch {
            continue; // skip a malformed chunk instead of losing the whole answer
          }
          if (ev.type === "tool") patchLast((m) => ({ ...m, tools: [...(m.tools || []), toolLabel(ev.name || "", ev.input || {})] }));
          else if (ev.type === "error") patchLast((m) => ({ ...m, content: ev.text || "Something went wrong.", error: true }));
          else patchLast((m) => ({ ...m, content: ev.text || "" }));
        }
      }
      patchLast((m) => (m.content ? m : { ...m, content: "No answer came back. Try again.", error: true }));
    } catch (e) {
      const stopped = e instanceof DOMException && e.name === "AbortError";
      patchLast((m) => ({ ...m, content: stopped ? "Stopped." : `${e instanceof Error ? e.message : e}`, error: !stopped }));
    } finally {
      abort.current = null;
      setBusy(false);
    }
  }

  function retry() {
    const lastUser = [...msgs].reverse().findIndex((m) => m.role === "user");
    if (lastUser < 0) return;
    const i = msgs.length - 1 - lastUser;
    send(msgs[i].content, msgs.slice(0, i));
  }

  async function copy(i: number) {
    try {
      await navigator.clipboard.writeText(msgs[i].content);
      setCopied(i);
      setTimeout(() => setCopied(null), 1500);
    } catch {
      /* clipboard blocked */
    }
  }

  const last = msgs.length - 1;
  return (
    <div className="flex flex-col h-full w-full min-w-0">
      <div className="px-4 py-3 border-b border-zinc-800 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="font-semibold text-white">Research Chat</div>
          <div className="text-xs text-zinc-500 truncate">Live data, the latest scan, news and web search</div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {msgs.length > 0 && (
            <button onClick={() => { abort.current?.abort(); setMsgs([]); }} disabled={busy}
              className="rounded-md border border-zinc-800 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-900 disabled:opacity-40" title="Start a new conversation">
              New chat
            </button>
          )}
          {onHide && (
            <button onClick={onHide} className="hidden lg:block rounded-md border border-zinc-800 px-2 py-1 text-xs text-zinc-300 hover:bg-zinc-900" title="Hide the chat panel">
              Hide »
            </button>
          )}
        </div>
      </div>

      <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto overflow-x-hidden p-4 space-y-3">
        {msgs.length === 0 && (
          <div className="space-y-2">
            <p className="text-xs text-zinc-500">Try one of these, or ask about any NSE stock or sector:</p>
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} className="block w-full text-left text-sm rounded-lg border border-zinc-800 px-3 py-2 text-zinc-300 hover:bg-zinc-900">
                {s}
              </button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "flex justify-end" : "group"}>
            <div className={`rounded-xl px-3 py-2 text-sm max-w-[92%] min-w-0 ${m.role === "user"
              ? "bg-emerald-700 text-white whitespace-pre-wrap break-words"
              : m.error ? "bg-red-950/40 text-red-200 border border-red-900" : "bg-zinc-900 text-zinc-200 border border-zinc-800"}`}>
              {m.tools && m.tools.length > 0 && (
                <div className="mb-1 space-y-0.5">
                  {m.tools.map((t, j) => (
                    <div key={j} className="text-[11px] text-zinc-500 truncate" title={t}>🔎 {t}</div>
                  ))}
                </div>
              )}
              {m.role === "user" ? m.content : m.content ? (
                <div className="chat-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>{m.content}</ReactMarkdown>
                </div>
              ) : (
                <span className="text-zinc-500 animate-pulse">{m.tools?.length ? "Researching…" : "Thinking…"}</span>
              )}
            </div>
            {m.role === "assistant" && m.content && !(busy && i === last) && (
              <div className="mt-1 flex gap-3 pl-1 text-[11px] text-zinc-500">
                {!m.error && m.content !== "Stopped." && (
                  <button onClick={() => copy(i)} className="hover:text-zinc-300">{copied === i ? "Copied ✓" : "Copy"}</button>
                )}
                {i === last && (m.error || m.content === "Stopped.") && (
                  <button onClick={retry} className="hover:text-zinc-300">↻ Retry</button>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="p-3 border-t border-zinc-800 flex items-end gap-2">
        <textarea
          ref={box}
          rows={1}
          value={input}
          maxLength={MAX_INPUT}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
              e.preventDefault();
              send(input);
            }
          }}
          placeholder="Ask about any stock or sector…"
          title="Enter to send · Shift+Enter for a new line"
          aria-label="Chat message"
          className="flex-1 resize-none rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-600"
        />
        {busy ? (
          <button type="button" onClick={() => abort.current?.abort()} className="rounded-lg border border-zinc-600 px-3 py-2 text-sm text-zinc-200 hover:bg-zinc-800">
            ■ Stop
          </button>
        ) : (
          <button disabled={!input.trim()} className="rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
            Send
          </button>
        )}
      </form>
    </div>
  );
}
