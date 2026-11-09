"use client";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown, { Components } from "react-markdown";
import remarkGfm from "remark-gfm";

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
  a: ({ children, href }) => <a href={href} target="_blank" className="text-sky-400 hover:underline">{children}</a>,
  blockquote: ({ children }) => <blockquote className="my-1.5 border-l-2 border-amber-600 pl-2 text-zinc-400">{children}</blockquote>,
};
import { apiFetch } from "@/lib/api";

type Msg = { role: "user" | "assistant"; content: string; tools?: string[] };

const SUGGESTIONS = [
  "What are today's top 3 swing picks?",
  "Any good swing trade in the banking sector?",
  "What is the 15-20 day outlook for RELIANCE?",
];

export default function ChatPanel() {
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth" });
  }, [msgs]);

  async function send(text: string) {
    if (!text.trim() || busy) return;
    const history: Msg[] = [...msgs, { role: "user", content: text }];
    setMsgs([...history, { role: "assistant", content: "", tools: [] }]);
    setInput("");
    setBusy(true);
    try {
      const r = await apiFetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: history.map(({ role, content }) => ({ role, content })) }),
      });
      if (!r.ok || !r.body) throw new Error(r.status === 401 ? "Please sign in again." : `Chat failed (${r.status})`);
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
          const data = p.replace(/^data: /, "");
          if (data === "[DONE]") continue;
          const ev = JSON.parse(data);
          setMsgs((m) => {
            const copy = [...m];
            const last = { ...copy[copy.length - 1] };
            if (ev.type === "tool") last.tools = [...(last.tools || []), `${ev.name}(${Object.values(ev.input).join(", ")})`];
            else last.content = ev.text;
            copy[copy.length - 1] = last;
            return copy;
          });
        }
      }
    } catch (e) {
      setMsgs((m) => [...m.slice(0, -1), { role: "assistant", content: `⚠️ ${e instanceof Error ? e.message : e}` }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="px-4 py-3 border-b border-zinc-800">
        <div className="font-semibold text-white">Research Chat</div>
        <div className="text-xs text-zinc-500">The agent uses live data, news and web search</div>
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {msgs.length === 0 && (
          <div className="space-y-2">
            {SUGGESTIONS.map((s) => (
              <button key={s} onClick={() => send(s)} className="block w-full text-left text-sm rounded-lg border border-zinc-800 px-3 py-2 text-zinc-300 hover:bg-zinc-900">
                {s}
              </button>
            ))}
          </div>
        )}
        {msgs.map((m, i) => (
          <div key={i} className={m.role === "user" ? "flex justify-end" : ""}>
            <div className={`rounded-xl px-3 py-2 text-sm max-w-[92%] ${m.role === "user" ? "bg-emerald-700 text-white" : "bg-zinc-900 text-zinc-200 border border-zinc-800"}`}>
              {m.tools && m.tools.length > 0 && (
                <div className="mb-1 space-y-0.5">
                  {m.tools.map((t, j) => (
                    <div key={j} className="text-[11px] text-zinc-500 truncate">🔎 {t}</div>
                  ))}
                </div>
              )}
              {m.content ? (
                <div className="chat-md">
                  <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>{m.content}</ReactMarkdown>
                </div>
              ) : (
                m.role === "assistant" && <span className="text-zinc-500 animate-pulse">Researching…</span>
              )}
            </div>
          </div>
        ))}
        <div ref={end} />
      </div>
      <form onSubmit={(e) => { e.preventDefault(); send(input); }} className="p-3 border-t border-zinc-800 flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about any stock or sector…"
          className="flex-1 rounded-lg bg-zinc-900 border border-zinc-800 px-3 py-2 text-sm text-white outline-none focus:border-emerald-600"
        />
        <button disabled={busy} className="rounded-lg bg-emerald-600 px-4 text-sm font-medium text-white disabled:opacity-50">
          Send
        </button>
      </form>
    </div>
  );
}
