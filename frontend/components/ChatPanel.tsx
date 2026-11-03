"use client";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { API } from "@/lib/api";

type Msg = { role: "user" | "assistant"; content: string; tools?: string[] };

const SUGGESTIONS = [
  "What are today's top 3 swing picks?",
  "Any good swing trade in the banking sector?",
  "What is the 15-20 day outlook for TATAMOTORS?",
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
      const r = await fetch(`${API}/api/chat`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ messages: history.map(({ role, content }) => ({ role, content })) }),
      });
      const reader = r.body!.getReader();
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
      setMsgs((m) => [...m.slice(0, -1), { role: "assistant", content: `Error: ${e}` }]);
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
                <div className="prose prose-invert prose-sm max-w-none [&_p]:my-1 [&_ul]:my-1 [&_li]:my-0">
                  <ReactMarkdown>{m.content}</ReactMarkdown>
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
