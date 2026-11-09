"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import ChatPanel from "@/components/ChatPanel";

// Desktop: a side panel the user can drag wider or narrower (or hide). Mobile: a full-screen overlay.
const DEFAULT_W = 400;
const MIN_W = 320;
const KEY = "chat-dock-v1";

function maxWidth() {
  return typeof window === "undefined" ? 900 : Math.max(MIN_W, Math.min(1000, Math.round(window.innerWidth * 0.65)));
}
const clamp = (w: number) => Math.max(MIN_W, Math.min(maxWidth(), Math.round(w)));

function load(): { w: number; hidden: boolean } {
  try {
    const v = JSON.parse(localStorage.getItem(KEY) || "{}");
    return { w: typeof v.w === "number" ? v.w : DEFAULT_W, hidden: !!v.hidden };
  } catch {
    return { w: DEFAULT_W, hidden: false };
  }
}

export default function ChatDock() {
  const [width, setWidth] = useState(DEFAULT_W);
  const [hidden, setHidden] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [dragging, setDragging] = useState(false);
  const ready = useRef(false);

  useEffect(() => {
    const s = load();
    setWidth(clamp(s.w));
    setHidden(s.hidden);
    ready.current = true;
    const onResize = () => setWidth((w) => clamp(w));
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    if (!ready.current) return;
    try {
      localStorage.setItem(KEY, JSON.stringify({ w: width, hidden }));
    } catch {
      /* private mode: the layout simply is not remembered */
    }
  }, [width, hidden]);

  const onPointerDown = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    setDragging(true);
  }, []);
  const onPointerMove = useCallback((e: React.PointerEvent<HTMLDivElement>) => {
    if (dragging) setWidth(clamp(window.innerWidth - e.clientX));
  }, [dragging]);
  const stopDrag = useCallback(() => setDragging(false), []);
  const onKey = (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 80 : 24;
    if (e.key === "ArrowLeft") setWidth((w) => clamp(w + step));
    else if (e.key === "ArrowRight") setWidth((w) => clamp(w - step));
    else if (e.key === "Home") setWidth(clamp(DEFAULT_W));
    else return;
    e.preventDefault();
  };

  return (
    <>
      {/* desktop: reopen tab when the panel is hidden */}
      {hidden && (
        <button onClick={() => setHidden(false)} title="Show research chat"
          className="hidden lg:flex fixed right-0 top-1/2 -translate-y-1/2 z-20 flex-col items-center gap-1 rounded-l-lg border border-r-0 border-zinc-700 bg-zinc-900 px-1.5 py-3 text-xs text-zinc-300 hover:bg-zinc-800">
          <span>💬</span>
          <span className="[writing-mode:vertical-rl]">Chat</span>
        </button>
      )}

      <aside
        style={{ "--chat-w": `${width}px` } as React.CSSProperties}
        className={`${mobileOpen ? "flex fixed inset-0 z-30" : "hidden"} ${hidden ? "lg:hidden" : "lg:flex"} lg:relative
          w-full lg:w-[var(--chat-w)] lg:shrink-0 min-w-0 overflow-hidden border-l border-zinc-800 bg-zinc-950`}
      >
        {/* drag handle (desktop only) */}
        <div
          role="separator" aria-orientation="vertical" aria-label="Resize chat panel" tabIndex={0}
          aria-valuemin={MIN_W} aria-valuemax={maxWidth()} aria-valuenow={width}
          title="Drag to resize · double-click to reset"
          onPointerDown={onPointerDown} onPointerMove={onPointerMove} onPointerUp={stopDrag} onPointerCancel={stopDrag}
          onDoubleClick={() => setWidth(clamp(DEFAULT_W))} onKeyDown={onKey}
          className="hidden lg:block absolute left-0 top-0 z-10 h-full w-2 -translate-x-1 cursor-col-resize group outline-none"
        >
          <div className={`mx-auto h-full w-0.5 transition-colors ${dragging ? "bg-emerald-500" : "bg-transparent group-hover:bg-emerald-600/70 group-focus-visible:bg-emerald-500"}`} />
        </div>
        <ChatPanel onHide={() => setHidden(true)} />
      </aside>
      {/* while dragging, a full-screen layer keeps the cursor and stops text selection under it */}
      {dragging && <div className="fixed inset-0 z-40 cursor-col-resize select-none" onPointerUp={stopDrag} />}

      <button
        onClick={() => setMobileOpen((o) => !o)}
        className={`lg:hidden fixed right-4 z-30 rounded-full bg-emerald-600 text-sm font-semibold text-white shadow-lg ${mobileOpen ? "top-3 px-3 py-1.5" : "bottom-4 px-4 py-3"}`}
      >
        {mobileOpen ? "✕ Close" : "💬 Chat"}
      </button>
    </>
  );
}
