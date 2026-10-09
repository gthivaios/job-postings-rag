"use client";
import { useState } from "react";

type Message = { role: "user" | "assistant"; content: string };
type Source = { job_id: number | null; portal: string | null; company: string | null; location: string | null; text: string };

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Source[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function send(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;
    const question = input;
    const history = messages;
    setMessages([...history, { role: "user", content: question }]);
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, history }),
      });
      if (!res.ok) throw new Error(`API error ${res.status}`);
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", content: data.answer }]);
      setSources(data.sources);
    } catch (err) {
      setMessages((m) => [...m, { role: "assistant", content: `Something went wrong: ${err}` }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="grid h-screen grid-cols-2 gap-4 p-4">
      <section className="flex flex-col rounded border p-4">
        <div className="flex-1 space-y-3 overflow-y-auto">
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-right" : ""}>
              <span className="inline-block whitespace-pre-wrap rounded bg-gray-100 px-3 py-2 text-left text-black">{m.content}</span>
            </div>
          ))}
          {loading && <p className="text-sm text-gray-500">Searching postings…</p>}
        </div>
        <form onSubmit={send} className="mt-3 flex gap-2">
          <input className="flex-1 rounded border px-3 py-2" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask anything about job postings..." />
          <button className="rounded bg-orange-500 px-4 py-2 text-white" disabled={loading}>Send</button>
        </form>
      </section>
      <section className="overflow-y-auto rounded border p-4">
        <h2 className="mb-3 font-semibold text-orange-500">Retrieved context</h2>
        {sources.map((s, i) => (
          <article key={i} className="mb-4 border-b pb-3 text-sm">
            <p className="font-medium text-orange-500">Job ID: {s.job_id ?? "unknown"} | Portal: {s.portal ?? "unknown"} | {s.company} | {s.location}</p>
            <p className="mt-1 whitespace-pre-wrap">{s.text}</p>
          </article>
        ))}
      </section>
    </main>
  );
}