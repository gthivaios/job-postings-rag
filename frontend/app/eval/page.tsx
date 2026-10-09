"use client";
import { useEffect, useState } from "react";

type Row = { id: string; question: string; mrr: number | null; ndcg: number | null; coverage: number | null; accuracy: number; completeness: number; relevance: number; feedback: string };
type Report = { date: string; judge_model: string; tests: number; summary: Record<string, number | null>; results: Row[] };

const fmt = (v: number | null, d = 2) => (v === null ? "n/a" : v.toFixed(d));

export default function EvalPage() {
  const [report, setReport] = useState<Report | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${process.env.NEXT_PUBLIC_API_URL}/eval/report`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`API error ${r.status}`))))
      .then(setReport)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <main className="p-6">{error}</main>;
  if (!report) return <main className="p-6">Loading…</main>;

  return (
    <main className="p-6">
      <h1 className="text-xl font-semibold">Evaluation</h1>
      <p className="text-sm text-gray-500">{report.tests} tests · {report.date} · judge: {report.judge_model}</p>
      <div className="my-4 grid grid-cols-3 gap-3 md:grid-cols-6">
        {Object.entries(report.summary).map(([name, value]) => (
          <div key={name} className="rounded border p-3">
            <div className="text-xs uppercase text-gray-500">{name}</div>
            <div className="text-2xl font-bold">{fmt(value)}</div>
          </div>
        ))}
      </div>
      <table className="w-full text-left text-sm">
        <thead>
          <tr className="border-b"><th>Test</th><th>MRR</th><th>nDCG</th><th>Coverage</th><th>Acc.</th><th>Compl.</th><th>Rel.</th></tr>
        </thead>
        <tbody>
          {report.results.map((r) => (
            <tr key={r.id} className="border-b" title={r.feedback}>
              <td className="py-1">{r.id}</td>
              <td>{fmt(r.mrr)}</td><td>{fmt(r.ndcg)}</td><td>{fmt(r.coverage, 0)}</td>
              <td>{r.accuracy}</td><td>{r.completeness}</td><td>{r.relevance}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </main>
  );
}