import { useState } from "react";
import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";
import { Play, RotateCcw, Radio, FileText, ShieldCheck, TrendingUp, Activity } from "lucide-react";
import { simulateSignal } from "@/api/client";
import { readNpyFile } from "@/utils/npy-parser";
import { fmtProb, fmtPct, friendlyError } from "@/utils/format";
import type { SimulateResponse } from "@/types";
import ECGChart from "@/components/ECGChart";
import FileDropZone from "@/components/FileDropZone";
import MetricCard from "@/components/MetricCard";
import PredictionBadge from "@/components/PredictionBadge";
import Spinner from "@/components/Spinner";
import ErrorBanner from "@/components/ErrorBanner";

const EXPECTED_LENGTH = 1000;

export default function Simulation() {
  const [signal, setSignal] = useState<number[] | null>(null);
  const [fileName, setFileName] = useState("");
  const [simResult, setSimResult] = useState<SimulateResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setSimResult(null);
    try {
      const parsed = await readNpyFile(file);
      if (parsed.length !== EXPECTED_LENGTH) {
        setError(`Expected ${EXPECTED_LENGTH} samples, got ${parsed.length}.`);
        return;
      }
      setSignal(parsed);
      setFileName(file.name);
    } catch (err) {
      setError(friendlyError(err));
    }
  }

  async function runSimulation() {
    if (!signal) return;
    setError(null);
    setSimResult(null);
    setLoading(true);
    try {
      const res = await simulateSignal(signal);
      setSimResult(res);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setSignal(null);
    setSimResult(null);
    setError(null);
    setFileName("");
  }

  const timelineData = simResult?.timeline_predictions.map((t) => ({
    chunk: t.chunk_index + 1,
    probability: Number(t.probability.toFixed(4)),
    label: `Chunk ${t.chunk_index + 1}`,
    samples: `${t.start_sample}–${t.end_sample}`,
  }));

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-purple-600 to-fuchsia-500 shadow-md shadow-purple-500/25">
            <Radio className="h-5 w-5 text-white" />
          </div>
          <h1 className="page-title">Real-Time Simulation</h1>
        </div>
        <p className="page-subtitle mt-3">
          Watch the model's prediction evolve chunk-by-chunk as more of the ECG
          signal arrives, simulating a real-time streaming scenario.
        </p>
      </div>

      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {!signal && <FileDropZone onFile={handleFile} />}

      {signal && (
        <div className="animate-fade-in space-y-8">
          {/* Toolbar */}
          <div className="flex flex-wrap items-center gap-4 rounded-2xl border border-gray-100 bg-white px-6 py-4 shadow-card">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50">
                <FileText className="h-5 w-5 text-brand-600" />
              </div>
              <div>
                <p className="text-sm font-bold text-gray-900">{fileName}</p>
                <p className="text-xs text-gray-400">
                  {signal.length.toLocaleString()} samples · Ready to simulate
                </p>
              </div>
            </div>
            <div className="ml-auto flex gap-3">
              <button
                onClick={runSimulation}
                disabled={loading}
                className="btn-primary"
              >
                <Play className="h-4 w-4" />
                {loading ? "Simulating…" : "Run Simulation"}
              </button>
              <button onClick={handleReset} className="btn-secondary">
                <RotateCcw className="h-4 w-4" />
                Reset
              </button>
            </div>
          </div>

          <ECGChart signal={signal} title="Input ECG Signal" />

          {loading && (
            <div className="rounded-2xl border border-purple-100 bg-purple-50/50 py-8">
              <Spinner text="Running simulation across 10 chunks…" />
            </div>
          )}

          {simResult && timelineData && (
            <div className="animate-slide-up space-y-8">
              {/* Probability evolution chart */}
              <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card">
                <div className="flex items-center gap-3 border-b border-gray-100 px-6 py-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-red-50">
                    <TrendingUp className="h-4 w-4 text-red-500" />
                  </div>
                  <h3 className="text-sm font-bold text-gray-800">Probability Evolution</h3>
                  <span className="ml-auto rounded-full bg-gray-100 px-3 py-0.5 text-[11px] font-semibold text-gray-500">
                    {timelineData.length} chunks
                  </span>
                </div>
                <div className="p-6">
                  <ResponsiveContainer width="100%" height={340}>
                    <ComposedChart data={timelineData}>
                      <defs>
                        <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#dc2626" stopOpacity={0.3} />
                          <stop offset="100%" stopColor="#dc2626" stopOpacity={0.02} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                      <XAxis
                        dataKey="chunk"
                        label={{ value: "Chunk", position: "insideBottomRight", offset: -5, style: { fontSize: 11, fill: "#94a3b8" } }}
                        tick={{ fontSize: 11, fill: "#94a3b8" }}
                        axisLine={{ stroke: "#e2e8f0" }}
                        tickLine={false}
                      />
                      <YAxis
                        domain={[0, 1]}
                        label={{
                          value: "Probability",
                          angle: -90,
                          position: "insideLeft",
                          offset: 10,
                          style: { fontSize: 11, fill: "#94a3b8" },
                        }}
                        tick={{ fontSize: 11, fill: "#94a3b8" }}
                        axisLine={false}
                        tickLine={false}
                      />
                      <Tooltip
                        contentStyle={{
                          fontSize: 12,
                          borderRadius: 12,
                          border: "none",
                          boxShadow: "0 10px 40px -10px rgba(0,0,0,0.15)",
                          padding: "10px 14px",
                        }}
                        formatter={(v: number) => [v.toFixed(4), "Probability"]}
                        labelFormatter={(l) => `Chunk ${l}`}
                      />
                      <ReferenceLine
                        y={0.5}
                        stroke="#cbd5e1"
                        strokeDasharray="6 3"
                        label={{ value: "Threshold 0.5", position: "right", fontSize: 10, fill: "#94a3b8" }}
                      />
                      <Area
                        type="monotone"
                        dataKey="probability"
                        fill="url(#probGrad)"
                        stroke="none"
                      />
                      <Line
                        type="monotone"
                        dataKey="probability"
                        stroke="#dc2626"
                        strokeWidth={2.5}
                        dot={{ r: 5, fill: "#dc2626", strokeWidth: 2.5, stroke: "#fff" }}
                        activeDot={{ r: 8, strokeWidth: 3, stroke: "#fff" }}
                        animationDuration={1200}
                      />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Chunk table */}
              <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card">
                <div className="flex items-center gap-3 border-b border-gray-100 px-6 py-4">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
                    <Activity className="h-4 w-4 text-brand-500" />
                  </div>
                  <h3 className="text-sm font-bold text-gray-800">Chunk Details</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-100 bg-gray-50/80 text-left text-[11px] font-bold uppercase tracking-[0.1em] text-gray-400">
                        <th className="px-6 py-3.5">Chunk</th>
                        <th className="px-6 py-3.5">Samples</th>
                        <th className="px-6 py-3.5">Probability</th>
                        <th className="px-6 py-3.5">Prediction</th>
                        <th className="px-6 py-3.5">Confidence</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-50">
                      {simResult.timeline_predictions.map((t) => (
                        <tr
                          key={t.chunk_index}
                          className="transition-colors duration-150 hover:bg-brand-50/40"
                        >
                          <td className="px-6 py-3.5 font-bold text-gray-900">
                            #{t.chunk_index + 1}
                          </td>
                          <td className="px-6 py-3.5 font-mono text-xs text-gray-500">
                            {t.start_sample}–{t.end_sample}
                          </td>
                          <td className="px-6 py-3.5">
                            <span className="font-mono text-xs font-semibold text-gray-700">
                              {fmtProb(t.probability)}
                            </span>
                          </td>
                          <td className="px-6 py-3.5">
                            <PredictionBadge prediction={t.prediction} size="sm" />
                          </td>
                          <td className="px-6 py-3.5 font-mono text-xs font-semibold text-gray-600">
                            {fmtPct(t.confidence)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Final result */}
              <div className="space-y-5">
                <div className="flex items-center gap-4 rounded-2xl border border-gray-100 bg-white px-6 py-5 shadow-card">
                  <div
                    className={`flex h-14 w-14 items-center justify-center rounded-2xl ${
                      simResult.final_prediction.prediction === 0
                        ? "bg-gradient-to-br from-emerald-500 to-teal-500 shadow-glow-emerald"
                        : "bg-gradient-to-br from-red-500 to-rose-500 shadow-glow-red"
                    }`}
                  >
                    <ShieldCheck className="h-7 w-7 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-semibold text-gray-500">Final Prediction</p>
                    <div className="mt-1 flex items-center gap-3">
                      <span
                        className={`text-2xl font-extrabold ${
                          simResult.final_prediction.prediction === 0
                            ? "text-emerald-700"
                            : "text-red-700"
                        }`}
                      >
                        {simResult.final_prediction.prediction === 0
                          ? "Normal Sinus Rhythm"
                          : "Arrhythmia Detected"}
                      </span>
                      <PredictionBadge
                        prediction={simResult.final_prediction.prediction}
                        size="sm"
                      />
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
                  <MetricCard
                    label="Prediction"
                    value={
                      simResult.final_prediction.prediction === 0
                        ? "Normal"
                        : "Arrhythmia"
                    }
                    variant={
                      simResult.final_prediction.prediction === 0
                        ? "success"
                        : "danger"
                    }
                    icon={<ShieldCheck className="h-5 w-5" />}
                  />
                  <MetricCard
                    label="Probability"
                    value={fmtProb(simResult.final_prediction.probability)}
                    variant="info"
                    icon={<TrendingUp className="h-5 w-5" />}
                  />
                  <MetricCard
                    label="Confidence"
                    value={fmtPct(simResult.final_prediction.confidence)}
                    icon={<Activity className="h-5 w-5" />}
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
