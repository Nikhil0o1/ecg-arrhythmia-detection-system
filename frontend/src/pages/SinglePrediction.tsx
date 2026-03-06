import { useState } from "react";
import { Activity, TrendingUp, ShieldCheck, FileText, Sparkles } from "lucide-react";
import { predictSignal } from "@/api/client";
import { readNpyFile } from "@/utils/npy-parser";
import { fmtProb, fmtPct, friendlyError } from "@/utils/format";
import type { PredictResponse } from "@/types";
import ECGChart from "@/components/ECGChart";
import FileDropZone from "@/components/FileDropZone";
import MetricCard from "@/components/MetricCard";
import PredictionBadge from "@/components/PredictionBadge";
import Spinner from "@/components/Spinner";
import ErrorBanner from "@/components/ErrorBanner";

const EXPECTED_LENGTH = 1000;

export default function SinglePrediction() {
  const [signal, setSignal] = useState<number[] | null>(null);
  const [fileName, setFileName] = useState<string>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleFile(file: File) {
    setError(null);
    setResult(null);

    try {
      const parsed = await readNpyFile(file);

      if (parsed.length !== EXPECTED_LENGTH) {
        setError(
          `Expected ${EXPECTED_LENGTH} samples, got ${parsed.length}. Please upload a valid ECG signal.`,
        );
        return;
      }

      setSignal(parsed);
      setFileName(file.name);

      setLoading(true);
      const prediction = await predictSignal(parsed);
      setResult(prediction);
    } catch (err) {
      setError(friendlyError(err));
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setSignal(null);
    setResult(null);
    setError(null);
    setFileName("");
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-brand-600 to-brand-400 shadow-md shadow-brand-500/25">
            <Sparkles className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="page-title">Single Prediction</h1>
          </div>
        </div>
        <p className="page-subtitle mt-3">
          Upload a <code className="rounded-lg bg-gray-100 px-2 py-0.5 font-mono text-xs font-semibold text-brand-600">.npy</code> file
          containing 1 000 ECG samples (10 s @ 100 Hz) for instant AI-powered classification.
        </p>
      </div>

      {error && <ErrorBanner message={error} onDismiss={() => setError(null)} />}

      {!signal && <FileDropZone onFile={handleFile} />}

      {signal && (
        <div className="animate-fade-in space-y-8">
          {/* File info bar */}
          <div className="flex items-center justify-between rounded-2xl border border-gray-100 bg-white px-6 py-4 shadow-card">
            <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-50">
                <FileText className="h-5 w-5 text-brand-600" />
              </div>
              <div>
                <p className="text-sm font-bold text-gray-900">{fileName}</p>
                <p className="text-xs text-gray-400">
                  {signal.length.toLocaleString()} samples · 10 seconds · 100 Hz
                </p>
              </div>
            </div>
            <button onClick={handleReset} className="btn-secondary text-xs">
              Upload New
            </button>
          </div>

          {/* ECG waveform */}
          <ECGChart signal={signal} title="ECG Waveform — Lead I" />

          {/* Loading */}
          {loading && (
            <div className="rounded-2xl border border-brand-100 bg-brand-50/50 py-8">
              <Spinner text="Running neural network inference…" />
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="animate-slide-up space-y-6">
              {/* Result header */}
              <div className="flex items-center gap-4 rounded-2xl border border-gray-100 bg-white px-6 py-5 shadow-card">
                <div
                  className={`flex h-14 w-14 items-center justify-center rounded-2xl ${
                    result.prediction === 0
                      ? "bg-gradient-to-br from-emerald-500 to-teal-500 shadow-glow-emerald"
                      : "bg-gradient-to-br from-red-500 to-rose-500 shadow-glow-red"
                  }`}
                >
                  <ShieldCheck className="h-7 w-7 text-white" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-gray-500">Classification Result</p>
                  <div className="mt-1 flex items-center gap-3">
                    <span
                      className={`text-2xl font-extrabold ${
                        result.prediction === 0 ? "text-emerald-700" : "text-red-700"
                      }`}
                    >
                      {result.prediction === 0 ? "Normal Sinus Rhythm" : "Arrhythmia Detected"}
                    </span>
                    <PredictionBadge prediction={result.prediction} size="sm" />
                  </div>
                </div>
              </div>

              {/* Metrics grid */}
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
                <MetricCard
                  label="Prediction"
                  value={result.prediction === 0 ? "Normal" : "Arrhythmia"}
                  variant={result.prediction === 0 ? "success" : "danger"}
                  icon={<ShieldCheck className="h-5 w-5" />}
                />
                <MetricCard
                  label="Probability"
                  value={fmtProb(result.probability)}
                  variant="info"
                  subtext="Arrhythmia probability score"
                  icon={<TrendingUp className="h-5 w-5" />}
                />
                <MetricCard
                  label="Confidence"
                  value={fmtPct(result.confidence)}
                  variant="default"
                  subtext="Model certainty"
                  icon={<Activity className="h-5 w-5" />}
                />
              </div>

              {/* Probability visualization */}
              <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white p-6 shadow-card">
                <p className="mb-4 text-[11px] font-bold uppercase tracking-[0.12em] text-gray-400">
                  Probability Spectrum
                </p>
                <div className="mb-3 flex items-center justify-between text-sm">
                  <span className="flex items-center gap-2 font-bold text-emerald-600">
                    <span className="h-3 w-3 rounded-full bg-emerald-500" />
                    Normal
                  </span>
                  <span className="flex items-center gap-2 font-bold text-red-600">
                    Arrhythmia
                    <span className="h-3 w-3 rounded-full bg-red-500" />
                  </span>
                </div>
                <div className="relative h-5 w-full overflow-hidden rounded-full bg-gradient-to-r from-emerald-100 via-gray-100 to-red-100">
                  <div
                    className="absolute left-0 top-0 h-full rounded-full bg-gradient-to-r from-emerald-500 to-red-500 transition-all duration-1000 ease-out"
                    style={{ width: `${Math.min(result.probability * 100, 100)}%` }}
                  />
                  <div
                    className="absolute top-1/2 h-7 w-1 -translate-y-1/2 rounded-full bg-white shadow-lg ring-2 ring-gray-300 transition-all duration-1000 ease-out"
                    style={{ left: `calc(${Math.min(result.probability * 100, 100)}% - 2px)` }}
                  />
                </div>
                <p className="mt-3 text-center text-xs font-semibold text-gray-500">
                  {fmtPct(result.probability)} arrhythmia probability
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
