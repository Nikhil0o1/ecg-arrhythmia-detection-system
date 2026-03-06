import { useEffect, useState } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Legend,
} from "recharts";
import { Trophy, Info, BarChart3 } from "lucide-react";
import type { ComparisonMetrics } from "@/types";
import ErrorBanner from "@/components/ErrorBanner";
import Spinner from "@/components/Spinner";

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  roc_auc: "ROC-AUC",
};

const MODEL_COLORS: Record<string, string> = {
  CNN1D: "#6366f1",
  LSTMClassifier: "#f59e0b",
  TransformerClassifier: "#10b981",
};

const MODEL_COLORS_LIST = ["#6366f1", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899"];

export default function ModelComparison() {
  const [metrics, setMetrics] = useState<ComparisonMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadMetrics();
  }, []);

  async function loadMetrics() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/comparison_metrics.json");
      if (!res.ok) throw new Error("Failed to load metrics file.");
      const data = (await res.json()) as ComparisonMetrics;
      setMetrics(data);
    } catch {
      try {
        const res2 = await fetch("/api/comparison_metrics.json");
        if (!res2.ok) throw new Error();
        setMetrics((await res2.json()) as ComparisonMetrics);
      } catch {
        setError(
          "Could not load comparison_metrics.json. Copy it to frontend/public/.",
        );
      }
    } finally {
      setLoading(false);
    }
  }

  if (loading)
    return (
      <div className="rounded-2xl border border-brand-100 bg-brand-50/50 py-16">
        <Spinner text="Loading model metrics…" />
      </div>
    );
  if (error) return <ErrorBanner message={error} />;
  if (!metrics) return null;

  const modelNames = Object.keys(metrics);
  const metricKeys = Object.keys(METRIC_LABELS);

  const tableRows = modelNames.map((name) => ({
    name,
    ...metrics[name],
  }));

  const bestModel = modelNames.reduce((a, b) =>
    (metrics[a]?.f1_score ?? 0) > (metrics[b]?.f1_score ?? 0) ? a : b,
  );

  function barData(metricKey: string) {
    return modelNames.map((name) => {
      const m = metrics![name];
      const val = m ? (m[metricKey as keyof typeof m] ?? 0) : 0;
      return { model: name, value: Number(val.toFixed(4)) };
    });
  }

  const radarData = metricKeys.map((key) => {
    const entry: Record<string, string | number> = { metric: METRIC_LABELS[key] ?? key };
    modelNames.forEach((m) => {
      entry[m] = Number((metrics[m]?.[key as keyof typeof metrics[typeof m]] ?? 0).toFixed(4));
    });
    return entry;
  });

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 shadow-md shadow-amber-500/25">
            <BarChart3 className="h-5 w-5 text-white" />
          </div>
          <h1 className="page-title">Model Comparison</h1>
        </div>
        <p className="page-subtitle mt-3">
          Performance metrics for all trained architectures on the hold-out test set.
        </p>
      </div>

      {/* Best model highlight */}
      <div className="relative overflow-hidden rounded-2xl border border-amber-200/60 bg-gradient-to-r from-amber-50 via-orange-50 to-yellow-50 p-5 shadow-card">
        <div className="absolute -right-4 -top-4 h-24 w-24 rounded-full bg-amber-200/30 blur-2xl" />
        <div className="relative flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-gradient-to-br from-amber-500 to-orange-500 shadow-lg shadow-amber-500/30">
            <Trophy className="h-6 w-6 text-white" />
          </div>
          <div>
            <p className="text-sm font-bold text-amber-900">
              Best Model — <span className="text-gradient bg-gradient-to-r from-amber-600 to-orange-600 bg-clip-text text-transparent">{bestModel}</span>
            </p>
            <p className="mt-0.5 text-xs font-medium text-amber-600/80">
              Highest F1 Score: <span className="font-bold">{(metrics[bestModel]?.f1_score ?? 0).toFixed(4)}</span>
            </p>
          </div>
        </div>
      </div>

      {/* Metrics table */}
      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card">
        <div className="flex items-center gap-3 border-b border-gray-100 px-6 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
            <BarChart3 className="h-4 w-4 text-brand-500" />
          </div>
          <h3 className="text-sm font-bold text-gray-800">Metrics Summary</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="border-b border-gray-100 bg-gray-50/80 text-left text-[11px] font-bold uppercase tracking-[0.1em] text-gray-400">
                <th className="px-6 py-3.5">Model</th>
                {metricKeys.map((k) => (
                  <th key={k} className="px-6 py-3.5">
                    {METRIC_LABELS[k]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {tableRows.map((row) => (
                <tr
                  key={row.name}
                  className={`transition-colors duration-150 hover:bg-brand-50/30 ${
                    row.name === bestModel ? "bg-amber-50/50" : ""
                  }`}
                >
                  <td className="px-6 py-3.5">
                    <div className="flex items-center gap-2.5">
                      <span
                        className="inline-block h-3 w-3 rounded-full shadow-sm"
                        style={{ background: MODEL_COLORS[row.name] ?? "#6b7280" }}
                      />
                      <span className="font-bold text-gray-900">{row.name}</span>
                      {row.name === bestModel && (
                        <Trophy className="h-3.5 w-3.5 text-amber-500" />
                      )}
                    </div>
                  </td>
                  {metricKeys.map((k) => {
                    const val = row[k as keyof typeof row] as number;
                    const isBest =
                      modelNames.every(
                        (m) =>
                          (metrics[m]?.[k as keyof typeof metrics[typeof m]] ?? 0) <=
                          (val ?? 0),
                      ) && val !== undefined;
                    return (
                      <td
                        key={k}
                        className={`px-6 py-3.5 font-mono text-xs ${
                          isBest ? "font-bold text-brand-600" : "text-gray-600"
                        }`}
                      >
                        {val?.toFixed(4) ?? "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Bar charts */}
      <div>
        <p className="mb-4 text-[11px] font-bold uppercase tracking-[0.12em] text-gray-400">
          Key Metrics Comparison
        </p>
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {(["accuracy", "f1_score", "roc_auc"] as const).map((mk) => (
            <div
              key={mk}
              className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card transition-shadow duration-300 hover:shadow-card-hover"
            >
              <div className="border-b border-gray-100 px-5 py-3">
                <h4 className="text-center text-xs font-bold uppercase tracking-wider text-gray-500">
                  {METRIC_LABELS[mk]}
                </h4>
              </div>
              <div className="p-5">
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={barData(mk)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
                    <XAxis
                      type="number"
                      domain={[0, 1]}
                      tick={{ fontSize: 10, fill: "#94a3b8" }}
                      axisLine={{ stroke: "#e2e8f0" }}
                      tickLine={false}
                    />
                    <YAxis
                      type="category"
                      dataKey="model"
                      width={130}
                      tick={{ fontSize: 11, fill: "#64748b", fontWeight: 600 }}
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
                      formatter={(v: number) => [v.toFixed(4), METRIC_LABELS[mk]]}
                    />
                    <Bar dataKey="value" radius={[0, 8, 8, 0]} barSize={24}>
                      {barData(mk).map((_, idx) => (
                        <Cell
                          key={idx}
                          fill={MODEL_COLORS_LIST[idx % MODEL_COLORS_LIST.length]}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Radar chart */}
      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card">
        <div className="border-b border-gray-100 px-6 py-4">
          <h3 className="text-center text-sm font-bold text-gray-800">
            Comprehensive Radar View
          </h3>
        </div>
        <div className="p-6">
          <ResponsiveContainer width="100%" height={420}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#e2e8f0" />
              <PolarAngleAxis
                dataKey="metric"
                tick={{ fontSize: 11, fill: "#64748b", fontWeight: 600 }}
              />
              <PolarRadiusAxis
                domain={[0, 1]}
                tick={{ fontSize: 10, fill: "#94a3b8" }}
                axisLine={false}
              />
              {modelNames.map((m, i) => (
                <Radar
                  key={m}
                  name={m}
                  dataKey={m}
                  stroke={MODEL_COLORS_LIST[i % MODEL_COLORS_LIST.length]}
                  fill={MODEL_COLORS_LIST[i % MODEL_COLORS_LIST.length]}
                  fillOpacity={0.1}
                  strokeWidth={2.5}
                />
              ))}
              <Legend
                wrapperStyle={{ fontSize: 12, fontWeight: 600, paddingTop: 16 }}
              />
              <Tooltip
                contentStyle={{
                  fontSize: 12,
                  borderRadius: 12,
                  border: "none",
                  boxShadow: "0 10px 40px -10px rgba(0,0,0,0.15)",
                }}
                formatter={(v: number) => v.toFixed(4)}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Note */}
      <div className="flex items-start gap-4 rounded-2xl border border-brand-100 bg-brand-50/50 p-5">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-brand-100">
          <Info className="h-4.5 w-4.5 text-brand-600" />
        </div>
        <div>
          <p className="text-sm font-semibold text-brand-900">About these metrics</p>
          <p className="mt-1 text-xs leading-relaxed text-brand-700/80">
            All metrics were computed on the hold-out test split. The best model
            for production is selected based on the highest F1 score, balancing
            precision and recall for clinical reliability.
          </p>
        </div>
      </div>
    </div>
  );
}
