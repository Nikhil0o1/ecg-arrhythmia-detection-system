import { useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { BarChart3, Hexagon } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import type { ModelMetrics } from "@/types/ecg";

const MODEL_COLORS: Record<string, string> = {
  IndustryCNN: "#10b981",
  CNN1D: "#3b82f6",
  LSTMClassifier: "#f59e0b",
  TransformerClassifier: "#8b5cf6",
};

const METRIC_LABELS: Record<string, string> = {
  accuracy: "Accuracy",
  precision: "Precision",
  recall: "Recall",
  f1_score: "F1 Score",
  roc_auc: "ROC-AUC",
};

type ChartMode = "bar" | "radar";

export default function ModelComparisonChart() {
  const { metrics } = useECGStore();
  const [chartMode, setChartMode] = useState<ChartMode>("bar");

  /* ── Build comparison data including primary model ────────── */
  const allModels = useMemo(() => {
    if (!metrics) return {};
    const models: Record<string, ModelMetrics> = {
      [metrics.primary_model_name]: metrics.primary,
      ...metrics.comparison,
    };
    return models;
  }, [metrics]);

  const modelNames = Object.keys(allModels);

  /* ── Bar chart data ──────────────────────────────────────── */
  const barData = useMemo(() => {
    const metricKeys = Object.keys(METRIC_LABELS);
    return metricKeys.map((key) => {
      const entry: Record<string, string | number> = {
        metric: METRIC_LABELS[key] ?? key,
      };
      for (const [model, m] of Object.entries(allModels)) {
        entry[model] = +((m[key as keyof ModelMetrics] as number ?? 0) * 100).toFixed(1);
      }
      return entry;
    });
  }, [allModels]);

  /* ── Radar chart data ────────────────────────────────────── */
  const radarData = useMemo(() => {
    const metricKeys = Object.keys(METRIC_LABELS);
    return metricKeys.map((key) => {
      const entry: Record<string, string | number> = {
        metric: METRIC_LABELS[key] ?? key,
        fullMark: 100,
      };
      for (const [model, m] of Object.entries(allModels)) {
        entry[model] = +((m[key as keyof ModelMetrics] as number ?? 0) * 100).toFixed(1);
      }
      return entry;
    });
  }, [allModels]);

  if (!metrics || modelNames.length === 0) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
          Model Comparison
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-[10px] dark:text-gray-600 text-gray-400">
            {modelNames.length} models
          </span>
          <button
            onClick={() =>
              setChartMode((m) => (m === "bar" ? "radar" : "bar"))
            }
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium dark:bg-white/5 bg-gray-100 dark:text-gray-400 text-gray-600 dark:hover:bg-white/10 hover:bg-gray-200 transition-colors"
          >
            {chartMode === "bar" ? (
              <>
                <Hexagon className="h-3.5 w-3.5" />
                Radar
              </>
            ) : (
              <>
                <BarChart3 className="h-3.5 w-3.5" />
                Bar
              </>
            )}
          </button>
        </div>
      </div>

      {/* Model legend */}
      <div className="mb-4 flex flex-wrap gap-3">
        {modelNames.map((name) => (
          <div key={name} className="flex items-center gap-1.5">
            <span
              className="h-2.5 w-2.5 rounded-full"
              style={{ backgroundColor: MODEL_COLORS[name] ?? "#6b7280" }}
            />
            <span className="text-[11px] font-medium dark:text-gray-400 text-gray-600">
              {name}
            </span>
          </div>
        ))}
      </div>

      {/* Chart */}
      <AnimatePresence mode="wait">
        {chartMode === "bar" ? (
          <motion.div
            key="bar"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            transition={{ duration: 0.3 }}
            className="rounded-xl dark:bg-white/[0.02] bg-gray-50/50 p-3"
          >
            <ResponsiveContainer width="100%" height={280}>
              <BarChart
                data={barData}
                margin={{ top: 10, right: 10, bottom: 5, left: 5 }}
                barGap={2}
                barCategoryGap="20%"
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.05)"
                  vertical={false}
                />
                <XAxis
                  dataKey="metric"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fontSize: 10, fill: "#64748b" }}
                  axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
                  tickLine={false}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "rgba(15,23,42,0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: "8px",
                    fontSize: "11px",
                    color: "#e2e8f0",
                    padding: "8px 12px",
                  }}
                  formatter={(v: number) => [`${v.toFixed(1)}%`]}
                />
                {modelNames.map((name) => (
                  <Bar
                    key={name}
                    dataKey={name}
                    fill={MODEL_COLORS[name] ?? "#6b7280"}
                    radius={[4, 4, 0, 0]}
                    isAnimationActive
                    animationDuration={1200}
                    animationEasing="ease-out"
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </motion.div>
        ) : (
          <motion.div
            key="radar"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className="rounded-xl dark:bg-white/[0.02] bg-gray-50/50 p-3"
          >
            <ResponsiveContainer width="100%" height={320}>
              <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="70%">
                <PolarGrid
                  stroke="rgba(255,255,255,0.08)"
                  gridType="polygon"
                />
                <PolarAngleAxis
                  dataKey="metric"
                  tick={{ fontSize: 10, fill: "#64748b" }}
                />
                <PolarRadiusAxis
                  angle={90}
                  domain={[0, 100]}
                  tick={{ fontSize: 8, fill: "#64748b" }}
                  tickCount={5}
                />
                {modelNames.map((name) => (
                  <Radar
                    key={name}
                    name={name}
                    dataKey={name}
                    stroke={MODEL_COLORS[name] ?? "#6b7280"}
                    fill={MODEL_COLORS[name] ?? "#6b7280"}
                    fillOpacity={0.1}
                    strokeWidth={2}
                    isAnimationActive
                    animationDuration={1200}
                  />
                ))}
              </RadarChart>
            </ResponsiveContainer>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
