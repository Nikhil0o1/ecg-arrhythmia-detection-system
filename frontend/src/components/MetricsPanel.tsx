import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  BarChart3,
  Target,
  Eye,
  Crosshair,
  Activity,
  Gauge,
  Loader2,
  AlertCircle,
} from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import { fetchMetrics, extractErrorMessage } from "@/api/api";
import type { ModelMetrics } from "@/types/ecg";

interface MetricDisplayItem {
  label: string;
  value: number;
  icon: React.ReactNode;
  color: string;
  bgColor: string;
}

function buildMetricItems(m: ModelMetrics): MetricDisplayItem[] {
  const items: MetricDisplayItem[] = [
    {
      label: "ROC-AUC",
      value: m.roc_auc,
      icon: <BarChart3 className="h-4 w-4" />,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/15",
    },
    {
      label: "Sensitivity",
      value: m.sensitivity ?? m.recall,
      icon: <Eye className="h-4 w-4" />,
      color: "text-teal-400",
      bgColor: "bg-teal-500/15",
    },
    {
      label: "Specificity",
      value: m.specificity ?? 0,
      icon: <Target className="h-4 w-4" />,
      color: "text-cyan-400",
      bgColor: "bg-cyan-500/15",
    },
    {
      label: "Precision",
      value: m.precision,
      icon: <Crosshair className="h-4 w-4" />,
      color: "text-blue-400",
      bgColor: "bg-blue-500/15",
    },
    {
      label: "F1 Score",
      value: m.f1_score,
      icon: <Activity className="h-4 w-4" />,
      color: "text-violet-400",
      bgColor: "bg-violet-500/15",
    },
    {
      label: "Accuracy",
      value: m.accuracy,
      icon: <Gauge className="h-4 w-4" />,
      color: "text-amber-400",
      bgColor: "bg-amber-500/15",
    },
  ];

  // Filter out specificity if not available
  return items.filter((item) => item.value > 0);
}

export default function MetricsPanel() {
  const {
    metrics,
    metricsLoading,
    setMetrics,
    setMetricsLoading,
    addToast,
  } = useECGStore();

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      setMetricsLoading(true);
      try {
        const data = await fetchMetrics();
        if (mounted) setMetrics(data);
      } catch (err) {
        if (mounted) addToast(extractErrorMessage(err), "warning");
      } finally {
        if (mounted) setMetricsLoading(false);
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, [setMetrics, setMetricsLoading, addToast]);

  const metricItems = useMemo(() => {
    if (!metrics) return [];
    return buildMetricItems(metrics.primary);
  }, [metrics]);

  const modelName = metrics?.primary_model_name ?? "IndustryCNN";

  /* ── Loading state ─────────────────────────────────────────── */
  if (metricsLoading) {
    return (
      <div className="glass-card flex items-center justify-center p-10">
        <Loader2 className="h-5 w-5 animate-spin dark:text-gray-500 text-gray-400" />
        <span className="ml-2 text-sm dark:text-gray-500 text-gray-400">
          Loading metrics...
        </span>
      </div>
    );
  }

  /* ── Error / no data state ─────────────────────────────────── */
  if (!metrics) {
    return (
      <div className="glass-card flex items-center justify-center gap-2 p-10 dark:text-gray-600 text-gray-400">
        <AlertCircle className="h-5 w-5" />
        <span className="text-sm">Metrics unavailable — is the backend running?</span>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
      className="glass-card relative overflow-hidden p-4 sm:p-5 md:p-6 lg:p-7 group"
    >
      {/* Decorative gradient */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-5 bg-gradient-to-br from-emerald-500 to-teal-500 transition-opacity duration-300" />
      
      <div className="relative">
        <div className="mb-4 sm:mb-5 md:mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-emerald-500/15 text-emerald-500 flex-shrink-0">
              <BarChart3 className="h-4 w-4" />
            </div>
            <h2 className="text-xs sm:text-sm font-semibold uppercase tracking-widest dark:text-gray-300 text-gray-700">
              Model Performance
            </h2>
          </div>
          <span className="rounded-md dark:bg-white/5 bg-gray-100 px-2 sm:px-3 py-0.5 sm:py-1 text-[10px] font-mono dark:text-gray-500 text-gray-400 w-fit">
            {modelName}
          </span>
        </div>

        <div className="grid grid-cols-1 gap-2 sm:gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
        {metricItems.map((m, i) => (
          <motion.div
            key={m.label}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.6 + i * 0.08 }}
            className="rounded-xl dark:bg-white/[0.03] bg-gray-50 border dark:border-white/5 border-gray-200/50 p-4 sm:p-5 transition-all duration-200 dark:hover:bg-white/[0.05] hover:bg-gray-100 dark:hover:border-white/10 hover:border-gray-300/50 group cursor-pointer"
          >
            <div className="mb-3 flex items-center gap-2">
              <div
                className={`flex h-7 w-7 items-center justify-center rounded-lg ${m.bgColor} ${m.color}`}
              >
                {m.icon}
              </div>
              <span className="text-[11px] font-medium dark:text-gray-500 text-gray-500">
                {m.label}
              </span>
            </div>
            <p className={`text-2xl font-bold tabular-nums ${m.color}`}>
              {(m.value * 100).toFixed(1)}
              <span className="text-sm font-semibold opacity-60">%</span>
            </p>
            {/* micro-bar */}
            <div className="mt-2 h-1 w-full overflow-hidden rounded-full dark:bg-white/10 bg-gray-200">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${m.value * 100}%` }}
                transition={{ duration: 1, delay: 0.8 + i * 0.08 }}
                className={`h-full rounded-full ${m.bgColor}`}
                style={{ opacity: 0.8 }}
              />
            </div>
          </motion.div>
        ))}
      </div>
      </div>
    </motion.div>
  );
}
