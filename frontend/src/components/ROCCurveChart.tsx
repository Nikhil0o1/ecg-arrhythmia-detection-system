import { useEffect, useMemo } from "react";
import { motion } from "framer-motion";
import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
  Area,
  AreaChart,
} from "recharts";
import { TrendingUp, Loader2 } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import { fetchROCCurve, extractErrorMessage } from "@/api/api";

interface ROCDatum {
  fpr: number;
  tpr: number;
}

export default function ROCCurveChart() {
  const { rocCurve, setROCCurve, addToast } = useECGStore();

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const data = await fetchROCCurve();
        if (mounted) setROCCurve(data);
      } catch (err) {
        if (mounted) addToast(extractErrorMessage(err), "warning");
      }
    };
    load();
    return () => {
      mounted = false;
    };
  }, [setROCCurve, addToast]);

  const chartData = useMemo<ROCDatum[]>(() => {
    if (!rocCurve) return [];
    return rocCurve.fpr.map((fpr, i) => ({
      fpr: +fpr.toFixed(4),
      tpr: +(rocCurve.tpr[i] ?? 0).toFixed(4),
    }));
  }, [rocCurve]);

  if (!rocCurve) {
    return (
      <div className="glass-card flex items-center justify-center p-10">
        <Loader2 className="h-5 w-5 animate-spin dark:text-gray-500 text-gray-400" />
        <span className="ml-2 text-sm dark:text-gray-500 text-gray-400">
          Loading ROC curve...
        </span>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="h-4 w-4 dark:text-gray-500 text-gray-400" />
          <h2 className="text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
            ROC Curve
          </h2>
        </div>
        <div className="flex items-center gap-3">
          <span className="rounded-md dark:bg-emerald-500/10 bg-emerald-50 border dark:border-emerald-500/20 border-emerald-200 px-2.5 py-1 text-xs font-semibold text-emerald-500">
            AUC = {(rocCurve.auc * 100).toFixed(2)}%
          </span>
          <span className="text-[10px] dark:text-gray-600 text-gray-400">
            n = {rocCurve.n_samples.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="rounded-xl dark:bg-white/[0.02] bg-gray-50/50 p-3">
        <ResponsiveContainer width="100%" height={280}>
          <AreaChart data={chartData} margin={{ top: 10, right: 10, bottom: 5, left: 5 }}>
            <defs>
              <linearGradient id="rocGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#10b981" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
            />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#64748b" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
              label={{
                value: "False Positive Rate",
                position: "insideBottomRight",
                offset: -5,
                style: { fontSize: 10, fill: "#475569" },
              }}
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              tick={{ fontSize: 10, fill: "#64748b" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
              label={{
                value: "True Positive Rate",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                style: { fontSize: 10, fill: "#475569" },
              }}
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
              formatter={(v: number) => [v.toFixed(4)]}
              labelFormatter={(v) => `FPR: ${Number(v).toFixed(4)}`}
            />
            {/* Diagonal reference (random classifier) */}
            <ReferenceLine
              segment={[
                { x: 0, y: 0 },
                { x: 1, y: 1 },
              ]}
              stroke="rgba(255,255,255,0.15)"
              strokeDasharray="6 4"
            />
            <Area
              type="monotone"
              dataKey="tpr"
              stroke="#10b981"
              strokeWidth={2}
              fill="url(#rocGrad)"
              isAnimationActive
              animationDuration={1800}
              animationEasing="ease-out"
              dot={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Footer */}
      <div className="mt-3 flex items-center justify-between text-[10px] dark:text-gray-600 text-gray-400">
        <span>{rocCurve.model_name} — Test Set Evaluation</span>
        <span>{chartData.length} data points</span>
      </div>
    </motion.div>
  );
}
