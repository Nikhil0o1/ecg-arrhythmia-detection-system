import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  CartesianGrid,
} from "recharts";
import { Timer } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";

interface ChartDatum {
  chunk: number;
  endSample: number;
  probability: number;
  prediction: number;
}

export default function SimulationChart() {
  const { simulationTimeline, simulationFinal } = useECGStore();

  const data = useMemo<ChartDatum[]>(() => {
    return simulationTimeline.map((t) => ({
      chunk: t.chunk_index,
      endSample: t.end_sample,
      probability: +(t.probability * 100).toFixed(2),
      prediction: t.prediction,
    }));
  }, [simulationTimeline]);

  if (data.length === 0) return null;

  const maxProb = Math.max(...data.map((d) => d.probability));
  const isArrhythmia = simulationFinal?.prediction === 1;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="glass-card p-6"
    >
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Timer className="h-4 w-4 dark:text-gray-500 text-gray-400" />
          <h2 className="text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
            Simulation Timeline
          </h2>
        </div>
        <span className="text-xs dark:text-gray-600 text-gray-400">
          {data.length} chunks analysed
        </span>
      </div>

      {/* Chart */}
      <div className="rounded-xl dark:bg-white/[0.02] bg-gray-50/50 p-3">
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={data} margin={{ top: 10, right: 10, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="probGrad" x1="0" y1="0" x2="0" y2="1">
                <stop
                  offset="5%"
                  stopColor={isArrhythmia ? "#ef4444" : "#10b981"}
                  stopOpacity={0.3}
                />
                <stop
                  offset="95%"
                  stopColor={isArrhythmia ? "#ef4444" : "#10b981"}
                  stopOpacity={0}
                />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
              vertical={false}
            />
            <XAxis
              dataKey="endSample"
              tick={{ fontSize: 10, fill: "#64748b" }}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickLine={false}
              label={{
                value: "Samples",
                position: "insideBottomRight",
                offset: -5,
                style: { fontSize: 10, fill: "#475569" },
              }}
            />
            <YAxis
              domain={[0, Math.max(100, maxProb + 10)]}
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
              formatter={(v: number) => [`${v.toFixed(2)}%`, "Probability"]}
              labelFormatter={(v) => `Sample ${v}`}
            />
            <ReferenceLine
              y={50}
              stroke="rgba(255,255,255,0.15)"
              strokeDasharray="4 4"
              label={{
                value: "Threshold",
                position: "insideTopRight",
                style: { fontSize: 9, fill: "#64748b" },
              }}
            />
            <Area
              type="monotone"
              dataKey="probability"
              stroke={isArrhythmia ? "#ef4444" : "#10b981"}
              strokeWidth={2}
              fill="url(#probGrad)"
              isAnimationActive
              animationDuration={1500}
              animationEasing="ease-out"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary */}
      {simulationFinal && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.8 }}
          className="mt-4 flex items-center justify-between rounded-lg dark:bg-white/5 bg-gray-100 px-4 py-2.5 text-xs"
        >
          <span className="dark:text-gray-400 text-gray-500">Final Prediction</span>
          <span
            className={`font-semibold ${
              isArrhythmia ? "text-red-400" : "text-emerald-400"
            }`}
          >
            {isArrhythmia ? "Arrhythmia" : "Normal"}{" "}
            ({(simulationFinal.probability * 100).toFixed(2)}%)
          </span>
        </motion.div>
      )}
    </motion.div>
  );
}
