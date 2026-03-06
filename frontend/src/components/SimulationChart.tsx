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
      className="glass-card relative overflow-hidden p-4 sm:p-5 md:p-6 lg:p-7 group"
    >
      {/* Decorative gradient */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-5 bg-gradient-to-br from-emerald-500 to-teal-500 transition-opacity duration-300" />
      
      <div className="relative">
        {/* Header (responsive) */}
        <div className="mb-4 sm:mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-amber-500/15 text-amber-500 flex-shrink-0">
              <Timer className="h-4 w-4" />
            </div>
            <h2 className="text-xs sm:text-sm font-semibold uppercase tracking-widest dark:text-gray-300 text-gray-700">
              Simulation Timeline
            </h2>
          </div>
          <span className="rounded-md dark:bg-white/5 bg-gray-100 px-2 sm:px-3 py-0.5 sm:py-1 text-[10px] font-mono dark:text-gray-500 text-gray-400 w-fit">
            {data.length} chunks
          </span>
        </div>

        {/* Chart */}
        <div className="rounded-lg sm:rounded-xl dark:bg-white/[0.02] bg-gray-50/50 p-2 sm:p-4">
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
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="mt-4 flex items-center justify-between rounded-lg dark:bg-white/5 bg-gray-100 px-4 py-3 text-xs border dark:border-white/5 border-gray-200/50"
        >
          <span className="dark:text-gray-400 text-gray-500 font-medium">Final Prediction</span>
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
      </div>
    </motion.div>
  );
}
