import { useMemo, useState, useCallback } from "react";
import { motion } from "framer-motion";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { ZoomIn, ZoomOut, Maximize2 } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";

const LEAD_NAMES = [
  "I",
  "II",
  "III",
  "aVR",
  "aVL",
  "aVF",
  "V1",
  "V2",
  "V3",
  "V4",
  "V5",
  "V6",
];

const LEAD_COLORS = [
  "#34d399",
  "#2dd4bf",
  "#22d3ee",
  "#38bdf8",
  "#60a5fa",
  "#818cf8",
  "#a78bfa",
  "#c084fc",
  "#e879f9",
  "#f472b6",
  "#fb7185",
  "#f87171",
];

interface LeadDatum {
  sample: number;
  value: number;
}

export default function WaveformViewer() {
  const { signal, prediction } = useECGStore();
  const [zoom, setZoom] = useState(1);
  const [expandedLead, setExpandedLead] = useState<number | null>(null);

  const isArrhythmia = prediction?.prediction === 1;

  /* ── Build per-lead data ─────────────────────────────────── */
  const leadsData = useMemo(() => {
    if (!signal) return null;
    const result: LeadDatum[][] = [];
    for (let lead = 0; lead < 12; lead++) {
      const data: LeadDatum[] = [];
      const step = Math.max(1, Math.floor(1 / zoom));
      for (let t = 0; t < signal.length; t += step) {
        const row = signal[t];
        if (row) {
          data.push({ sample: t, value: row[lead] ?? 0 });
        }
      }
      result.push(data);
    }
    return result;
  }, [signal, zoom]);

  const zoomIn = useCallback(() => {
    setZoom((z) => Math.min(z * 1.5, 4));
  }, []);

  const zoomOut = useCallback(() => {
    setZoom((z) => Math.max(z / 1.5, 0.5));
  }, []);

  const resetZoom = useCallback(() => {
    setZoom(1);
    setExpandedLead(null);
  }, []);

  if (!signal) return null;

  const leadsToRender =
    expandedLead !== null ? [expandedLead] : Array.from({ length: 12 }, (_, i) => i);
  const chartHeight = expandedLead !== null ? 300 : 80;

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
          12-Lead ECG Waveform
        </h2>
        <div className="flex items-center gap-1.5">
          <button
            onClick={zoomIn}
            className="rounded-lg p-1.5 dark:text-gray-400 text-gray-500 dark:hover:bg-white/10 hover:bg-gray-100 transition-colors"
            aria-label="Zoom in"
          >
            <ZoomIn className="h-4 w-4" />
          </button>
          <button
            onClick={zoomOut}
            className="rounded-lg p-1.5 dark:text-gray-400 text-gray-500 dark:hover:bg-white/10 hover:bg-gray-100 transition-colors"
            aria-label="Zoom out"
          >
            <ZoomOut className="h-4 w-4" />
          </button>
          <button
            onClick={resetZoom}
            className="rounded-lg p-1.5 dark:text-gray-400 text-gray-500 dark:hover:bg-white/10 hover:bg-gray-100 transition-colors"
            aria-label="Reset"
          >
            <Maximize2 className="h-4 w-4" />
          </button>
          <span className="ml-2 rounded-md dark:bg-white/5 bg-gray-100 px-2 py-0.5 text-[10px] font-mono dark:text-gray-500 text-gray-400">
            {zoom.toFixed(1)}×
          </span>
        </div>
      </div>

      {/* Scrollable lead stack */}
      <div
        className="space-y-1 overflow-y-auto pr-1"
        style={{ maxHeight: expandedLead !== null ? 380 : 520 }}
      >
        {leadsData &&
          leadsToRender.map((leadIdx) => {
            const data = leadsData[leadIdx];
            if (!data) return null;
            return (
              <motion.div
                key={leadIdx}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: leadIdx * 0.04 }}
                className="group relative rounded-lg dark:bg-white/[0.02] bg-gray-50/50 p-1 cursor-pointer dark:hover:bg-white/[0.04] hover:bg-gray-50 transition-colors"
                onClick={() =>
                  setExpandedLead((prev) =>
                    prev === leadIdx ? null : leadIdx,
                  )
                }
              >
                <div className="absolute left-2 top-1 z-10 flex items-center gap-1.5">
                  <span
                    className="h-1.5 w-1.5 rounded-full"
                    style={{ backgroundColor: LEAD_COLORS[leadIdx] }}
                  />
                  <span className="text-[10px] font-mono font-medium dark:text-gray-500 text-gray-400">
                    {LEAD_NAMES[leadIdx]}
                  </span>
                </div>
                <ResponsiveContainer width="100%" height={chartHeight}>
                  <LineChart
                    data={data}
                    margin={{ top: 12, right: 8, bottom: 2, left: 30 }}
                  >
                    <XAxis dataKey="sample" hide />
                    <YAxis
                      hide={expandedLead === null}
                      domain={["auto", "auto"]}
                      tick={{ fontSize: 9, fill: "#64748b" }}
                      width={30}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(15,23,42,0.95)",
                        border: "1px solid rgba(255,255,255,0.1)",
                        borderRadius: "8px",
                        fontSize: "11px",
                        color: "#e2e8f0",
                        padding: "6px 10px",
                      }}
                      labelFormatter={(v) => `Sample ${v}`}
                      formatter={(v: number) => [v.toFixed(4), LEAD_NAMES[leadIdx]]}
                    />
                    <Line
                      type="monotone"
                      dataKey="value"
                      stroke={LEAD_COLORS[leadIdx]}
                      strokeWidth={expandedLead !== null ? 1.5 : 1}
                      dot={false}
                      isAnimationActive
                      animationDuration={1200}
                      animationEasing="ease-out"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>
            );
          })}
      </div>

      {/* Status bar */}
      <div className="mt-3 flex items-center justify-between text-[10px] dark:text-gray-600 text-gray-400">
        <span>1000 samples × 12 leads</span>
        {prediction && (
          <span
            className={
              isArrhythmia
                ? "text-red-400/70"
                : "text-emerald-400/70"
            }
          >
            {isArrhythmia ? "Abnormal pattern" : "Normal sinus rhythm"}
          </span>
        )}
      </div>
    </motion.div>
  );
}
