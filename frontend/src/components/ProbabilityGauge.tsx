import { motion } from "framer-motion";
import { useECGStore } from "@/store/useECGStore";

/** Animated semicircular probability gauge */
export default function ProbabilityGauge() {
  const { prediction } = useECGStore();

  if (!prediction) return null;

  const probability = prediction.probability ?? 0;
  const pct = Math.round(probability * 100);
  const isAbnormal = prediction.prediction === 1;

  /* ── SVG arc parameters ─────────────────────────────────── */
  const size = 220;
  const cx = size / 2;
  const cy = size / 2 + 10;
  const r = 85;
  // Arc goes from 180° (left) to 0° (right) = semicircle
  const startAngle = Math.PI; // 180°
  const endAngle = 0; // 0°
  const totalAngle = Math.PI; // 180°

  const arcLength = r * totalAngle;
  const filledLength = arcLength * probability;

  // Helper to get point on arc
  const arcPoint = (angle: number) => ({
    x: cx + r * Math.cos(angle),
    y: cy - r * Math.sin(angle),
  });

  // Background arc path (full semicircle)
  const startPt = arcPoint(startAngle);
  const endPt = arcPoint(endAngle);
  const bgArc = `M ${startPt.x} ${startPt.y} A ${r} ${r} 0 0 1 ${endPt.x} ${endPt.y}`;

  // Color based on prediction
  const gaugeColor = isAbnormal ? "#ef4444" : "#10b981";

  // Tick marks
  const ticks = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5, delay: 0.15 }}
      className="glass-card flex flex-col items-center p-6"
    >
      <h2 className="mb-2 self-start text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
        Probability Gauge
      </h2>

      <div className="relative" style={{ width: size, height: size / 2 + 40 }}>
        <svg
          width={size}
          height={size / 2 + 40}
          viewBox={`0 0 ${size} ${size / 2 + 40}`}
        >
          {/* Glow filter */}
          <defs>
            <filter id="gauge-glow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <linearGradient id="gauge-grad" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#10b981" />
              <stop offset="50%" stopColor="#f59e0b" />
              <stop offset="100%" stopColor="#ef4444" />
            </linearGradient>
          </defs>

          {/* Background track */}
          <path
            d={bgArc}
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth={14}
            strokeLinecap="round"
          />

          {/* Filled arc */}
          <motion.path
            d={bgArc}
            fill="none"
            stroke={gaugeColor}
            strokeWidth={14}
            strokeLinecap="round"
            strokeDasharray={arcLength}
            filter="url(#gauge-glow)"
            initial={{ strokeDashoffset: arcLength }}
            animate={{ strokeDashoffset: arcLength - filledLength }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.3 }}
          />

          {/* Tick marks */}
          {ticks.map((t) => {
            const angle = startAngle - totalAngle * t;
            const inner = {
              x: cx + (r - 12) * Math.cos(angle),
              y: cy - (r - 12) * Math.sin(angle),
            };
            const outer = {
              x: cx + (r + 12) * Math.cos(angle),
              y: cy - (r + 12) * Math.sin(angle),
            };
            const label = {
              x: cx + (r + 24) * Math.cos(angle),
              y: cy - (r + 24) * Math.sin(angle),
            };
            return (
              <g key={t}>
                <line
                  x1={inner.x}
                  y1={inner.y}
                  x2={outer.x}
                  y2={outer.y}
                  stroke="rgba(255,255,255,0.15)"
                  strokeWidth={1.5}
                />
                <text
                  x={label.x}
                  y={label.y}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  className="fill-gray-500 dark:fill-gray-600"
                  fontSize={9}
                >
                  {Math.round(t * 100)}
                </text>
              </g>
            );
          })}

          {/* Needle */}
          <motion.g
            initial={{ rotate: 0 }}
            animate={{ rotate: probability * 180 }}
            transition={{ duration: 1.5, ease: "easeOut", delay: 0.3 }}
            style={{ originX: `${cx}px`, originY: `${cy}px` }}
          >
            <line
              x1={cx}
              y1={cy}
              x2={cx - r + 10}
              y2={cy}
              stroke={gaugeColor}
              strokeWidth={2.5}
              strokeLinecap="round"
              opacity={0.8}
            />
          </motion.g>

          {/* Center circle */}
          <circle
            cx={cx}
            cy={cy}
            r={6}
            fill={gaugeColor}
            opacity={0.9}
          />
        </svg>

        {/* Center label */}
        <div className="absolute inset-x-0 flex flex-col items-center" style={{ bottom: 4 }}>
          <motion.span
            key={pct}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-3xl font-bold tabular-nums"
            style={{ color: gaugeColor }}
          >
            {pct}%
          </motion.span>
          <span className="text-[10px] font-medium dark:text-gray-500 text-gray-400">
            {isAbnormal ? "Abnormal" : "Normal"} probability
          </span>
        </div>
      </div>

      {/* Footer stats */}
      <div className="mt-2 flex w-full justify-between px-2 text-[10px] dark:text-gray-600 text-gray-400">
        <span>Model: IndustryCNN</span>
        <span>Confidence: {pct >= 70 ? "High" : pct >= 40 ? "Medium" : "Low"}</span>
      </div>
    </motion.div>
  );
}
