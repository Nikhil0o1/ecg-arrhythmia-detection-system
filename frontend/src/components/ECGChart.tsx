import {
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  ComposedChart,
} from "recharts";
import { Activity } from "lucide-react";

interface ECGChartProps {
  signal: number[];
  sampleRate?: number;
  title?: string;
  height?: number;
}

export default function ECGChart({
  signal,
  sampleRate = 100,
  title,
  height = 300,
}: ECGChartProps) {
  const maxPoints = 600;
  const step = Math.max(1, Math.floor(signal.length / maxPoints));

  const data = signal
    .filter((_, i) => i % step === 0)
    .map((v, i) => ({
      time: Number(((i * step) / sampleRate).toFixed(3)),
      amplitude: Number(v.toFixed(4)),
    }));

  return (
    <div className="group overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card transition-all duration-300 hover:shadow-card-hover">
      {title && (
        <div className="flex items-center gap-2.5 border-b border-gray-100 px-6 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-100">
            <Activity className="h-4 w-4 text-brand-600" />
          </div>
          <h3 className="text-sm font-bold text-gray-800">{title}</h3>
          <span className="ml-auto rounded-full bg-gray-100 px-3 py-1 text-[10px] font-semibold text-gray-500">
            {signal.length.toLocaleString()} samples
          </span>
        </div>
      )}
      <div className="p-4 pt-2">
        <ResponsiveContainer width="100%" height={height}>
          <ComposedChart data={data}>
            <defs>
              <linearGradient id="ecgGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366f1" stopOpacity={0.15} />
                <stop offset="95%" stopColor="#6366f1" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis
              dataKey="time"
              label={{ value: "Time (s)", position: "insideBottomRight", offset: -5, style: { fontSize: 11, fill: "#94a3b8" } }}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#e2e8f0"
              tickLine={false}
            />
            <YAxis
              label={{
                value: "Amplitude",
                angle: -90,
                position: "insideLeft",
                offset: 10,
                style: { fontSize: 11, fill: "#94a3b8" },
              }}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              stroke="#e2e8f0"
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                fontSize: 12,
                borderRadius: 12,
                border: "none",
                boxShadow: "0 4px 20px rgba(0,0,0,0.08)",
                padding: "8px 14px",
              }}
              formatter={(v: number) => [v.toFixed(4), "Amplitude"]}
              labelFormatter={(l) => `${l} s`}
            />
            <Area
              type="monotone"
              dataKey="amplitude"
              stroke="none"
              fill="url(#ecgGradient)"
              animationDuration={1200}
            />
            <Line
              type="monotone"
              dataKey="amplitude"
              stroke="#6366f1"
              strokeWidth={1.8}
              dot={false}
              animationDuration={1200}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
