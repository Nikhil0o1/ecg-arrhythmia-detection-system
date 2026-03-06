import { type ReactNode } from "react";

interface MetricCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  subtext?: string;
  variant?: "default" | "success" | "danger" | "info";
}

const config: Record<string, { bg: string; border: string; iconBg: string; accent: string }> = {
  default: {
    bg: "bg-white",
    border: "border-gray-100",
    iconBg: "bg-gray-100 text-gray-500",
    accent: "text-gray-900",
  },
  success: {
    bg: "bg-gradient-to-br from-emerald-50 to-teal-50",
    border: "border-emerald-100",
    iconBg: "bg-emerald-100 text-emerald-600",
    accent: "text-emerald-900",
  },
  danger: {
    bg: "bg-gradient-to-br from-red-50 to-rose-50",
    border: "border-red-100",
    iconBg: "bg-red-100 text-red-600",
    accent: "text-red-900",
  },
  info: {
    bg: "bg-gradient-to-br from-brand-50 to-indigo-50",
    border: "border-brand-100",
    iconBg: "bg-brand-100 text-brand-600",
    accent: "text-brand-900",
  },
};

export default function MetricCard({
  label,
  value,
  icon,
  subtext,
  variant = "default",
}: MetricCardProps) {
  const c = config[variant] ?? config.default!;
  return (
    <div
      className={`group relative overflow-hidden rounded-lg sm:rounded-2xl border p-3 sm:p-4 md:p-6 transition-all duration-300 hover:shadow-card-hover ${c.bg} ${c.border}`}
    >
      {/* Decorative shimmer on hover */}
      <div className="pointer-events-none absolute -inset-full top-0 z-0 h-full w-1/2 -skew-x-12 bg-gradient-to-r from-transparent via-white/30 to-transparent opacity-0 transition-opacity duration-500 group-hover:animate-shimmer group-hover:opacity-100" />

      <div className="relative z-10">
        <div className="flex items-center justify-between gap-1">
          <p className="text-[10px] sm:text-[11px] font-bold uppercase tracking-[0.12em] text-gray-400">
            {label}
          </p>
          {icon && (
            <span className={`flex h-7 sm:h-9 w-7 sm:w-9 items-center justify-center rounded-lg sm:rounded-xl flex-shrink-0 ${c.iconBg}`}>
              {icon}
            </span>
          )}
        </div>
        <p className={`mt-2 sm:mt-3 text-2xl sm:text-3xl font-extrabold tracking-tight ${c.accent}`}>
          {value}
        </p>
        {subtext && (
          <p className="mt-1 text-xs font-medium text-gray-400">{subtext}</p>
        )}
      </div>
    </div>
  );
}
