interface PredictionBadgeProps {
  prediction: number; // 0 = Normal, 1 = Arrhythmia
  size?: "sm" | "md" | "lg";
}

const sizeClasses = {
  sm: "px-3 py-1 text-[11px] gap-1.5",
  md: "px-4 py-1.5 text-xs gap-2",
  lg: "px-5 py-2.5 text-sm gap-2.5",
};

export default function PredictionBadge({
  prediction,
  size = "md",
}: PredictionBadgeProps) {
  const isNormal = prediction === 0;

  return (
    <span
      className={`inline-flex items-center rounded-full font-bold tracking-wide shadow-sm transition-all duration-200 ${
        sizeClasses[size]
      } ${
        isNormal
          ? "bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-800 shadow-emerald-100"
          : "bg-gradient-to-r from-red-100 to-rose-100 text-red-800 shadow-red-100"
      }`}
    >
      <span className="relative flex h-2 w-2">
        <span
          className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-50 ${
            isNormal ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
        <span
          className={`relative inline-flex h-2 w-2 rounded-full ${
            isNormal ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
      </span>
      {isNormal ? "NORMAL" : "ARRHYTHMIA"}
    </span>
  );
}
