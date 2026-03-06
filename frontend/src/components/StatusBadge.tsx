interface StatusBadgeProps {
  online: boolean;
  label?: string;
}

export default function StatusBadge({
  online,
  label,
}: StatusBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-2.5 rounded-full px-5 py-2 text-sm font-bold tracking-wide shadow-sm transition-all duration-300 ${
        online
          ? "bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-800 shadow-emerald-200/50"
          : "bg-gradient-to-r from-red-100 to-rose-100 text-red-800 shadow-red-200/50"
      }`}
    >
      <span className="relative flex h-2.5 w-2.5">
        {online && (
          <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
        )}
        <span
          className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
            online ? "bg-emerald-500" : "bg-red-500"
          }`}
        />
      </span>
      {label ?? (online ? "Online" : "Offline")}
    </span>
  );
}
