import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, AlertTriangle, AlertCircle, Info } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";

const ICON_MAP = {
  error: <AlertCircle className="h-4 w-4 text-red-400" />,
  warning: <AlertTriangle className="h-4 w-4 text-amber-400" />,
  info: <Info className="h-4 w-4 text-blue-400" />,
};

const BG_MAP = {
  error: "border-red-500/20 bg-red-500/10",
  warning: "border-amber-500/20 bg-amber-500/10",
  info: "border-blue-500/20 bg-blue-500/10",
};

export default function ToastContainer() {
  const { toasts, removeToast } = useECGStore();

  return (
    <div className="pointer-events-none fixed bottom-6 right-6 z-[100] flex flex-col gap-2">
      <AnimatePresence>
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            id={toast.id}
            message={toast.message}
            type={toast.type}
            onDismiss={removeToast}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}

function ToastItem({
  id,
  message,
  type,
  onDismiss,
}: {
  id: string;
  message: string;
  type: "error" | "warning" | "info";
  onDismiss: (id: string) => void;
}) {
  useEffect(() => {
    const timer = setTimeout(() => onDismiss(id), 5000);
    return () => clearTimeout(timer);
  }, [id, onDismiss]);

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 100, scale: 0.95 }}
      transition={{ duration: 0.25 }}
      className={`pointer-events-auto flex max-w-sm items-start gap-3 rounded-xl border px-4 py-3 shadow-lg backdrop-blur-xl ${BG_MAP[type]}`}
    >
      <div className="mt-0.5 shrink-0">{ICON_MAP[type]}</div>
      <p className="flex-1 text-xs font-medium dark:text-gray-200 text-gray-800 leading-relaxed">
        {message}
      </p>
      <button
        onClick={() => onDismiss(id)}
        className="shrink-0 dark:text-gray-500 text-gray-400 hover:text-white transition-colors"
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </motion.div>
  );
}
