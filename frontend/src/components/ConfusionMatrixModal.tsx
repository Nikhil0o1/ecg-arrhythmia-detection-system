import { useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Grid3X3 } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import { fetchConfusionMatrix, extractErrorMessage } from "@/api/api";

export default function ConfusionMatrixModal() {
  const {
    confusionMatrix,
    setConfusionMatrix,
    cmModalOpen,
    setCmModalOpen,
    addToast,
  } = useECGStore();

  useEffect(() => {
    let mounted = true;
    const load = async () => {
      try {
        const data = await fetchConfusionMatrix();
        if (mounted) setConfusionMatrix(data);
      } catch (err) {
        if (mounted) addToast(extractErrorMessage(err), "warning");
      }
    };
    if (!confusionMatrix) load();
    return () => {
      mounted = false;
    };
  }, [confusionMatrix, setConfusionMatrix, addToast]);

  const cm = confusionMatrix;

  return (
    <>
      {/* Trigger button (responsive) */}
      <motion.button
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        onClick={() => setCmModalOpen(true)}
        disabled={!cm}
        className="glass-card flex flex-col gap-2 sm:gap-0 sm:flex-row sm:items-center sm:gap-4 px-3 sm:px-6 py-4 sm:py-5 md:py-6 cursor-pointer w-full disabled:opacity-50 group hover:shadow-lg transition-all duration-300"
      >
        <div className="flex h-9 sm:h-10 w-9 sm:w-10 items-center justify-center rounded-lg sm:rounded-xl bg-violet-500/15 text-violet-400 flex-shrink-0">
          <Grid3X3 className="h-4 sm:h-5 w-4 sm:w-5" />
        </div>
        <div className="flex-1 text-left">
          <p className="text-xs sm:text-sm font-semibold dark:text-gray-200 text-gray-800">
            Confusion Matrix
          </p>
          <p className="text-xs dark:text-gray-500 text-gray-400 mt-0.5">
            {cm
              ? `${cm.total.toLocaleString()} samples`
              : "Loading..."}
          </p>
        </div>
      </motion.button>

      {/* Modal overlay */}
      <AnimatePresence>
        {cmModalOpen && cm && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setCmModalOpen(false)}
            className="fixed inset-0 z-[90] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.9, opacity: 0, y: 20 }}
              transition={{ type: "spring", stiffness: 300, damping: 25 }}
              onClick={(e) => e.stopPropagation()}
              className="relative w-full max-w-lg rounded-xl sm:rounded-2xl dark:bg-clinical-card bg-white border dark:border-white/10 border-gray-200 shadow-2xl p-4 sm:p-5 md:p-6"
            >
              {/* Close button */}
              <button
                onClick={() => setCmModalOpen(false)}
                className="absolute right-3 sm:right-4 top-3 sm:top-4 rounded-lg p-1 sm:p-1.5 dark:text-gray-500 text-gray-400 hover:text-white dark:hover:bg-white/10 hover:bg-gray-100 transition-colors"
              >
                <X className="h-4 w-4" />
              </button>

              {/* Header (responsive) */}
              <div className="mb-4 sm:mb-5 md:mb-6 pr-8">
                <h3 className="text-base sm:text-lg font-bold dark:text-white text-gray-900">
                  Confusion Matrix
                </h3>
                <p className="text-[10px] sm:text-xs dark:text-gray-500 text-gray-400 mt-1">
                  {cm.model_name} — {cm.total.toLocaleString()} samples
                </p>
              </div>

              {/* Matrix grid */}
              <div className="flex flex-col items-center">
                {/* Column headers */}
                <div className="mb-2 ml-24 flex gap-2">
                  <span className="w-28 text-center text-[10px] font-semibold uppercase tracking-wider dark:text-gray-500 text-gray-400">
                    Pred Normal
                  </span>
                  <span className="w-28 text-center text-[10px] font-semibold uppercase tracking-wider dark:text-gray-500 text-gray-400">
                    Pred Arrhythmia
                  </span>
                </div>

                {/* Row 0: True Normal */}
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-20 text-right text-[10px] font-semibold uppercase tracking-wider dark:text-gray-500 text-gray-400">
                    True Normal
                  </span>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.1 }}
                    className="flex h-28 w-28 flex-col items-center justify-center rounded-xl bg-emerald-500/15 border border-emerald-500/20"
                  >
                    <span className="text-3xl font-bold text-emerald-400 tabular-nums">
                      {cm.tn.toLocaleString()}
                    </span>
                    <span className="text-[9px] text-emerald-500/70 mt-1">
                      TN ({((cm.tn / cm.total) * 100).toFixed(1)}%)
                    </span>
                  </motion.div>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.2 }}
                    className="flex h-28 w-28 flex-col items-center justify-center rounded-xl bg-red-500/10 border border-red-500/15"
                  >
                    <span className="text-3xl font-bold text-red-400/80 tabular-nums">
                      {cm.fp.toLocaleString()}
                    </span>
                    <span className="text-[9px] text-red-500/50 mt-1">
                      FP ({((cm.fp / cm.total) * 100).toFixed(1)}%)
                    </span>
                  </motion.div>
                </div>

                {/* Row 1: True Arrhythmia */}
                <div className="flex items-center gap-2">
                  <span className="w-20 text-right text-[10px] font-semibold uppercase tracking-wider dark:text-gray-500 text-gray-400">
                    True Arrhythmia
                  </span>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3 }}
                    className="flex h-28 w-28 flex-col items-center justify-center rounded-xl bg-amber-500/10 border border-amber-500/15"
                  >
                    <span className="text-3xl font-bold text-amber-400/80 tabular-nums">
                      {cm.fn.toLocaleString()}
                    </span>
                    <span className="text-[9px] text-amber-500/50 mt-1">
                      FN ({((cm.fn / cm.total) * 100).toFixed(1)}%)
                    </span>
                  </motion.div>
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.4 }}
                    className="flex h-28 w-28 flex-col items-center justify-center rounded-xl bg-emerald-500/15 border border-emerald-500/20"
                  >
                    <span className="text-3xl font-bold text-emerald-400 tabular-nums">
                      {cm.tp.toLocaleString()}
                    </span>
                    <span className="text-[9px] text-emerald-500/70 mt-1">
                      TP ({((cm.tp / cm.total) * 100).toFixed(1)}%)
                    </span>
                  </motion.div>
                </div>
              </div>

              {/* Summary stats */}
              <div className="mt-6 grid grid-cols-4 gap-2">
                {[
                  {
                    label: "Accuracy",
                    value: ((cm.tp + cm.tn) / cm.total) * 100,
                    color: "text-emerald-400",
                  },
                  {
                    label: "Precision",
                    value:
                      cm.tp + cm.fp > 0
                        ? (cm.tp / (cm.tp + cm.fp)) * 100
                        : 0,
                    color: "text-blue-400",
                  },
                  {
                    label: "Recall",
                    value:
                      cm.tp + cm.fn > 0
                        ? (cm.tp / (cm.tp + cm.fn)) * 100
                        : 0,
                    color: "text-teal-400",
                  },
                  {
                    label: "Specificity",
                    value:
                      cm.tn + cm.fp > 0
                        ? (cm.tn / (cm.tn + cm.fp)) * 100
                        : 0,
                    color: "text-cyan-400",
                  },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    className="rounded-lg dark:bg-white/5 bg-gray-50 p-3 text-center"
                  >
                    <p className="text-[10px] dark:text-gray-500 text-gray-400">
                      {stat.label}
                    </p>
                    <p
                      className={`text-lg font-bold tabular-nums ${stat.color}`}
                    >
                      {stat.value.toFixed(1)}%
                    </p>
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
