import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ShieldCheck, ShieldAlert, TrendingUp } from "lucide-react";
import { useECGStore } from "@/store/useECGStore";

/* ── Animated counter hook ───────────────────────────────────── */
function useAnimatedNumber(target: number, duration = 1200): number {
  const [current, setCurrent] = useState(0);
  const frameRef = useRef<number>(0);

  useEffect(() => {
    const start = performance.now();
    const from = 0;

    const tick = (now: number) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      // ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      setCurrent(from + (target - from) * eased);
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(tick);
      }
    };

    frameRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration]);

  return current;
}

/* ── Skeleton loader ─────────────────────────────────────────── */
function SkeletonCard() {
  return (
    <div className="glass-card p-6 space-y-4 animate-pulse">
      <div className="h-4 w-32 rounded-lg dark:bg-white/10 bg-gray-200" />
      <div className="h-16 rounded-xl dark:bg-white/5 bg-gray-100" />
      <div className="h-6 w-48 rounded-lg dark:bg-white/10 bg-gray-200" />
      <div className="h-3 rounded-full dark:bg-white/5 bg-gray-100" />
    </div>
  );
}

export default function PredictionCard() {
  const { prediction, loading } = useECGStore();

  const probability = prediction?.probability ?? 0;
  const isArrhythmia = prediction ? prediction.prediction === 1 : false;
  const confidence = prediction?.confidence ?? 0;

  const animatedProb = useAnimatedNumber(
    prediction ? probability * 100 : 0,
    1400,
  );
  const animatedConf = useAnimatedNumber(
    prediction ? confidence * 100 : 0,
    1200,
  );

  if (loading) return <SkeletonCard />;

  return (
    <AnimatePresence mode="wait">
      {prediction ? (
        <motion.div
          key="result"
          initial={{ opacity: 0, y: 20, scale: 0.97 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={{ opacity: 0, y: -10 }}
          transition={{ duration: 0.5, ease: "easeOut" }}
          className={`glass-card relative overflow-hidden p-4 sm:p-5 md:p-6 lg:p-7 group ${
            isArrhythmia
              ? "border-red-500/20 shadow-glow-red"
              : "border-emerald-500/20 shadow-glow-emerald"
          }`}
        >
          {/* Decorative gradient background */}
          <div
            className={`absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity duration-300 ${
              isArrhythmia ? "bg-red-500/20" : "bg-emerald-500/20"
            }`}
          />
          <div
            className={`absolute -right-20 -top-20 h-40 w-40 rounded-full blur-3xl opacity-0 group-hover:opacity-50 transition-opacity duration-500 ${
              isArrhythmia ? "bg-red-500/10" : "bg-emerald-500/10"
            }`}
          />

          {/* Pulse ring for arrhythmia */}
          {isArrhythmia && (
            <div className="absolute right-6 top-6">
              <span className="absolute inline-flex h-4 w-4 animate-pulse-ring rounded-full bg-red-400 opacity-75" />
              <span className="relative inline-flex h-4 w-4 rounded-full bg-red-500" />
            </div>
          )}

          {/* Header */}
          <div className="relative mb-3 sm:mb-4 flex items-center gap-2">
            <h2 className="text-xs sm:text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
              Prediction Result
            </h2>
          </div>

          {/* ── Main result (responsive) ─────────────────────────────────── */}
          <div className="relative flex items-center gap-3 sm:gap-4 mb-4 sm:mb-5">
            <motion.div
              initial={{ scale: 0, rotate: -20 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ type: "spring", stiffness: 200, damping: 15 }}
              className={`flex h-12 sm:h-14 w-12 sm:w-14 shrink-0 items-center justify-center rounded-xl sm:rounded-2xl ${
                isArrhythmia
                  ? "bg-red-500/20 text-red-400"
                  : "bg-emerald-500/20 text-emerald-400"
              }`}
            >
              {isArrhythmia ? (
                <ShieldAlert className="h-6 sm:h-7 w-6 sm:w-7" />
              ) : (
                <ShieldCheck className="h-6 sm:h-7 w-6 sm:w-7" />
              )}
            </motion.div>
            <div className="min-w-0">
              <motion.p
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.2 }}
                className={`text-sm sm:text-lg font-bold tracking-wide leading-tight ${
                  isArrhythmia
                    ? "text-red-400"
                    : "text-emerald-400"
                }`}
              >
                {isArrhythmia
                  ? "ARRHYTHMIA DETECTED"
                  : "NORMAL SINUS RHYTHM"}
              </motion.p>
              <p className="text-xs dark:text-gray-500 text-gray-400 mt-0.5 leading-tight">
                12-Lead IndustryCNN
              </p>
            </div>
          </div>

          {/* ── Probability (responsive) ─────────────────────────────────── */}
          <div className="relative mb-4 sm:mb-5 rounded-lg sm:rounded-xl dark:bg-white/5 bg-gray-50 p-3 sm:p-4">
            <div className="flex items-end justify-between gap-3">
              <div className="min-w-0">
                <p className="text-xs dark:text-gray-500 text-gray-400 mb-1">
                  Arrhythmia Probability
                </p>
                <p
                  className={`text-3xl sm:text-4xl font-extrabold tabular-nums ${
                    isArrhythmia ? "text-red-400" : "text-emerald-400"
                  }`}
                >
                  {animatedProb.toFixed(2)}
                  <span className="text-sm sm:text-lg font-semibold ml-0.5">%</span>
                </p>
              </div>
              <TrendingUp
                className={`h-6 sm:h-8 w-6 sm:w-8 flex-shrink-0 ${
                  isArrhythmia
                    ? "text-red-500/30"
                    : "text-emerald-500/30"
                }`}
              />
            </div>
          </div>

          {/* ── Confidence bar (responsive) ──────────────────────────────── */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <p className="text-xs dark:text-gray-500 text-gray-400">
                Confidence
              </p>
              <p className="text-xs font-semibold dark:text-gray-300 text-gray-700 tabular-nums">
                {animatedConf.toFixed(1)}%
              </p>
            </div>
            <div className="h-2 sm:h-2.5 w-full overflow-hidden rounded-full dark:bg-white/10 bg-gray-200">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${confidence * 100}%` }}
                transition={{ duration: 1.2, ease: "easeOut", delay: 0.3 }}
                className={`h-full rounded-full ${
                  isArrhythmia
                    ? "bg-gradient-to-r from-red-500 to-rose-500"
                    : "bg-gradient-to-r from-emerald-500 to-teal-500"
                }`}
              />
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          key="empty"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="glass-card flex flex-col items-center justify-center p-10 dark:text-gray-600 text-gray-400"
        >
          <ShieldCheck className="h-12 w-12 mb-3 opacity-30" />
          <p className="text-sm">
            Upload an ECG signal and run prediction
          </p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
