import { useCallback, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Upload,
  FileText,
  X,
  Code2,
  Zap,
  Play,
  Loader2,
} from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import { predictFile, predictJSON, simulate, extractErrorMessage } from "@/api/api";
import { readNpyFile } from "@/utils/npy-parser";
import type { ECGSignal } from "@/types/ecg";

export default function UploadCard() {
  const {
    signal,
    fileName,
    setSignal,
    clearSignal,
    setPrediction,
    setSimulation,
    loading,
    setLoading,
    addToast,
    simulationMode,
    toggleSimulationMode,
    jsonInputMode,
    toggleJsonInputMode,
    clearPrediction,
    clearSimulation,
  } = useECGStore();

  const fileRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [npyFile, setNpyFile] = useState<File | null>(null);

  /* ── File handling ─────────────────────────────────────────── */
  const handleFile = useCallback(
    async (file: File) => {
      if (!file.name.endsWith(".npy")) {
        addToast("Only .npy files are supported.", "warning");
        return;
      }
      setNpyFile(file);
      try {
        const flat = await readNpyFile(file);
        // reshape to 1000×12
        if (flat.length !== 12000) {
          addToast(
            `Expected 12 000 values (1000×12), got ${flat.length}.`,
            "warning",
          );
          return;
        }
        const matrix: ECGSignal = [];
        for (let t = 0; t < 1000; t++) {
          matrix.push(flat.slice(t * 12, t * 12 + 12));
        }
        setSignal(matrix, file.name);
        clearPrediction();
        clearSimulation();
      } catch (err) {
        addToast(extractErrorMessage(err), "error");
      }
    },
    [setSignal, clearPrediction, clearSimulation, addToast],
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const onFileChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  /* ── JSON paste handling ───────────────────────────────────── */
  const parseJsonSignal = useCallback((): ECGSignal | null => {
    try {
      const parsed = JSON.parse(jsonText) as unknown;
      if (!Array.isArray(parsed)) throw new Error("Must be an array.");
      if (parsed.length !== 1000) throw new Error("Need 1000 time-steps.");
      for (const row of parsed) {
        if (!Array.isArray(row) || row.length !== 12)
          throw new Error("Each step must have 12 leads.");
      }
      return parsed as ECGSignal;
    } catch (err) {
      addToast(
        `JSON parse error: ${err instanceof Error ? err.message : String(err)}`,
        "warning",
      );
      return null;
    }
  }, [jsonText, addToast]);

  /* ── Predict ───────────────────────────────────────────────── */
  const handlePredict = useCallback(async () => {
    setLoading(true);
    clearPrediction();
    clearSimulation();

    try {
      if (simulationMode) {
        let sig: ECGSignal;
        if (jsonInputMode) {
          const parsed = parseJsonSignal();
          if (!parsed) {
            setLoading(false);
            return;
          }
          sig = parsed;
        } else {
          if (!signal) {
            addToast("Upload a signal first.", "warning");
            setLoading(false);
            return;
          }
          sig = signal;
        }
        const res = await simulate(sig);
        setSimulation(res.timeline_predictions, res.final_prediction);
        setPrediction(res.final_prediction);
      } else if (jsonInputMode) {
        const parsed = parseJsonSignal();
        if (!parsed) {
          setLoading(false);
          return;
        }
        const res = await predictJSON(parsed);
        setPrediction(res);
        setSignal(parsed, "JSON Input");
      } else {
        if (npyFile) {
          const res = await predictFile(npyFile);
          setPrediction(res);
        } else if (signal) {
          const res = await predictJSON(signal);
          setPrediction(res);
        } else {
          addToast("Upload a signal first.", "warning");
        }
      }
    } catch (err) {
      addToast(extractErrorMessage(err), "error");
    } finally {
      setLoading(false);
    }
  }, [
    signal,
    npyFile,
    jsonInputMode,
    simulationMode,
    jsonText,
    setLoading,
    setPrediction,
    setSimulation,
    setSignal,
    clearPrediction,
    clearSimulation,
    addToast,
    parseJsonSignal,
  ]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.1 }}
      className="glass-card p-4 sm:p-5 md:p-6"
    >
      {/* ── Header (responsive) ──────────────────────────────── */}
      <div className="mb-4 sm:mb-5 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <h2 className="text-xs sm:text-sm font-semibold uppercase tracking-widest dark:text-gray-400 text-gray-500">
          ECG Input
        </h2>
        <div className="flex flex-wrap gap-1.5 sm:gap-2">
          {/* JSON toggle */}
          <button
            onClick={toggleJsonInputMode}
            className={`flex items-center gap-1 sm:gap-1.5 rounded-lg px-2.5 sm:px-3 py-1 sm:py-1.5 text-xs font-medium transition-all whitespace-nowrap ${
              jsonInputMode
                ? "bg-blue-500/20 text-blue-400 border border-blue-500/30"
                : "dark:bg-white/5 bg-gray-100 dark:text-gray-400 text-gray-600 border border-transparent dark:hover:bg-white/10 hover:bg-gray-200"
            }`}
          >
            <Code2 className="h-3 sm:h-3.5 w-3 sm:w-3.5" />
            <span className="hidden sm:inline">JSON</span>
            <span className="sm:hidden">JSON</span>
          </button>
          {/* Simulation toggle */}
          <button
            onClick={toggleSimulationMode}
            className={`flex items-center gap-1 sm:gap-1.5 rounded-lg px-2.5 sm:px-3 py-1 sm:py-1.5 text-xs font-medium transition-all whitespace-nowrap ${
              simulationMode
                ? "bg-amber-500/20 text-amber-400 border border-amber-500/30"
                : "dark:bg-white/5 bg-gray-100 dark:text-gray-400 text-gray-600 border border-transparent dark:hover:bg-white/10 hover:bg-gray-200"
            }`}
          >
            <Zap className="h-3 sm:h-3.5 w-3 sm:w-3.5" />
            <span className="hidden sm:inline">Simulate</span>
            <span className="sm:hidden">Sim</span>
          </button>
        </div>
      </div>

      {/* ── Input area ──────────────────────────────────────── */}
      <AnimatePresence mode="wait">
        {jsonInputMode ? (
          <motion.div
            key="json"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            <textarea
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              placeholder='Paste 1000×12 JSON array: [[lead1..lead12], ...]'
              rows={6}
              className="w-full rounded-lg sm:rounded-xl border dark:border-white/10 border-gray-200 dark:bg-white/5 bg-gray-50 px-3 sm:px-4 py-2 sm:py-3 font-mono text-xs dark:text-gray-300 text-gray-700 placeholder-gray-500 focus:border-emerald-500/50 focus:outline-none focus:ring-1 focus:ring-emerald-500/30 transition-colors"
            />
          </motion.div>
        ) : (
          <motion.div
            key="upload"
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
          >
            {/* Drop zone */}
            <div
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={onDrop}
              onClick={() => fileRef.current?.click()}
              className={`cursor-pointer rounded-lg sm:rounded-xl border-2 border-dashed p-6 sm:p-8 text-center transition-all ${
                dragOver
                  ? "border-emerald-500 bg-emerald-500/10"
                  : signal
                    ? "border-emerald-500/30 dark:bg-emerald-500/5 bg-emerald-50"
                    : "dark:border-white/10 border-gray-300 dark:hover:border-white/20 hover:border-gray-400 dark:hover:bg-white/[0.02] hover:bg-gray-50"
              }`}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".npy"
                onChange={onFileChange}
                className="hidden"
              />
              <AnimatePresence mode="wait">
                {signal ? (
                  <motion.div
                    key="loaded"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    className="flex flex-col items-center gap-2 sm:gap-3"
                  >
                    <div className="flex h-10 sm:h-12 w-10 sm:w-12 items-center justify-center rounded-lg sm:rounded-xl bg-emerald-500/20">
                      <FileText className="h-5 sm:h-6 w-5 sm:w-6 text-emerald-400" />
                    </div>
                    <div>
                      <p className="text-xs sm:text-sm font-medium dark:text-white text-gray-900 break-all">
                        {fileName}
                      </p>
                      <p className="mt-1 text-xs dark:text-gray-500 text-gray-400">
                        1000 × 12 leads
                      </p>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        clearSignal();
                        setNpyFile(null);
                      }}
                      className="mt-1 flex items-center gap-1 text-xs dark:text-gray-500 text-gray-400 hover:text-red-400 transition-colors"
                    >
                      <X className="h-3 w-3" />
                      Remove
                    </button>
                  </motion.div>
                ) : (
                  <motion.div
                    key="empty"
                    initial={{ scale: 0.9, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    exit={{ scale: 0.9, opacity: 0 }}
                    className="flex flex-col items-center gap-2 sm:gap-3"
                  >
                    <div className="flex h-10 sm:h-12 w-10 sm:w-12 items-center justify-center rounded-lg sm:rounded-xl dark:bg-white/5 bg-gray-100">
                      <Upload className="h-5 sm:h-6 w-5 sm:w-6 dark:text-gray-400 text-gray-500" />
                    </div>
                    <div>
                      <p className="text-xs sm:text-sm font-medium dark:text-gray-300 text-gray-700">
                        Drop ECG file here
                      </p>
                      <p className="mt-1 text-xs dark:text-gray-500 text-gray-400">
                        .npy format — 1000 × 12 leads
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Predict button (responsive) ──────────────────────────────── */}
      <motion.button
        whileHover={{ scale: 1.01 }}
        whileTap={{ scale: 0.98 }}
        onClick={handlePredict}
        disabled={loading}
        className="mt-4 sm:mt-5 flex w-full items-center justify-center gap-2 rounded-lg sm:rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 px-4 sm:px-6 py-2.5 sm:py-3.5 text-xs sm:text-sm font-semibold text-white shadow-lg shadow-emerald-600/20 transition-all hover:shadow-xl hover:shadow-emerald-600/30 disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {loading ? (
          <>
            <Loader2 className="h-3.5 sm:h-4 w-3.5 sm:w-4 animate-spin" />
            <span className="hidden sm:inline">
              {simulationMode ? "Simulating..." : "Predicting..."}
            </span>
            <span className="sm:hidden">
              {simulationMode ? "Simulating..." : "Running..."}
            </span>
          </>
        ) : (
          <>
            <Play className="h-3.5 sm:h-4 w-3.5 sm:w-4" />
            <span className="hidden sm:inline">
              {simulationMode ? "Run Simulation" : "Run Prediction"}
            </span>
            <span className="sm:hidden">
              {simulationMode ? "Simulate" : "Predict"}
            </span>
          </>
        )}
      </motion.button>
    </motion.div>
  );
}
