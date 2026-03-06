import { create } from "zustand";
import type {
  PredictResponse,
  TimelineEntry,
  ECGSignal,
  ThemeMode,
  ToastEntry,
  AllMetricsResponse,
  ROCCurveResponse,
  ConfusionMatrixResponse,
  PRCurveResponse,
} from "@/types/ecg";

interface ECGState {
  /* ── Signal ────────────────────────────────────────── */
  signal: ECGSignal | null;
  fileName: string | null;
  setSignal: (signal: ECGSignal, fileName?: string) => void;
  clearSignal: () => void;

  /* ── Prediction ────────────────────────────────────── */
  prediction: PredictResponse | null;
  setPrediction: (p: PredictResponse) => void;
  clearPrediction: () => void;

  /* ── Loading ───────────────────────────────────────── */
  loading: boolean;
  setLoading: (v: boolean) => void;

  /* ── Simulation ────────────────────────────────────── */
  simulationTimeline: TimelineEntry[];
  simulationFinal: PredictResponse | null;
  setSimulation: (
    timeline: TimelineEntry[],
    final_pred: PredictResponse,
  ) => void;
  clearSimulation: () => void;
  simulationMode: boolean;
  toggleSimulationMode: () => void;

  /* ── Health ────────────────────────────────────────── */
  connected: boolean;
  device: string;
  setHealth: (connected: boolean, device: string) => void;

  /* ── Theme ─────────────────────────────────────────── */
  theme: ThemeMode;
  toggleTheme: () => void;

  /* ── Toasts ────────────────────────────────────────── */
  toasts: ToastEntry[];
  addToast: (message: string, type?: ToastEntry["type"]) => void;
  removeToast: (id: string) => void;

  /* ── JSON input mode ───────────────────────────────── */
  jsonInputMode: boolean;
  toggleJsonInputMode: () => void;

  /* ── Metrics ───────────────────────────────────────── */
  metrics: AllMetricsResponse | null;
  metricsLoading: boolean;
  setMetrics: (m: AllMetricsResponse) => void;
  setMetricsLoading: (v: boolean) => void;

  /* ── ROC Curve ─────────────────────────────────────── */
  rocCurve: ROCCurveResponse | null;
  setROCCurve: (r: ROCCurveResponse) => void;

  /* ── Confusion Matrix ──────────────────────────────── */
  confusionMatrix: ConfusionMatrixResponse | null;
  setConfusionMatrix: (cm: ConfusionMatrixResponse) => void;
  cmModalOpen: boolean;
  setCmModalOpen: (v: boolean) => void;

  /* ── PR Curve ──────────────────────────────────────── */
  prCurve: PRCurveResponse | null;
  setPRCurve: (pr: PRCurveResponse) => void;
}

let toastCounter = 0;

export const useECGStore = create<ECGState>((set) => ({
  signal: null,
  fileName: null,
  setSignal: (signal, fileName) =>
    set({ signal, fileName: fileName ?? null }),
  clearSignal: () =>
    set({
      signal: null,
      fileName: null,
      prediction: null,
      simulationTimeline: [],
      simulationFinal: null,
    }),

  prediction: null,
  setPrediction: (p) => set({ prediction: p }),
  clearPrediction: () => set({ prediction: null }),

  loading: false,
  setLoading: (v) => set({ loading: v }),

  simulationTimeline: [],
  simulationFinal: null,
  setSimulation: (timeline, final_pred) =>
    set({ simulationTimeline: timeline, simulationFinal: final_pred }),
  clearSimulation: () =>
    set({ simulationTimeline: [], simulationFinal: null }),
  simulationMode: false,
  toggleSimulationMode: () =>
    set((s) => ({ simulationMode: !s.simulationMode })),

  connected: false,
  device: "unknown",
  setHealth: (connected, device) => set({ connected, device }),

  theme: "dark",
  toggleTheme: () =>
    set((s) => {
      const next = s.theme === "dark" ? "light" : "dark";
      if (next === "dark") {
        document.documentElement.classList.add("dark");
      } else {
        document.documentElement.classList.remove("dark");
      }
      return { theme: next };
    }),

  toasts: [],
  addToast: (message, type = "error") =>
    set((s) => ({
      toasts: [
        ...s.toasts,
        { id: `toast-${++toastCounter}`, message, type },
      ],
    })),
  removeToast: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  jsonInputMode: false,
  toggleJsonInputMode: () =>
    set((s) => ({ jsonInputMode: !s.jsonInputMode })),

  metrics: null,
  metricsLoading: false,
  setMetrics: (m) => set({ metrics: m }),
  setMetricsLoading: (v) => set({ metricsLoading: v }),

  rocCurve: null,
  setROCCurve: (r) => set({ rocCurve: r }),

  confusionMatrix: null,
  setConfusionMatrix: (cm) => set({ confusionMatrix: cm }),
  cmModalOpen: false,
  setCmModalOpen: (v) => set({ cmModalOpen: v }),

  prCurve: null,
  setPRCurve: (pr) => set({ prCurve: pr }),
}));
