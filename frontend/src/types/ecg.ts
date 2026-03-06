/* ──────────────────────────────────────────────────────────────
   Domain types — strict, no `any`
   ────────────────────────────────────────────────────────────── */

/** POST /predict & POST /predict-file response */
export interface PredictResponse {
  probability: number;
  prediction: number; // 0 = Normal, 1 = Arrhythmia
  confidence: number;
}

/** GET /health response */
export interface HealthResponse {
  status: string;
  device: string;
}

/** A single entry in the simulation timeline */
export interface TimelineEntry {
  chunk_index: number;
  end_sample: number;
  probability: number;
  prediction: number;
  confidence: number;
}

/** POST /simulate response */
export interface SimulateResponse {
  timeline_predictions: TimelineEntry[];
  final_prediction: PredictResponse;
}

/** Per-model evaluation metrics */
export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
  sensitivity?: number;
  specificity?: number;
}

/** GET /metrics response */
export interface AllMetricsResponse {
  primary: ModelMetrics;
  primary_model_name: string;
  comparison: Record<string, ModelMetrics>;
}

/** GET /roc-curve response */
export interface ROCCurveResponse {
  fpr: number[];
  tpr: number[];
  auc: number;
  model_name: string;
  n_samples: number;
}

/** GET /confusion-matrix response */
export interface ConfusionMatrixResponse {
  matrix: number[][];
  labels: string[];
  tn: number;
  fp: number;
  fn: number;
  tp: number;
  total: number;
  model_name: string;
}

/** GET /pr-curve response */
export interface PRCurveResponse {
  precision: number[];
  recall: number[];
  auc: number;
  model_name: string;
}

/** 12-lead ECG signal — 1000 time-steps × 12 leads */
export type ECGSignal = number[][];

/** Theme mode */
export type ThemeMode = "dark" | "light";

/** Error toast entry */
export interface ToastEntry {
  id: string;
  message: string;
  type: "error" | "warning" | "info";
}
