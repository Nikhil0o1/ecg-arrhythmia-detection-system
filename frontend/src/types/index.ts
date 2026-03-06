/* ──────────────────────────────────────────────────────────────
   Domain types mirroring the FastAPI response schemas
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
  start_sample: number;
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

/** Per-model evaluation metrics (comparison_metrics.json) */
export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
}

/** Full comparison map keyed by model name */
export type ComparisonMetrics = Record<string, ModelMetrics>;

/** Navigation tab identifier */
export type TabId =
  | "prediction"
  | "simulation"
  | "comparison"
  | "status";
