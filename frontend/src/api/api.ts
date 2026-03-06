import axios, { AxiosError } from "axios";
import type {
  HealthResponse,
  PredictResponse,
  SimulateResponse,
  AllMetricsResponse,
  ROCCurveResponse,
  ConfusionMatrixResponse,
  PRCurveResponse,
  ECGSignal,
} from "@/types/ecg";

/* ──────────────────────────────────────────────────────────────
   Axios instance — Vite proxy rewrites /api → backend
   ────────────────────────────────────────────────────────────── */
const client = axios.create({
  baseURL: "/api",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

/* ── Health check ───────────────────────────────────────────── */
export async function healthCheck(): Promise<HealthResponse> {
  const { data } = await client.get<HealthResponse>("/health");
  return data;
}

/* ── Predict (JSON signal 1000×12) ──────────────────────────── */
export async function predictJSON(
  signal: ECGSignal,
): Promise<PredictResponse> {
  const { data } = await client.post<PredictResponse>("/predict", { signal });
  return data;
}

/* ── Predict (file upload) ──────────────────────────────────── */
export async function predictFile(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post<PredictResponse>("/predict-file", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/* ── Simulate (streaming prediction) ────────────────────────── */
export async function simulate(
  signal: ECGSignal,
): Promise<SimulateResponse> {
  const { data } = await client.post<SimulateResponse>("/simulate", {
    signal,
  });
  return data;
}

/* ── Metrics (real training artifacts) ──────────────────────── */
export async function fetchMetrics(): Promise<AllMetricsResponse> {
  const { data } = await client.get<AllMetricsResponse>("/metrics");
  return data;
}

/* ── ROC Curve data ─────────────────────────────────────────── */
export async function fetchROCCurve(): Promise<ROCCurveResponse> {
  const { data } = await client.get<ROCCurveResponse>("/roc-curve");
  return data;
}

/* ── Confusion Matrix data ──────────────────────────────────── */
export async function fetchConfusionMatrix(): Promise<ConfusionMatrixResponse> {
  const { data } = await client.get<ConfusionMatrixResponse>("/confusion-matrix");
  return data;
}

/* ── Precision-Recall Curve data ────────────────────────────── */
export async function fetchPRCurve(): Promise<PRCurveResponse> {
  const { data } = await client.get<PRCurveResponse>("/pr-curve");
  return data;
}

/* ── Error helper ───────────────────────────────────────────── */
export function extractErrorMessage(err: unknown): string {
  if (err instanceof AxiosError) {
    const status = err.response?.status;
    const detail =
      (err.response?.data as Record<string, unknown> | undefined)?.detail;

    if (status === 422) {
      if (typeof detail === "string") return `Validation: ${detail}`;
      if (Array.isArray(detail)) {
        const msgs = detail.map(
          (d: Record<string, string>) => d.msg ?? JSON.stringify(d),
        );
        return `Validation: ${msgs.join("; ")}`;
      }
      return "Signal shape mismatch — expected 1000×12 float array.";
    }
    if (typeof detail === "string") return `API ${status}: ${detail}`;
    if (err.code === "ECONNABORTED") return "Request timed out.";
    if (err.message.includes("Network Error"))
      return "Cannot reach the backend. Is the server running?";
    return err.message;
  }
  if (err instanceof Error) return err.message;
  return "An unexpected error occurred.";
}
