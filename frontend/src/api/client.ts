import axios from "axios";
import type {
  HealthResponse,
  PredictResponse,
  SimulateResponse,
} from "@/types";

/* ──────────────────────────────────────────────────────────────
   Axios instance — all requests go through the Vite dev proxy
   that rewrites /api → http://localhost:8000
   ────────────────────────────────────────────────────────────── */
const api = axios.create({
  baseURL: "/api",
  timeout: 30_000,
  headers: { "Content-Type": "application/json" },
});

/* ── Health ─────────────────────────────────────────────────── */
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

/* ── Predict (JSON signal) ──────────────────────────────────── */
export async function predictSignal(
  signal: number[],
): Promise<PredictResponse> {
  const { data } = await api.post<PredictResponse>("/predict", { signal });
  return data;
}

/* ── Predict (file upload) ──────────────────────────────────── */
export async function predictFile(file: File): Promise<PredictResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<PredictResponse>("/predict-file", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/* ── Simulate ───────────────────────────────────────────────── */
export async function simulateSignal(
  signal: number[],
): Promise<SimulateResponse> {
  const { data } = await api.post<SimulateResponse>("/simulate", { signal });
  return data;
}
