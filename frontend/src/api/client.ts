import axios from "axios";
import type {
  HealthResponse,
  PredictResponse,
  SimulateResponse,
} from "@/types";

/* ──────────────────────────────────────────────────────────────
   Axios instance — 
   - Dev: Uses Vite proxy (/api) → localhost:8000
   - Prod: Uses VITE_API_URL environment variable (e.g., Render backend)
   ────────────────────────────────────────────────────────────── */
const getBaseURL = () => {
  const apiUrl = import.meta.env.VITE_API_URL;
  return apiUrl || "/api"; // Default to proxy path for local dev
};

const api = axios.create({
  baseURL: getBaseURL(),
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
