import { useEffect, useState, useCallback } from "react";
import {
  Server,
  Cpu,
  RefreshCw,
  Wifi,
  WifiOff,
  MonitorSmartphone,
  Clock,
  Activity,
} from "lucide-react";
import { fetchHealth } from "@/api/client";
import type { HealthResponse } from "@/types";
import StatusBadge from "@/components/StatusBadge";
import MetricCard from "@/components/MetricCard";
import Spinner from "@/components/Spinner";
import ErrorBanner from "@/components/ErrorBanner";

const ENDPOINTS = [
  { method: "GET", path: "/health", desc: "Service health check" },
  { method: "POST", path: "/predict", desc: "Predict from JSON signal" },
  { method: "POST", path: "/predict-file", desc: "Predict from .npy upload" },
  { method: "POST", path: "/simulate", desc: "Streaming simulation" },
];

export default function SystemStatus() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastChecked, setLastChecked] = useState<Date | null>(null);

  const check = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setError("Cannot reach the backend API. Is the server running?");
      setHealth(null);
    } finally {
      setLoading(false);
      setLastChecked(new Date());
    }
  }, []);

  useEffect(() => {
    check();
  }, [check]);

  const online = health !== null && health.status === "ok";

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500 to-teal-500 shadow-md shadow-emerald-500/25">
              <Activity className="h-5 w-5 text-white" />
            </div>
            <h1 className="page-title">System Status</h1>
          </div>
          <p className="page-subtitle mt-3">
            Live health check against the backend inference API.
          </p>
        </div>
        <button onClick={check} disabled={loading} className="btn-primary">
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
          Refresh
        </button>
      </div>

      {error && <ErrorBanner message={error} />}

      {loading && !health && (
        <div className="rounded-2xl border border-brand-100 bg-brand-50/50 py-12">
          <Spinner text="Checking API health…" />
        </div>
      )}

      {/* Status overview */}
      <div className="relative overflow-hidden rounded-2xl border border-gray-100 bg-white p-6 shadow-card">
        {/* Decorative glow */}
        <div
          className={`absolute -right-6 -top-6 h-28 w-28 rounded-full blur-3xl ${
            online ? "bg-emerald-200/40" : "bg-red-200/40"
          }`}
        />
        <div className="relative flex flex-wrap items-center gap-6">
          <div className="flex items-center gap-5">
            {online ? (
              <div className="relative">
                <div className="absolute inset-0 animate-pulse-slow rounded-2xl bg-emerald-400/20" />
                <div className="relative flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 shadow-lg shadow-emerald-500/30">
                  <Wifi className="h-8 w-8 text-white" />
                </div>
              </div>
            ) : (
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br from-red-500 to-rose-500 shadow-lg shadow-red-500/30">
                <WifiOff className="h-8 w-8 text-white" />
              </div>
            )}
            <div>
              <StatusBadge online={online} label={online ? "API Online" : "API Offline"} />
              {lastChecked && (
                <p className="mt-1.5 text-xs font-medium text-gray-400">
                  Last checked: {lastChecked.toLocaleTimeString()}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Detail cards */}
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
        <MetricCard
          label="API Status"
          value={online ? "ONLINE" : "OFFLINE"}
          variant={online ? "success" : "danger"}
          icon={<Server className="h-5 w-5" />}
        />
        <MetricCard
          label="Device"
          value={health?.device?.toUpperCase() ?? "N/A"}
          variant="info"
          icon={<Cpu className="h-5 w-5" />}
          subtext={
            health?.device?.includes("cuda")
              ? "GPU-accelerated inference"
              : "CPU inference"
          }
        />
        <MetricCard
          label="Framework"
          value="PyTorch"
          icon={<MonitorSmartphone className="h-5 w-5" />}
          subtext="FastAPI backend"
        />
        <MetricCard
          label="Last Check"
          value={lastChecked ? lastChecked.toLocaleTimeString() : "—"}
          icon={<Clock className="h-5 w-5" />}
        />
      </div>

      {/* Endpoints reference */}
      <div className="overflow-hidden rounded-2xl border border-gray-100 bg-white shadow-card">
        <div className="flex items-center gap-3 border-b border-gray-100 px-6 py-4">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-50">
            <Server className="h-4 w-4 text-brand-500" />
          </div>
          <h3 className="text-sm font-bold text-gray-800">API Endpoints</h3>
        </div>
        <div className="divide-y divide-gray-50">
          {ENDPOINTS.map((ep) => (
            <div
              key={ep.path}
              className="flex items-center gap-4 px-6 py-3.5 text-sm transition-colors duration-150 hover:bg-brand-50/30"
            >
              <span
                className={`inline-block w-16 rounded-lg px-2.5 py-1 text-center text-[11px] font-bold tracking-wide ${
                  ep.method === "GET"
                    ? "bg-gradient-to-r from-emerald-50 to-teal-50 text-emerald-700 ring-1 ring-emerald-200/50"
                    : "bg-gradient-to-r from-brand-50 to-indigo-50 text-brand-700 ring-1 ring-brand-200/50"
                }`}
              >
                {ep.method}
              </span>
              <code className="rounded-lg bg-gray-50 px-2.5 py-1 font-mono text-xs font-semibold text-gray-800">
                {ep.path}
              </code>
              <span className="text-xs text-gray-400">{ep.desc}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
