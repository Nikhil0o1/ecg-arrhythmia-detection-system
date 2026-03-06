import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import {
  Activity,
  Wifi,
  WifiOff,
  Cpu,
  Moon,
  Sun,
} from "lucide-react";
import { useECGStore } from "@/store/useECGStore";
import { healthCheck } from "@/api/api";

/* ── Animated ECG heartbeat SVG ──────────────────────────────── */
function HeartbeatLine() {
  const pathRef = useRef<SVGPathElement>(null);

  useEffect(() => {
    const el = pathRef.current;
    if (!el) return;
    const len = el.getTotalLength();
    el.style.strokeDasharray = `${len}`;
    el.style.strokeDashoffset = `${len}`;
    el.style.animation = "none";
    void el.getBoundingClientRect();
    el.style.animation = `ecgDraw 2.2s ease-in-out infinite`;
  }, []);

  return (
    <svg
      viewBox="0 0 200 40"
      className="h-8 w-32 opacity-60"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      <style>{`
        @keyframes ecgDraw {
          0% { stroke-dashoffset: 340; opacity: 0.3; }
          50% { opacity: 1; }
          100% { stroke-dashoffset: 0; opacity: 0.3; }
        }
      `}</style>
      <path
        ref={pathRef}
        d="M0 20 L40 20 L50 20 L55 8 L60 32 L65 4 L70 36 L75 20 L85 20 L95 20 L100 14 L105 26 L110 20 L200 20"
        stroke="currentColor"
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-emerald-400"
      />
    </svg>
  );
}

export default function Navbar() {
  const { connected, device, setHealth, theme, toggleTheme } =
    useECGStore();

  useEffect(() => {
    let mounted = true;
    const poll = async () => {
      try {
        const res = await healthCheck();
        if (mounted) setHealth(res.status === "ok", res.device);
      } catch {
        if (mounted) setHealth(false, "unknown");
      }
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, [setHealth]);

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: "easeOut" }}
      className="sticky top-0 z-50 border-b border-white/5 bg-clinical-bg/80 backdrop-blur-xl dark:bg-clinical-bg/80 bg-white/80 dark:border-white/5 border-gray-200/60"
    >
      <div className="mx-auto flex h-14 sm:h-16 max-w-[1600px] items-center justify-between px-3 sm:px-4 md:px-6">
        {/* ── Left: Logo + heartbeat ─────────────────────────── */}
        <div className="flex items-center gap-2 sm:gap-3 md:gap-4 min-w-0">
          <div className="flex items-center gap-1.5 sm:gap-2">
            <div className="flex h-8 sm:h-9 w-8 sm:w-9 items-center justify-center rounded-lg sm:rounded-xl bg-gradient-to-br from-emerald-500 to-teal-600 shadow-lg shadow-emerald-500/20 flex-shrink-0">
              <Activity className="h-4 sm:h-5 w-4 sm:w-5 text-white" strokeWidth={2.5} />
            </div>
            <span className="text-lg sm:text-xl font-bold tracking-tight dark:text-white text-gray-900 whitespace-nowrap">
              Cardio
              <span className="bg-gradient-to-r from-emerald-400 to-teal-400 bg-clip-text text-transparent">
                AI
              </span>
            </span>
          </div>
          <div className="hidden sm:block">
            <HeartbeatLine />
          </div>
        </div>

        {/* ── Right: status badges (responsive) ───────────────────────────── */}
        <div className="flex items-center gap-2 sm:gap-3">
          {/* Connection status */}
          <motion.div
            layout
            className={`flex items-center gap-1.5 sm:gap-2 rounded-full px-2.5 sm:px-3.5 py-1 sm:py-1.5 text-xs font-medium backdrop-blur-sm ${
              connected
                ? "border border-emerald-500/20 bg-emerald-500/10 text-emerald-400"
                : "border border-red-500/20 bg-red-500/10 text-red-400"
            }`}
          >
            {connected ? (
              <Wifi className="h-3 sm:h-3.5 w-3 sm:w-3.5" />
            ) : (
              <WifiOff className="h-3 sm:h-3.5 w-3 sm:w-3.5" />
            )}
            <span className="hidden sm:inline">
              {connected ? "Connected" : "Disconnected"}
            </span>
            <span className="hidden sm:block h-1.5 w-1.5 rounded-full" />
            <span
              className={`h-1 w-1 sm:h-1.5 sm:w-1.5 rounded-full ${
                connected
                  ? "bg-emerald-400 shadow-sm shadow-emerald-400"
                  : "bg-red-400 shadow-sm shadow-red-400"
              }`}
            />
          </motion.div>

          {/* Device badge - hidden on small mobile */}
          {connected && (
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              className="hidden sm:flex items-center gap-1.5 rounded-full border border-blue-500/20 bg-blue-500/10 px-3 py-1.5 text-xs font-medium text-blue-400"
            >
              <Cpu className="h-3.5 w-3.5" />
              {device.toUpperCase()}
            </motion.div>
          )}

          {/* Theme toggle */}
          <button
            onClick={toggleTheme}
            className="flex h-8 sm:h-9 w-8 sm:w-9 items-center justify-center rounded-lg sm:rounded-xl border border-white/10 dark:border-white/10 border-gray-200 bg-white/5 dark:bg-white/5 bg-gray-100 transition-colors hover:bg-white/10 dark:hover:bg-white/10 hover:bg-gray-200"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? (
              <Sun className="h-4 w-4 dark:text-gray-400 text-gray-600" />
            ) : (
              <Moon className="h-4 w-4 dark:text-gray-400 text-gray-600" />
            )}
          </button>
        </div>
      </div>
    </motion.nav>
  );
}
