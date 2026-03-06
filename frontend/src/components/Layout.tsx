import { type ReactNode, useState } from "react";
import {
  Activity,
  BarChart3,
  Heart,
  LineChart,
  MonitorSmartphone,
  Menu,
  X,
  Zap,
} from "lucide-react";
import type { TabId } from "@/types";

interface LayoutProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
  children: ReactNode;
}

const NAV_ITEMS: { id: TabId; label: string; icon: ReactNode; desc: string }[] = [
  { id: "prediction", label: "Prediction", icon: <Activity className="h-[18px] w-[18px]" />, desc: "ECG classification" },
  { id: "simulation", label: "Simulation", icon: <LineChart className="h-[18px] w-[18px]" />, desc: "Real-time streaming" },
  { id: "comparison", label: "Comparison", icon: <BarChart3 className="h-[18px] w-[18px]" />, desc: "Model benchmarks" },
  { id: "status", label: "System", icon: <MonitorSmartphone className="h-[18px] w-[18px]" />, desc: "Health & status" },
];

export default function Layout({ activeTab, onTabChange, children }: LayoutProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden bg-surface-50">
      {/* ── Sidebar (desktop) ────────────────────────── */}
      <aside className="hidden w-[272px] flex-shrink-0 lg:block">
        <div className="flex h-full flex-col bg-sidebar-gradient">
          <SidebarContent activeTab={activeTab} onTabChange={onTabChange} />
        </div>
      </aside>

      {/* ── Mobile drawer ────────────────────────────── */}
      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex lg:hidden">
          <div
            className="fixed inset-0 bg-black/50 backdrop-blur-sm transition-opacity"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="relative z-50 flex w-[272px] flex-col bg-sidebar-gradient shadow-2xl animate-slide-in-left">
            <button
              onClick={() => setMobileOpen(false)}
              className="absolute right-3 top-5 rounded-lg p-1.5 text-white/40 transition-colors hover:bg-white/10 hover:text-white"
            >
              <X className="h-5 w-5" />
            </button>
            <SidebarContent
              activeTab={activeTab}
              onTabChange={(t) => {
                onTabChange(t);
                setMobileOpen(false);
              }}
            />
          </aside>
        </div>
      )}

      {/* ── Main content ─────────────────────────────── */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile top bar */}
        <header className="flex h-16 items-center gap-3 border-b border-gray-200/80 bg-white/80 px-4 backdrop-blur-xl lg:hidden">
          <button
            onClick={() => setMobileOpen(true)}
            className="rounded-xl p-2 text-gray-500 transition-colors hover:bg-gray-100 hover:text-gray-700"
          >
            <Menu className="h-5 w-5" />
          </button>
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-brand-600 to-brand-400">
              <Heart className="h-4 w-4 text-white" />
            </div>
            <span className="text-sm font-bold text-gray-900">ECG Detection</span>
          </div>
        </header>

        {/* Scrollable content area with subtle background pattern */}
        <main className="relative flex-1 overflow-y-auto">
          {/* Decorative background blobs */}
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className="absolute -right-40 -top-40 h-[500px] w-[500px] rounded-full bg-brand-100/30 blur-3xl" />
            <div className="absolute -bottom-20 -left-40 h-[400px] w-[400px] rounded-full bg-purple-100/20 blur-3xl" />
          </div>

          <div className="relative px-4 py-8 sm:px-8 lg:px-12 lg:py-10">
            <div className="mx-auto max-w-6xl animate-fade-in">{children}</div>
          </div>
        </main>
      </div>
    </div>
  );
}

/* ── Sidebar inner content (shared desktop + mobile) ─── */
function SidebarContent({
  activeTab,
  onTabChange,
}: {
  activeTab: TabId;
  onTabChange: (t: TabId) => void;
}) {
  return (
    <>
      {/* Brand */}
      <div className="flex h-20 items-center gap-3.5 px-6">
        <div className="relative flex h-10 w-10 items-center justify-center rounded-xl bg-white/15 shadow-lg shadow-black/10 backdrop-blur-sm">
          <Heart className="h-5 w-5 text-white" />
          <span className="absolute -right-0.5 -top-0.5 flex h-3 w-3">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400" />
          </span>
        </div>
        <div>
          <p className="text-[15px] font-bold text-white">ECG Detection</p>
          <p className="text-[11px] font-medium text-white/40">Arrhythmia System v1.0</p>
        </div>
      </div>

      {/* Divider */}
      <div className="mx-5 h-px bg-white/10" />

      {/* Navigation */}
      <nav className="flex-1 space-y-1.5 px-4 py-6">
        <p className="mb-3 px-3 text-[10px] font-bold uppercase tracking-[0.15em] text-white/30">
          Dashboard
        </p>
        {NAV_ITEMS.map((item) => {
          const active = item.id === activeTab;
          return (
            <button
              key={item.id}
              onClick={() => onTabChange(item.id)}
              className={`group flex w-full items-center gap-3.5 rounded-xl px-3.5 py-3 text-sm font-medium transition-all duration-200 ${
                active
                  ? "bg-white/15 text-white shadow-lg shadow-black/10 backdrop-blur-sm"
                  : "text-white/50 hover:bg-white/8 hover:text-white/80"
              }`}
            >
              <span
                className={`flex h-9 w-9 items-center justify-center rounded-lg transition-all duration-200 ${
                  active
                    ? "bg-brand-500 text-white shadow-md shadow-brand-500/30"
                    : "bg-white/8 text-white/50 group-hover:bg-white/12 group-hover:text-white/70"
                }`}
              >
                {item.icon}
              </span>
              <div className="text-left">
                <p className="leading-tight">{item.label}</p>
                <p
                  className={`text-[10px] font-normal transition-colors ${
                    active ? "text-white/50" : "text-white/25 group-hover:text-white/40"
                  }`}
                >
                  {item.desc}
                </p>
              </div>
              {active && (
                <div className="ml-auto h-6 w-1 rounded-full bg-brand-400" />
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="mx-5 h-px bg-white/10" />
      <div className="flex items-center gap-3 px-6 py-5">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-white/8">
          <Zap className="h-4 w-4 text-brand-400" />
        </div>
        <div>
          <p className="text-[11px] font-semibold text-white/50">Powered by</p>
          <p className="text-[11px] text-white/30">PyTorch · FastAPI</p>
        </div>
      </div>
    </>
  );
}
