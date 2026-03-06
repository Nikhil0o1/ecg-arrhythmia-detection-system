import { motion } from "framer-motion";
import UploadCard from "@/components/UploadCard";
import PredictionCard from "@/components/PredictionCard";
import WaveformViewer from "@/components/WaveformViewer";
import SimulationChart from "@/components/SimulationChart";
import MetricsPanel from "@/components/MetricsPanel";
import ROCCurveChart from "@/components/ROCCurveChart";
import ConfusionMatrixModal from "@/components/ConfusionMatrixModal";
import ModelComparisonChart from "@/components/ModelComparisonChart";
import ProbabilityGauge from "@/components/ProbabilityGauge";
import { useECGStore } from "@/store/useECGStore";

export default function Dashboard() {
  const { signal, simulationTimeline, prediction } = useECGStore();

  return (
    <div className="mx-auto max-w-[1600px] px-6 py-6">
      {/* ── Two-column layout ───────────────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
        {/* LEFT PANEL */}
        <div className="space-y-6">
          <UploadCard />
          {prediction && <ProbabilityGauge />}
        </div>

        {/* RIGHT PANEL */}
        <div className="space-y-6">
          <PredictionCard />

          {signal && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.2 }}
            >
              <WaveformViewer />
            </motion.div>
          )}

          {simulationTimeline.length > 0 && <SimulationChart />}
        </div>
      </div>

      {/* ── Metrics row (full width) ────────────────────────── */}
      <div className="mt-6">
        <MetricsPanel />
      </div>

      {/* ── Analytics row: ROC curve + Confusion matrix ─────── */}
      <div className="mt-6 grid gap-6 lg:grid-cols-2">
        <ROCCurveChart />
        <ConfusionMatrixModal />
      </div>

      {/* ── Model comparison (full width) ───────────────────── */}
      <div className="mt-6">
        <ModelComparisonChart />
      </div>
    </div>
  );
}
