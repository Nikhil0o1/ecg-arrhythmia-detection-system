import { AlertCircle, X } from "lucide-react";

interface ErrorBannerProps {
  message: string;
  onDismiss?: () => void;
}

export default function ErrorBanner({ message, onDismiss }: ErrorBannerProps) {
  return (
    <div className="animate-slide-up flex items-start gap-3 rounded-2xl border border-red-200/80 bg-gradient-to-r from-red-50 to-rose-50 p-4 shadow-sm">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl bg-red-100">
        <AlertCircle className="h-4 w-4 text-red-600" />
      </div>
      <div className="flex-1 pt-1">
        <p className="text-sm font-semibold text-red-800">{message}</p>
      </div>
      {onDismiss && (
        <button
          onClick={onDismiss}
          className="rounded-lg p-1.5 text-red-400 transition-colors hover:bg-red-100 hover:text-red-600"
        >
          <X className="h-4 w-4" />
        </button>
      )}
    </div>
  );
}
