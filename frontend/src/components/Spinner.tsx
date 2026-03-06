import { Loader2 } from "lucide-react";

interface SpinnerProps {
  text?: string;
  className?: string;
}

export default function Spinner({ text, className = "" }: SpinnerProps) {
  return (
    <div className={`flex items-center justify-center gap-3 ${className}`}>
      <div className="relative">
        <div className="absolute inset-0 animate-ping rounded-full bg-brand-400/20" />
        <Loader2 className="relative h-6 w-6 animate-spin text-brand-600" />
      </div>
      {text && <span className="text-sm font-medium text-gray-500">{text}</span>}
    </div>
  );
}
