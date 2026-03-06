import { useCallback, useRef, useState, type ReactNode } from "react";
import { Upload, FileUp } from "lucide-react";

interface FileDropZoneProps {
  accept?: string;
  label?: string;
  hint?: string;
  icon?: ReactNode;
  onFile: (file: File) => void;
}

export default function FileDropZone({
  accept = ".npy",
  label = "Drop your .npy file here or click to browse",
  hint = "Only .npy files with 1 000 float samples are accepted",
  icon,
  onFile,
}: FileDropZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback(
    (f: File | undefined) => {
      if (f) onFile(f);
    },
    [onFile],
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFile(e.dataTransfer.files[0]);
      }}
      onClick={() => inputRef.current?.click()}
      className={`group relative flex cursor-pointer flex-col items-center justify-center gap-4 overflow-hidden rounded-2xl border-2 border-dashed px-8 py-14 transition-all duration-300 ${
        dragging
          ? "border-brand-500 bg-brand-50/80 shadow-glow"
          : "border-gray-200 bg-white hover:border-brand-300 hover:bg-brand-50/30 hover:shadow-glow"
      }`}
    >
      {/* Background decoration */}
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_50%_120%,rgba(99,102,241,0.05),transparent_70%)]" />

      <div
        className={`relative flex h-16 w-16 items-center justify-center rounded-2xl transition-all duration-300 ${
          dragging
            ? "bg-brand-100 text-brand-600 scale-110"
            : "bg-gray-100 text-gray-400 group-hover:bg-brand-100 group-hover:text-brand-600 group-hover:scale-105"
        }`}
      >
        {icon ?? (dragging ? <FileUp className="h-7 w-7" /> : <Upload className="h-7 w-7" />)}
      </div>
      <div className="relative text-center">
        <p className="text-sm font-semibold text-gray-700">{label}</p>
        <p className="mt-1 text-xs text-gray-400">{hint}</p>
      </div>
      <div className="relative">
        <span className="rounded-xl bg-brand-600 px-5 py-2.5 text-xs font-bold text-white shadow-md shadow-brand-500/25 transition-all duration-200 group-hover:shadow-lg group-hover:shadow-brand-500/30">
          Browse Files
        </span>
      </div>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
    </div>
  );
}
