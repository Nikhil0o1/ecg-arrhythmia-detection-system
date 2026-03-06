/** Format a probability to 4 decimal places */
export function fmtProb(p: number): string {
  return p.toFixed(4);
}

/** Format a value as a percentage string */
export function fmtPct(p: number): string {
  return `${(p * 100).toFixed(2)}%`;
}

/** Convert prediction int → human label */
export function predictionLabel(pred: number): string {
  return pred === 0 ? "Normal" : "Arrhythmia";
}

/** Friendly error message from an Axios error or generic Error */
export function friendlyError(err: unknown): string {
  if (typeof err === "object" && err !== null) {
    const axiosErr = err as {
      response?: { data?: { detail?: string }; status?: number };
      message?: string;
    };
    if (axiosErr.response?.data?.detail) {
      return `API Error ${axiosErr.response.status}: ${axiosErr.response.data.detail}`;
    }
    if (axiosErr.message) {
      if (axiosErr.message.includes("Network Error")) {
        return "Cannot reach the backend. Is the server running?";
      }
      return axiosErr.message;
    }
  }
  return "An unexpected error occurred.";
}
