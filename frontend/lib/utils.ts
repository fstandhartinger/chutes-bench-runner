import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatDuration(ms: number): string {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

export function formatPercent(value: number | undefined | null): string {
  if (value === undefined || value === null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDate(dateStr: string): string {
  const date = new Date(dateStr);
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function getStatusColor(status: string): string {
  switch (status) {
    case "queued":
      return "text-ink-400";
    case "running":
      return "text-moss";
    case "succeeded":
      return "text-moss";
    case "failed":
      return "text-red-400";
    case "canceled":
      return "text-yellow-400";
    case "pending":
      return "text-ink-400";
    case "needs_setup":
      return "text-yellow-400";
    case "skipped":
      return "text-ink-400";
    default:
      return "text-ink-300";
  }
}

export function getStatusBgColor(status: string): string {
  switch (status) {
    case "queued":
      return "bg-ink-600";
    case "running":
      return "bg-moss/20";
    case "succeeded":
      return "bg-moss/20";
    case "failed":
      return "bg-red-500/20";
    case "canceled":
      return "bg-yellow-500/20";
    default:
      return "bg-ink-600";
  }
}

