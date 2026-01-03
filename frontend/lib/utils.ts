import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function parseDateValue(value?: string | null): Date | null {
  if (!value) return null;
  const hasTimezone = /[zZ]|[+-]\d{2}:\d{2}$/.test(value);
  const normalized = hasTimezone ? value : `${value}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return null;
  return date;
}

export function formatDuration(ms: number): string {
  if (ms < 1000) {
    return `${Math.round(ms)}ms`;
  }
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

export function formatDurationSeconds(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "-";
  return formatDuration(Math.max(0, Math.round(seconds * 1000)));
}

export function formatPercent(value: number | undefined | null): string {
  if (value === undefined || value === null) return "-";
  return `${(value * 100).toFixed(1)}%`;
}

export function formatDate(dateStr: string): string {
  const date = parseDateValue(dateStr) ?? new Date(dateStr);
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











