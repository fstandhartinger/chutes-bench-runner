import { describe, it, expect } from "vitest";
import { formatPercent, formatDuration, getStatusColor } from "../lib/utils";

describe("formatPercent", () => {
  it("formats decimal as percentage", () => {
    expect(formatPercent(0.856)).toBe("85.6%");
    expect(formatPercent(1.0)).toBe("100.0%");
    expect(formatPercent(0)).toBe("0.0%");
  });

  it("handles undefined/null", () => {
    expect(formatPercent(undefined)).toBe("-");
    expect(formatPercent(null)).toBe("-");
  });
});

describe("formatDuration", () => {
  it("formats seconds", () => {
    expect(formatDuration(5000)).toBe("5s");
    expect(formatDuration(45000)).toBe("45s");
  });

  it("formats minutes", () => {
    expect(formatDuration(60000)).toBe("1m 0s");
    expect(formatDuration(90000)).toBe("1m 30s");
  });

  it("formats hours", () => {
    expect(formatDuration(3600000)).toBe("1h 0m");
    expect(formatDuration(5400000)).toBe("1h 30m");
  });
});

describe("getStatusColor", () => {
  it("returns correct colors", () => {
    expect(getStatusColor("succeeded")).toBe("text-moss");
    expect(getStatusColor("failed")).toBe("text-red-400");
    expect(getStatusColor("running")).toBe("text-moss");
    expect(getStatusColor("queued")).toBe("text-ink-400");
    expect(getStatusColor("canceled")).toBe("text-yellow-400");
  });

  it("returns default for unknown status", () => {
    expect(getStatusColor("unknown")).toBe("text-ink-300");
  });
});


