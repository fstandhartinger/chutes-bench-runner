import type { BenchmarkRunBenchmark, Run } from "./api";
import { parseDateValue } from "./utils";

export interface QueueEstimate {
  queuePosition: number;
  startDelaySeconds: number | null;
  etaSeconds: number | null;
}

export function getWorkerSlots(): number {
  const raw = process.env.NEXT_PUBLIC_WORKER_SLOTS;
  const parsed = raw ? Number.parseInt(raw, 10) : Number.NaN;
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return 1;
  }
  return parsed;
}

function parseDate(value?: string | null): Date | null {
  return parseDateValue(value);
}

function expectedItems(rb: BenchmarkRunBenchmark, run: Run): number {
  if (rb.sampled_items && rb.sampled_items > 0) {
    return rb.sampled_items;
  }
  if (run.subset_count && run.subset_count > 0) {
    if (rb.total_items && rb.total_items > 0) {
      return Math.min(run.subset_count, rb.total_items);
    }
    return run.subset_count;
  }
  if (rb.total_items && rb.total_items > 0) {
    return Math.max(1, Math.floor((rb.total_items * run.subset_pct) / 100));
  }
  return 0;
}

export function getRunProgress(run: Run): { completed: number; total: number; percent: number } {
  let total = 0;
  let completed = 0;
  for (const rb of run.benchmarks) {
    total += expectedItems(rb, run);
    completed += rb.completed_items || 0;
  }
  if (total <= 0) {
    return { completed, total, percent: 0 };
  }
  const percent = Math.min(100, Math.max(0, (completed / total) * 100));
  return { completed, total, percent };
}

function perItemSeconds(rb: BenchmarkRunBenchmark, now: Date): number | null {
  if (!rb.completed_items || rb.completed_items <= 0) return null;
  const start = parseDate(rb.started_at);
  if (!start) return null;
  const end = parseDate(rb.completed_at) ?? now;
  const elapsed = Math.max(0, (end.getTime() - start.getTime()) / 1000);
  if (elapsed <= 0) return null;
  return elapsed / rb.completed_items;
}

function averagePerItemSeconds(runs: Run[], now: Date): number | null {
  let weightedTotal = 0;
  let totalItems = 0;
  for (const run of runs) {
    for (const rb of run.benchmarks) {
      const perItem = perItemSeconds(rb, now);
      if (!perItem || !rb.completed_items) continue;
      weightedTotal += perItem * rb.completed_items;
      totalItems += rb.completed_items;
    }
  }
  if (totalItems <= 0) return null;
  return weightedTotal / totalItems;
}

export function estimateRunRemainingSeconds(run: Run, now = new Date()): number | null {
  const perItemByBenchmark = new Map<string, number>();
  let weightedTotal = 0;
  let totalItems = 0;

  for (const rb of run.benchmarks) {
    const perItem = perItemSeconds(rb, now);
    if (!perItem || !rb.completed_items) continue;
    perItemByBenchmark.set(rb.benchmark_name, perItem);
    weightedTotal += perItem * rb.completed_items;
    totalItems += rb.completed_items;
  }

  if (totalItems <= 0) return null;
  const avgPerItem = weightedTotal / totalItems;

  let remainingSeconds = 0;
  for (const rb of run.benchmarks) {
    if (["succeeded", "failed", "skipped", "needs_setup"].includes(rb.status)) {
      continue;
    }
    const total = expectedItems(rb, run);
    if (total <= 0) continue;
    const remainingItems = Math.max(0, total - (rb.completed_items || 0));
    const perItem = perItemByBenchmark.get(rb.benchmark_name) ?? avgPerItem;
    remainingSeconds += remainingItems * perItem;
  }

  return remainingSeconds;
}

export function estimateRunTotalSeconds(
  run: Run,
  avgPerItemSeconds: number | null
): number | null {
  if (!avgPerItemSeconds || avgPerItemSeconds <= 0) return null;
  let totalItems = 0;
  for (const rb of run.benchmarks) {
    totalItems += expectedItems(rb, run);
  }
  if (totalItems <= 0) return null;
  return totalItems * avgPerItemSeconds;
}

export function computeQueueSchedule(
  runs: Run[],
  workerSlots: number,
  now = new Date()
): Record<string, QueueEstimate> {
  const slots = Math.max(1, workerSlots);
  const runningRuns = runs.filter((run) => run.status === "running");
  const queuedRuns = runs
    .filter((run) => run.status === "queued")
    .sort((a, b) => {
      const aTime = parseDate(a.created_at)?.getTime() ?? 0;
      const bTime = parseDate(b.created_at)?.getTime() ?? 0;
      return aTime - bTime;
    });

  const avgPerItem = averagePerItemSeconds(runs, now);
  const runningSlotTimes = runningRuns
    .map((run) => estimateRunRemainingSeconds(run, now) ?? estimateRunTotalSeconds(run, avgPerItem) ?? 0)
    .sort((a, b) => a - b);

  const slotTimes = runningSlotTimes.slice(0, slots);
  while (slotTimes.length < slots) {
    slotTimes.push(0);
  }

  const queue: Record<string, QueueEstimate> = {};
  queuedRuns.forEach((run, index) => {
    const estimatedTotal = estimateRunTotalSeconds(run, avgPerItem);
    const minValue = Math.min(...slotTimes);
    const slotIndex = slotTimes.indexOf(minValue);
    const startDelay = minValue;
    queue[run.id] = {
      queuePosition: index + 1,
      startDelaySeconds: estimatedTotal === null ? null : startDelay,
      etaSeconds: estimatedTotal === null ? null : startDelay + estimatedTotal,
    };
    if (estimatedTotal !== null) {
      slotTimes[slotIndex] = startDelay + estimatedTotal;
    }
  });

  return queue;
}
