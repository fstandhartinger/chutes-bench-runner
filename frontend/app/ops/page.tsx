"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  getOpsOverview,
  getSandyMetrics,
  getSandyResources,
  getSandySandboxStats,
  type OpsOverview,
  type SandyMetricsPoint,
  type SandyResourcesResponse,
  type SandySandboxStats,
} from "@/lib/api";
import { formatDate, formatDurationSeconds, formatPercent, parseDateValue } from "@/lib/utils";
import { Loader2, Activity, Server, ListChecks } from "lucide-react";

type LineSeries = {
  label: string;
  color: string;
  values: { timestamp: string; value: number | null }[];
};

function LineChart({
  series,
  height = 180,
}: {
  series: LineSeries[];
  height?: number;
}) {
  const width = 640;
  const padding = 28;
  const allValues = series.flatMap((entry) =>
    entry.values.map((point) => (point.value ?? 0))
  );
  const maxValue = Math.max(1, ...allValues);
  const totalPoints = Math.max(1, ...series.map((entry) => entry.values.length));
  const xStep = totalPoints > 1 ? (width - padding * 2) / (totalPoints - 1) : 0;

  const buildPath = (values: { value: number | null }[]) => {
    const points = values.map((point, idx) => {
      const value = point.value ?? 0;
      const x = padding + idx * xStep;
      const y = height - padding - (value / maxValue) * (height - padding * 2);
      return `${x},${y}`;
    });
    return `M ${points.join(" L ")}`;
  };

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full">
        <rect x={0} y={0} width={width} height={height} fill="transparent" />
        {series.map((entry) => (
          <path
            key={entry.label}
            d={buildPath(entry.values)}
            fill="none"
            stroke={entry.color}
            strokeWidth={2}
            strokeLinecap="round"
          />
        ))}
      </svg>
      <div className="flex flex-wrap gap-3 text-xs text-ink-400">
        {series.map((entry) => (
          <div key={entry.label} className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: entry.color }} />
            {entry.label}
          </div>
        ))}
      </div>
    </div>
  );
}

function DualAxisLineChart({
  left,
  right,
  height = 180,
}: {
  left: LineSeries;
  right: LineSeries;
  height?: number;
}) {
  const width = 640;
  const padding = 28;
  const maxLeft = Math.max(1, ...left.values.map((point) => point.value ?? 0));
  const maxRight = Math.max(1, ...right.values.map((point) => point.value ?? 0));
  const totalPoints = Math.max(left.values.length, right.values.length, 1);
  const xStep = totalPoints > 1 ? (width - padding * 2) / (totalPoints - 1) : 0;

  const buildPath = (
    values: { value: number | null }[],
    maxValue: number
  ) => {
    const points = values.map((point, idx) => {
      const value = point.value ?? 0;
      const x = padding + idx * xStep;
      const y = height - padding - (value / maxValue) * (height - padding * 2);
      return `${x},${y}`;
    });
    return `M ${points.join(" L ")}`;
  };

  return (
    <div className="space-y-3">
      <svg viewBox={`0 0 ${width} ${height}`} className="h-48 w-full">
        <rect x={0} y={0} width={width} height={height} fill="transparent" />
        <path
          d={buildPath(left.values, maxLeft)}
          fill="none"
          stroke={left.color}
          strokeWidth={2}
          strokeLinecap="round"
        />
        <path
          d={buildPath(right.values, maxRight)}
          fill="none"
          stroke={right.color}
          strokeWidth={2}
          strokeLinecap="round"
        />
      </svg>
      <div className="flex flex-wrap gap-3 text-xs text-ink-400">
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: left.color }} />
          {left.label} (left)
        </div>
        <div className="flex items-center gap-2">
          <span className="h-2 w-2 rounded-full" style={{ backgroundColor: right.color }} />
          {right.label} (right)
        </div>
      </div>
    </div>
  );
}

function UsageBar({
  label,
  ratio,
  detail,
}: {
  label: string;
  ratio: number | null;
  detail?: string;
}) {
  const percent = ratio !== null ? Math.max(0, Math.min(1, ratio)) : null;
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-[11px] text-ink-400">
        <span>
          {label}
          {detail ? <span className="ml-2 text-ink-500">{detail}</span> : null}
        </span>
        <span>{percent === null ? "-" : formatPercent(percent)}</span>
      </div>
      <div className="h-2 w-full rounded-full bg-ink-900">
        <div
          className="h-2 rounded-full bg-moss/70"
          style={{ width: percent === null ? "0%" : `${percent * 100}%` }}
        />
      </div>
    </div>
  );
}

function formatBytes(bytes?: number | null): string {
  if (!bytes || bytes <= 0) return "-";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = 0;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 0 : 1)}${units[unitIndex]}`;
}

export default function OpsPage() {
  const [overview, setOverview] = useState<OpsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sandyMetrics, setSandyMetrics] = useState<SandyMetricsPoint[]>([]);
  const [sandyResources, setSandyResources] = useState<SandyResourcesResponse | null>(null);
  const [sandboxStats, setSandboxStats] = useState<SandySandboxStats[]>([]);

  const loadOverview = useCallback(async () => {
    try {
      const data = await getOpsOverview(720);
      setOverview(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load ops overview");
    } finally {
      setLoading(false);
    }

    try {
      const metrics = await getSandyMetrics(12);
      setSandyMetrics(metrics);
    } catch {
      setSandyMetrics([]);
    }

    try {
      const resources = await getSandyResources();
      setSandyResources(resources);
    } catch {
      setSandyResources(null);
    }

    try {
      const ids = data.workers.map((worker) => worker.worker_id);
      const stats = await getSandySandboxStats(ids);
      setSandboxStats(stats);
    } catch {
      setSandboxStats([]);
    }
  }, []);

  useEffect(() => {
    loadOverview();
    const interval = window.setInterval(loadOverview, 15000);
    return () => window.clearInterval(interval);
  }, [loadOverview]);

  const summarizeSeries = useCallback((values: Array<number | null>) => {
    const clean = values.filter((value): value is number => value !== null);
    if (clean.length === 0) {
      return { latest: null, average: null, peak: null };
    }
    const latest = clean[clean.length - 1];
    const average = clean.reduce((sum, value) => sum + value, 0) / clean.length;
    const peak = Math.max(...clean);
    return { latest, average, peak };
  }, []);

  const sandyUtilizationSeries = useMemo(() => {
    if (sandyMetrics.length === 0) return [];
    return [
      {
        label: "CPU",
        color: "#6BC46D",
        values: sandyMetrics.map((point) => ({
          timestamp: point.timestamp,
          value: point.cpu_ratio ?? null,
        })),
      },
      {
        label: "Memory",
        color: "#F28D57",
        values: sandyMetrics.map((point) => ({
          timestamp: point.timestamp,
          value: point.memory_ratio ?? null,
        })),
      },
      {
        label: "Disk",
        color: "#7B8CFB",
        values: sandyMetrics.map((point) => ({
          timestamp: point.timestamp,
          value: point.disk_ratio ?? null,
        })),
      },
    ];
  }, [sandyMetrics]);

  const workerQueueSeries = useMemo(() => {
    if (!overview) return null;
    const workerValues = overview.timeseries.map((point) => ({
      timestamp: point.timestamp,
      value: point.worker_count,
    }));
    const queuedValues = overview.timeseries.map((point) => ({
      timestamp: point.timestamp,
      value: point.queued_runs,
    }));
    return {
      left: {
        label: "Workers",
        color: "#6BC46D",
        values: workerValues,
      },
      right: {
        label: "Queued runs",
        color: "#F28D57",
        values: queuedValues,
      },
    };
  }, [overview]);

  const sandyUtilizationSummary = useMemo(() => {
    const cpuValues = sandyMetrics.map((point) => point.cpu_ratio ?? null);
    const memoryValues = sandyMetrics.map((point) => point.memory_ratio ?? null);
    const diskValues = sandyMetrics.map((point) => point.disk_ratio ?? null);
    return {
      cpu: summarizeSeries(cpuValues),
      memory: summarizeSeries(memoryValues),
      disk: summarizeSeries(diskValues),
    };
  }, [sandyMetrics, summarizeSeries]);

  const workerQueueSummary = useMemo(() => {
    if (!overview) return { workers: summarizeSeries([]), queued: summarizeSeries([]) };
    const workerValues = overview.timeseries.map((point) => point.worker_count);
    const queuedValues = overview.timeseries.map((point) => point.queued_runs);
    return {
      workers: summarizeSeries(workerValues),
      queued: summarizeSeries(queuedValues),
    };
  }, [overview, summarizeSeries]);

  const usageSnapshot = useMemo(() => {
    const latestMetric = sandyMetrics.length > 0 ? sandyMetrics[sandyMetrics.length - 1] : null;
    const cpuRatio =
      sandyResources?.cpu_percent !== null && sandyResources?.cpu_percent !== undefined
        ? sandyResources.cpu_percent / 100
        : latestMetric?.cpu_ratio ?? null;
    const memoryRatio =
      sandyResources?.memory_percent !== null && sandyResources?.memory_percent !== undefined
        ? sandyResources.memory_percent / 100
        : latestMetric?.memory_ratio ?? null;
    const diskRatio =
      sandyResources?.disk_used_ratio ?? latestMetric?.disk_ratio ?? null;
    return { cpuRatio, memoryRatio, diskRatio };
  }, [sandyMetrics, sandyResources]);

  const sandboxStatsById = useMemo(() => {
    const map = new Map<string, SandySandboxStats>();
    sandboxStats.forEach((stat) => {
      if (stat.sandbox_id) {
        map.set(stat.sandbox_id, stat);
      }
      if (stat.container_id) {
        map.set(stat.container_id, stat);
      }
    });
    return map;
  }, [sandboxStats]);

  const maxSandboxDiskBytes = useMemo(() => {
    const values = sandboxStats.map((stat) => stat.disk_bytes ?? 0);
    return Math.max(1, ...values, 1);
  }, [sandboxStats]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-moss" />
      </div>
    );
  }

  if (error || !overview) {
    return (
      <div className="mx-auto max-w-screen-xl px-6 py-10">
        <Card>
          <CardContent className="py-10 text-center text-red-300">
            {error ?? "No ops data available."}
          </CardContent>
        </Card>
      </div>
    );
  }

  const queueCounts = overview.queue_counts;

  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-4xl font-semibold text-ink-100">Ops Overview</h1>
        <p className="mt-2 text-lg text-ink-300">
          Live view of workers, queue health, and recent runs.
        </p>
      </div>

      <div className="grid gap-4 lg:grid-cols-4">
        <Card>
          <CardContent className="p-6 space-y-2">
            <div className="flex items-center gap-2 text-ink-400 text-sm">
              <Server className="h-4 w-4" />
              Active Workers
            </div>
            <div className="text-3xl font-semibold text-ink-100">
              {overview.workers.length}
            </div>
            <div className="text-xs text-ink-400">
              Max run slots per worker: {overview.worker_config.worker_max_concurrent}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 space-y-2">
            <div className="flex items-center gap-2 text-ink-400 text-sm">
              <Activity className="h-4 w-4" />
              Running Runs
            </div>
            <div className="text-3xl font-semibold text-moss">
              {queueCounts.running ?? 0}
            </div>
            <div className="text-xs text-ink-400">
              Item concurrency per worker: {overview.worker_config.worker_item_concurrency}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 space-y-2">
            <div className="flex items-center gap-2 text-ink-400 text-sm">
              <ListChecks className="h-4 w-4" />
              Queued Runs
            </div>
            <div className="text-3xl font-semibold text-ink-100">
              {queueCounts.queued ?? 0}
            </div>
            <div className="text-xs text-ink-400">
              Failed/Canceled: {(queueCounts.failed ?? 0) + (queueCounts.canceled ?? 0)}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="p-6 space-y-2">
            <div className="flex items-center gap-2 text-ink-400 text-sm">
              <ListChecks className="h-4 w-4" />
              Completed Runs
            </div>
            <div className="text-3xl font-semibold text-ink-100">
              {queueCounts.succeeded ?? 0}
            </div>
            <div className="text-xs text-ink-400">Succeeded total</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle>Token Usage</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3 text-sm text-ink-300">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-ink-400">Last 24h</span>
              <span>
                In {overview.token_stats ? (overview.token_stats.last_24h.input_tokens / 1_000_000).toFixed(2) : "-"}M · Out{" "}
                {overview.token_stats ? (overview.token_stats.last_24h.output_tokens / 1_000_000).toFixed(2) : "-"}M
              </span>
            </div>
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-ink-400">Last 7d</span>
              <span>
                In {overview.token_stats ? (overview.token_stats.last_7d.input_tokens / 1_000_000).toFixed(2) : "-"}M · Out{" "}
                {overview.token_stats ? (overview.token_stats.last_7d.output_tokens / 1_000_000).toFixed(2) : "-"}M
              </span>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Host Snapshot</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm text-ink-300">
            <div>CPU: {usageSnapshot.cpuRatio !== null ? formatPercent(usageSnapshot.cpuRatio) : "-"}</div>
            <div>Memory: {usageSnapshot.memoryRatio !== null ? formatPercent(usageSnapshot.memoryRatio) : "-"}</div>
            <div>Disk: {usageSnapshot.diskRatio !== null ? formatPercent(usageSnapshot.diskRatio) : "-"}</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Sandy Utilization (Last 12h)</CardTitle>
          </CardHeader>
          <CardContent>
            {sandyUtilizationSeries.length > 0 ? (
              <div className="space-y-4">
                <div className="grid gap-3 text-xs text-ink-400 md:grid-cols-3">
                  <div>
                    CPU: {sandyUtilizationSummary.cpu.latest !== null ? formatPercent(sandyUtilizationSummary.cpu.latest) : "-"}
                    <span className="ml-2 text-ink-500">
                      avg {sandyUtilizationSummary.cpu.average !== null ? formatPercent(sandyUtilizationSummary.cpu.average) : "-"}
                    </span>
                  </div>
                  <div>
                    Memory: {sandyUtilizationSummary.memory.latest !== null ? formatPercent(sandyUtilizationSummary.memory.latest) : "-"}
                    <span className="ml-2 text-ink-500">
                      avg {sandyUtilizationSummary.memory.average !== null ? formatPercent(sandyUtilizationSummary.memory.average) : "-"}
                    </span>
                  </div>
                  <div>
                    Disk: {sandyUtilizationSummary.disk.latest !== null ? formatPercent(sandyUtilizationSummary.disk.latest) : "-"}
                    <span className="ml-2 text-ink-500">
                      avg {sandyUtilizationSummary.disk.average !== null ? formatPercent(sandyUtilizationSummary.disk.average) : "-"}
                    </span>
                  </div>
                </div>
                <LineChart series={sandyUtilizationSeries} />
              </div>
            ) : (
              <div className="text-sm text-ink-400">No Sandy telemetry yet.</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Workers vs Queue (Last 12h)</CardTitle>
          </CardHeader>
          <CardContent>
            {workerQueueSeries ? (
              <div className="space-y-4">
                <div className="grid gap-3 text-xs text-ink-400 md:grid-cols-2">
                  <div>
                    Workers: {workerQueueSummary.workers.latest ?? "-"}
                    <span className="ml-2 text-ink-500">
                      avg {workerQueueSummary.workers.average !== null ? workerQueueSummary.workers.average.toFixed(1) : "-"}
                    </span>
                  </div>
                  <div>
                    Queued: {workerQueueSummary.queued.latest ?? "-"}
                    <span className="ml-2 text-ink-500">
                      peak {workerQueueSummary.queued.peak ?? "-"}
                    </span>
                  </div>
                </div>
                <DualAxisLineChart left={workerQueueSeries.left} right={workerQueueSeries.right} />
              </div>
            ) : (
              <div className="text-sm text-ink-400">No queue data yet.</div>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Active Worker Instances</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {overview.workers.length === 0 && (
            <div className="text-sm text-ink-400">No active workers detected.</div>
          )}
          {overview.workers.map((worker) => {
            const lastSeen = parseDateValue(worker.last_seen);
            const ageSeconds = lastSeen ? (Date.now() - lastSeen.getTime()) / 1000 : null;
            const sandbox = sandboxStatsById.get(worker.worker_id);
            const cpuRatio = sandbox?.cpu_ratio ?? usageSnapshot.cpuRatio;
            const memoryRatio = sandbox?.memory_ratio ?? usageSnapshot.memoryRatio;
            const diskRatio =
              sandbox?.disk_bytes !== undefined && sandbox?.disk_bytes !== null
                ? sandbox.disk_bytes / maxSandboxDiskBytes
                : usageSnapshot.diskRatio;
            return (
              <div key={worker.worker_id} className="rounded-lg border border-ink-600 bg-ink-800/50 p-4">
                <div className="text-sm font-medium text-ink-100">{worker.worker_id}</div>
                <div className="text-xs text-ink-400">
                  {worker.hostname ? `Host ${worker.hostname}` : "Host unknown"}
                </div>
                <div className="mt-3 text-xs text-ink-300">
                  Running runs: {worker.running_runs}/{worker.max_concurrent_runs}
                </div>
                <div className="text-xs text-ink-400">
                  Item concurrency: {worker.item_concurrency}
                </div>
                <div className="text-xs text-ink-400">
                  Last seen {ageSeconds !== null ? formatDurationSeconds(ageSeconds) : "-"} ago
                </div>
                <div className="mt-3 space-y-2">
                  <UsageBar label="CPU" ratio={cpuRatio ?? null} />
                  <UsageBar
                    label="Memory"
                    ratio={memoryRatio ?? null}
                    detail={
                      sandbox?.memory_usage_bytes
                        ? `${formatBytes(sandbox.memory_usage_bytes)}${sandbox.memory_limit_bytes ? ` / ${formatBytes(sandbox.memory_limit_bytes)}` : ""}`
                        : undefined
                    }
                  />
                  <UsageBar
                    label="Disk"
                    ratio={diskRatio ?? null}
                    detail={sandbox?.disk_bytes ? formatBytes(sandbox.disk_bytes) : undefined}
                  />
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Recently Completed</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {overview.completed_runs.length === 0 && (
              <div className="text-sm text-ink-400">No recent completions.</div>
            )}
            {overview.completed_runs.map((run) => (
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3 min-w-0">
                <Link
                  href={`/runs/${run.id}`}
                  className="block truncate text-sm font-medium text-ink-100 hover:text-moss"
                >
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400 truncate">
                  {run.benchmarks?.length ? run.benchmarks.join(", ") : "Benchmarks: -"}
                </div>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {run.completed_at ? formatDate(run.completed_at) : formatDate(run.created_at)} · {run.provider}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Running Runs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {overview.running_runs.length === 0 && (
              <div className="text-sm text-ink-400">No active runs.</div>
            )}
            {overview.running_runs.map((run) => (
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3 min-w-0">
                <Link
                  href={`/runs/${run.id}`}
                  className="block truncate text-sm font-medium text-ink-100 hover:text-moss"
                >
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400 truncate">
                  {run.benchmarks?.length ? run.benchmarks.join(", ") : "Benchmarks: -"}
                </div>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {run.started_at ? formatDate(run.started_at) : "Starting..."} · {run.provider}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Queued Runs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {overview.queued_runs.length === 0 && (
              <div className="text-sm text-ink-400">No queued runs.</div>
            )}
            {overview.queued_runs.map((run) => (
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3 min-w-0">
                <Link
                  href={`/runs/${run.id}`}
                  className="block truncate text-sm font-medium text-ink-100 hover:text-moss"
                >
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400 truncate">
                  {run.benchmarks?.length ? run.benchmarks.join(", ") : "Benchmarks: -"}
                </div>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {formatDate(run.created_at)} · {run.provider}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>How to read this page</CardTitle>
        </CardHeader>
        <CardContent className="text-sm text-ink-300 space-y-2">
          <p>
            <span className="font-medium text-ink-100">Sandboxes</span> are the isolated Sandy containers that execute
            benchmark items. A{" "}
            <span className="font-medium text-ink-100">worker</span> is the bench-runner process inside a sandbox.{" "}
          </p>
          <p>
            <span className="font-medium text-ink-100">Runs per worker</span> is the max number of benchmark runs a
            single worker can process concurrently inside one sandbox, while{" "}
            <span className="font-medium text-ink-100">item concurrency</span> is how many items within a run can be
            evaluated in parallel.
          </p>
          <p>
            Worker/queue charts show how many workers are active vs. how many runs are waiting, so you can see scaling
            behaviour at a glance.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
