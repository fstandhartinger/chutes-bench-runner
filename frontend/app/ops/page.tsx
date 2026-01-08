"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getOpsOverview, type OpsOverview } from "@/lib/api";
import { formatDate, formatDurationSeconds, parseDateValue, cn } from "@/lib/utils";
import { Loader2, Activity, Server, ListChecks } from "lucide-react";

function Sparkline({
  points,
  color,
}: {
  points: { value: number; label: string }[];
  color: string;
}) {
  const max = Math.max(1, ...points.map((p) => p.value));
  return (
    <div className="flex h-20 items-end gap-1">
      {points.map((point, idx) => (
        <div key={`${point.label}-${idx}`} className="flex-1">
          <div
            className={cn("w-full rounded-sm", color)}
            style={{ height: `${(point.value / max) * 100}%` }}
            title={`${point.label}: ${point.value}`}
          />
        </div>
      ))}
    </div>
  );
}

export default function OpsPage() {
  const [overview, setOverview] = useState<OpsOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadOverview = useCallback(async () => {
    try {
      const data = await getOpsOverview();
      setOverview(data);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load ops overview");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadOverview();
    const interval = window.setInterval(loadOverview, 15000);
    return () => window.clearInterval(interval);
  }, [loadOverview]);

  const workerSeries = useMemo(() => {
    if (!overview) return [];
    return overview.timeseries.map((point) => ({
      value: point.worker_count,
      label: new Date(point.timestamp).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    }));
  }, [overview]);

  const runSeries = useMemo(() => {
    if (!overview) return [];
    return overview.timeseries.map((point) => ({
      value: point.running_runs,
      label: new Date(point.timestamp).toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    }));
  }, [overview]);

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

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Worker Instances (Last 6h)</CardTitle>
          </CardHeader>
          <CardContent>
            {workerSeries.length > 0 ? (
              <Sparkline points={workerSeries} color="bg-moss/60" />
            ) : (
              <div className="text-sm text-ink-400">No worker data yet.</div>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Running Runs (Last 6h)</CardTitle>
          </CardHeader>
          <CardContent>
            {runSeries.length > 0 ? (
              <Sparkline points={runSeries} color="bg-ink-500" />
            ) : (
              <div className="text-sm text-ink-400">No run data yet.</div>
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
              </div>
            );
          })}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Queued Runs</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {overview.queued_runs.length === 0 && (
              <div className="text-sm text-ink-400">No queued runs.</div>
            )}
            {overview.queued_runs.map((run) => (
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3">
                <Link href={`/runs/${run.id}`} className="text-sm font-medium text-ink-100 hover:text-moss">
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {formatDate(run.created_at)}
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
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3">
                <Link href={`/runs/${run.id}`} className="text-sm font-medium text-ink-100 hover:text-moss">
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {run.started_at ? formatDate(run.started_at) : "Starting..."}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Recently Completed</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            {overview.completed_runs.length === 0 && (
              <div className="text-sm text-ink-400">No recent completions.</div>
            )}
            {overview.completed_runs.map((run) => (
              <div key={run.id} className="rounded-lg border border-ink-600 bg-ink-800/40 p-3">
                <Link href={`/runs/${run.id}`} className="text-sm font-medium text-ink-100 hover:text-moss">
                  {run.model_slug}
                </Link>
                <div className="text-xs text-ink-400">
                  {run.subset_count ? `${run.subset_count} items` : `${run.subset_pct}%`} ·{" "}
                  {run.completed_at ? formatDate(run.completed_at) : formatDate(run.created_at)}
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
