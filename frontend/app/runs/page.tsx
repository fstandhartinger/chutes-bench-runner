"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cancelRun, getRuns, type Run } from "@/lib/api";
import {
  computeQueueSchedule,
  estimateRunRemainingSeconds,
  getRunProgress,
  getWorkerSlots,
} from "@/lib/eta";
import {
  formatDate,
  formatDuration,
  formatDurationSeconds,
  formatPercent,
  getStatusColor,
  getStatusBgColor,
  parseDateValue,
  cn,
} from "@/lib/utils";
import { Loader2, ExternalLink, Download, Search } from "lucide-react";
import { Input } from "@/components/ui/input";

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelFilter, setModelFilter] = useState("");
  const [cancelingRunId, setCancelingRunId] = useState<string | null>(null);

  const loadRuns = useCallback(async () => {
    try {
      const data = await getRuns(undefined, 100);
      setRuns(data.runs);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load runs");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadRuns();
  }, [loadRuns]);

  useEffect(() => {
    if (!runs.some((run) => ["queued", "running"].includes(run.status))) return;
    const interval = window.setInterval(loadRuns, 15000);
    return () => window.clearInterval(interval);
  }, [runs, loadRuns]);

  const handleCancel = async (runId: string) => {
    if (!confirm("Cancel this benchmark run?")) return;
    setCancelingRunId(runId);
    try {
      await cancelRun(runId);
      await loadRuns();
    } catch (e) {
      console.error("Failed to cancel run", e);
    } finally {
      setCancelingRunId(null);
    }
  };

  const filteredRuns = runs.filter((run) =>
    run.model_slug.toLowerCase().includes(modelFilter.toLowerCase())
  );
  const workerSlots = getWorkerSlots();
  const queueSchedule = useMemo(
    () =>
      computeQueueSchedule(
        runs.filter((run) => ["queued", "running"].includes(run.status)),
        workerSlots
      ),
    [runs, workerSlots]
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-moss" />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10">
      <div className="mb-10">
        <h1 className="text-4xl font-semibold text-ink-100">Run History</h1>
        <p className="mt-2 text-lg text-ink-300">
          View past benchmark runs and their results
        </p>
      </div>

      <div className="mb-8 flex items-center gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-400" />
          <Input
            placeholder="Filter by model name..."
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="pl-9 bg-ink-800/50 border-ink-600 focus:border-moss/50"
          />
        </div>
        {modelFilter && (
          <Button
            variant="ghost"
            onClick={() => setModelFilter("")}
            className="text-ink-400 hover:text-ink-200"
          >
            Clear Filter
          </Button>
        )}
      </div>

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-red-400">
          {error}
        </div>
      )}

      {filteredRuns.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-ink-400">
              {modelFilter ? "No runs found matching your filter" : "No benchmark runs yet"}
            </p>
            {!modelFilter && (
              <Button asChild className="mt-4">
                <Link href="/">Start Your First Run</Link>
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {filteredRuns.map((run) => {
            const now = new Date();
            const startedAt = parseDateValue(run.started_at);
            const completedAt = parseDateValue(run.completed_at);
            const elapsedMs = startedAt
              ? Math.max(0, (completedAt ? completedAt.getTime() : now.getTime()) - startedAt.getTime())
              : null;
            const etaSeconds =
              run.status === "running" ? estimateRunRemainingSeconds(run, now) : null;
            const queueInfo =
              run.status === "queued" ? queueSchedule[run.id] : undefined;
            const queueDelaySeconds = queueInfo?.startDelaySeconds ?? null;
            const progress = getRunProgress(run);

            return (
              <Card key={run.id} className="overflow-hidden">
              <div className="flex flex-col gap-4 p-6 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 space-y-1">
                  <div className="flex flex-wrap items-center gap-3">
                    <h3 className="text-lg font-medium break-words">{run.model_slug}</h3>
                    <span
                      className={cn(
                        "rounded-full px-2.5 py-0.5 text-xs font-medium",
                        getStatusBgColor(run.status),
                        getStatusColor(run.status)
                      )}
                    >
                      {run.status}
                    </span>
                  </div>
                  <p className="text-sm text-ink-400">
                    {run.subset_count
                      ? `${run.subset_count} items`
                      : `${run.subset_pct}% subset`}{" "}
                    · {run.benchmarks.length} benchmarks · {formatDate(run.created_at)}
                  </p>
                  {run.status === "running" && (
                    <p className="text-xs text-ink-400">
                      {elapsedMs !== null
                        ? `Elapsed ${formatDuration(elapsedMs)}`
                        : "Elapsed -"}{" "}
                      {etaSeconds !== null
                        ? `· ETA ~ ${formatDurationSeconds(etaSeconds)}`
                        : "· ETA estimating..."}
                    </p>
                  )}
                  {run.status === "queued" && (
                    <p className="text-xs text-ink-400">
                      Queued
                      {queueInfo ? ` (pos ${queueInfo.queuePosition})` : ""}{" "}
                      {queueDelaySeconds !== null
                        ? `· start in ~${formatDurationSeconds(queueDelaySeconds)}`
                        : "· waiting for worker"}
                    </p>
                  )}
                  {["running", "queued"].includes(run.status) && progress.total > 0 && (
                    <div className="max-w-sm space-y-1 pt-2">
                      <div className="flex items-center justify-between text-xs text-ink-400">
                        <span>
                          {progress.completed}/{progress.total} items
                        </span>
                        <span>{Math.round(progress.percent)}%</span>
                      </div>
                      <Progress
                        value={progress.percent}
                        className={run.status === "running" ? "progress-animate" : ""}
                      />
                    </div>
                  )}
                  {["succeeded", "failed", "canceled"].includes(run.status) &&
                    elapsedMs !== null && (
                      <p className="text-xs text-ink-400">
                        Duration {formatDuration(elapsedMs)}
                      </p>
                    )}
                </div>

                <div className="flex w-full flex-col gap-4 sm:w-auto sm:flex-row sm:items-center sm:gap-6">
                  {run.overall_score !== undefined && run.overall_score !== null && (
                    <div className="text-left sm:text-right">
                      <div className="text-2xl font-semibold text-moss">
                        {formatPercent(run.overall_score)}
                      </div>
                      <div className="text-xs text-ink-400">Overall Score</div>
                    </div>
                  )}

                  <div className="flex w-full flex-wrap gap-2 sm:w-auto sm:justify-end">
                    <Button variant="secondary" size="sm" asChild className="w-full sm:w-auto">
                      <Link href={`/runs/${run.id}`}>
                        <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
                        View
                      </Link>
                    </Button>
                    {["queued", "running"].includes(run.status) && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full sm:w-auto"
                        onClick={() => handleCancel(run.id)}
                        disabled={cancelingRunId === run.id}
                      >
                        {cancelingRunId === run.id ? (
                          <>
                            <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin" />
                            Canceling
                          </>
                        ) : (
                          "Cancel"
                        )}
                      </Button>
                    )}
                    {(run.status === "succeeded" || run.status === "failed") && (
                      <Button variant="outline" size="sm" asChild className="w-full sm:w-auto">
                        <a
                          href={`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000"}/api/runs/${run.id}/export?format=csv`}
                          download
                        >
                          <Download className="mr-1.5 h-3.5 w-3.5" />
                          CSV
                        </a>
                      </Button>
                    )}
                  </div>
                </div>
              </div>

              {/* Benchmark mini summary */}
              <div className="flex flex-wrap gap-4 border-t border-ink-500 bg-ink-800/30 px-6 py-3">
                {run.benchmarks.slice(0, 6).map((rb) => (
                  <div key={rb.id} className="flex items-center gap-2 text-sm">
                    <span
                      className={cn(
                        "h-2 w-2 rounded-full",
                        rb.status === "succeeded" ? "bg-moss" : 
                        rb.status === "failed" ? "bg-red-400" :
                        rb.status === "running" ? "bg-moss animate-pulse" :
                        "bg-ink-500"
                      )}
                    />
                    <span className="text-ink-300">{rb.benchmark_name}</span>
                    {rb.score !== undefined && rb.score !== null && (
                      <span className="text-moss">{formatPercent(rb.score)}</span>
                    )}
                  </div>
                ))}
                {run.benchmarks.length > 6 && (
                  <span className="text-ink-400">
                    +{run.benchmarks.length - 6} more
                  </span>
                )}
              </div>
            </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
