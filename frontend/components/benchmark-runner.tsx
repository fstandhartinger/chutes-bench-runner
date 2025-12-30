"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  getModels,
  getBenchmarks,
  createRun,
  getRun,
  getRuns,
  createEventSource,
  type Model,
  type Benchmark,
  type Run,
  type RunEvent,
} from "@/lib/api";
import {
  computeQueueSchedule,
  estimateRunRemainingSeconds,
  getWorkerSlots,
  type QueueEstimate,
} from "@/lib/eta";
import {
  cn,
  formatDuration,
  formatDurationSeconds,
  formatPercent,
  getStatusColor,
} from "@/lib/utils";
import { Play, Loader2, AlertCircle, CheckCircle2, XCircle } from "lucide-react";

const SUBSET_OPTIONS = [
  { value: "100", label: "100% (Full)" },
  { value: "50", label: "50%" },
  { value: "25", label: "25%" },
  { value: "10", label: "10%" },
  { value: "5", label: "5%" },
  { value: "1", label: "1% (Quick test)" },
];

export function BenchmarkRunner() {
  const [models, setModels] = useState<Model[]>([]);
  const [benchmarks, setBenchmarks] = useState<Benchmark[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>("");
  const [subsetPct, setSubsetPct] = useState<string>("10");
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [currentRun, setCurrentRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [queueStats, setQueueStats] = useState<{ running: number; queued: number } | null>(null);
  const [queueSchedule, setQueueSchedule] = useState<Record<string, QueueEstimate>>({});
  const progressRef = useRef<HTMLDivElement | null>(null);
  const eventSourceRef = useRef<EventSource | null>(null);
  const lastEventIdRef = useRef<string | null>(null);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const workerSlots = getWorkerSlots();
  const subsetValue = Number(subsetPct);

  useEffect(() => {
    runningRef.current = running;
  }, [running]);

  useEffect(() => {
    if (currentRun) {
      requestAnimationFrame(() => {
        progressRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    }
  }, [currentRun]);

  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
        eventSourceRef.current = null;
      }
      if (reconnectTimeoutRef.current !== null) {
        window.clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, []);

  const loadQueueStatus = useCallback(async () => {
    try {
      const [runningRes, queuedRes] = await Promise.all([
        getRuns("running", 200),
        getRuns("queued", 200),
      ]);
      setQueueStats({
        running: runningRes.runs.length,
        queued: queuedRes.runs.length,
      });
      setQueueSchedule(
        computeQueueSchedule(
          [...runningRes.runs, ...queuedRes.runs],
          workerSlots
        )
      );
    } catch (e) {
      console.warn("Failed to load queue status", e);
    }
  }, [workerSlots]);

  useEffect(() => {
    loadQueueStatus();
    const interval = window.setInterval(loadQueueStatus, 15000);
    return () => window.clearInterval(interval);
  }, [loadQueueStatus]);

  useEffect(() => {
    if (!currentRun || !["queued", "running"].includes(currentRun.status)) return;
    const interval = window.setInterval(async () => {
      try {
        const updated = await getRun(currentRun.id);
        setCurrentRun(updated);
      } catch (e) {
        console.warn("Failed to refresh run", e);
      }
    }, 15000);
    return () => window.clearInterval(interval);
  }, [currentRun?.id, currentRun?.status]);

  // Load models and benchmarks
  useEffect(() => {
    async function load() {
      try {
        const [modelsRes, benchmarksRes] = await Promise.all([
          getModels(),
          getBenchmarks(),
        ]);
        setModels(modelsRes.models);
        setBenchmarks(benchmarksRes.benchmarks);
        // Select all enabled benchmarks by default
        setSelectedBenchmarks(
          new Set(
            benchmarksRes.benchmarks
              .filter((b) => b.is_enabled)
              .map((b) => b.name)
          )
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  // Handle benchmark selection
  const toggleBenchmark = (name: string) => {
    setSelectedBenchmarks((prev) => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  // Start benchmark run
  const startRun = async () => {
    if (!selectedModel || selectedBenchmarks.size === 0) return;

    setRunning(true);
    setError(null);
    setEvents([]);
    lastEventIdRef.current = null;
    reconnectAttemptRef.current = 0;
    if (reconnectTimeoutRef.current !== null) {
      window.clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    try {
      const run = await createRun(
        selectedModel,
        parseInt(subsetPct),
        Array.from(selectedBenchmarks)
      );
      setCurrentRun(run);

      const connectEventSource = (runId: string) => {
        if (eventSourceRef.current) {
          eventSourceRef.current.close();
        }

        const eventSource = createEventSource(runId, lastEventIdRef.current || undefined);
        eventSourceRef.current = eventSource;

        eventSource.onopen = () => {
          reconnectAttemptRef.current = 0;
        };

        eventSource.onmessage = (e) => {
          try {
            const event = JSON.parse(e.data) as RunEvent;
            lastEventIdRef.current = e.lastEventId || event.id;
            setEvents((prev) => [...prev, event]);

            // Check for run completion/failure events
            if (event.event_type === "run_completed" || event.event_type === "run_failed") {
              setRunning(false);
              setCurrentRun((prev) =>
                prev
                  ? {
                      ...prev,
                      status: event.event_type === "run_completed" ? "succeeded" : "failed",
                    }
                  : null
              );
              eventSource.close();
              eventSourceRef.current = null;
            }
          } catch (err) {
            console.warn("Failed to parse event:", err);
          }
        };

        eventSource.addEventListener("done", () => {
          setRunning(false);
          eventSource.close();
          eventSourceRef.current = null;
        });

        eventSource.onerror = () => {
          if (!runningRef.current) return;

          if (reconnectTimeoutRef.current !== null) return;
          const attempt = reconnectAttemptRef.current;
          const delay = Math.min(10000, 1000 * 2 ** attempt);
          reconnectAttemptRef.current += 1;

          reconnectTimeoutRef.current = window.setTimeout(() => {
            reconnectTimeoutRef.current = null;
            connectEventSource(runId);
          }, delay);
        };
      };

      // Subscribe to events
      connectEventSource(run.id);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start run");
      setRunning(false);
    }
  };

  // Compute progress from events
  const progressData = useCallback(() => {
    if (!currentRun) return { overall: 0, benchmarks: {} as Record<string, { completed: number; total: number; score?: number; status?: string }>, status: "queued" };

    const benchmarkProgress: Record<string, { completed: number; total: number; score?: number; status?: string }> = {};
    let runStatus = currentRun.status;

    // Initialize from run benchmarks (may have 0 values initially)
    for (const rb of currentRun.benchmarks) {
      benchmarkProgress[rb.benchmark_name] = {
        completed: rb.completed_items || 0,
        total: rb.sampled_items || rb.total_items || 0,
        score: rb.score,
        status: rb.status,
      };
    }

    // Update from events - events have the most current data
    for (const event of events) {
      if (event.event_type === "run_started") {
        runStatus = "running";
      } else if (event.event_type === "run_completed") {
        runStatus = "succeeded";
      } else if (event.event_type === "run_failed") {
        runStatus = "failed";
      } else if (event.event_type === "benchmark_started" && event.benchmark_name) {
        if (benchmarkProgress[event.benchmark_name]) {
          benchmarkProgress[event.benchmark_name].status = "running";
        }
      } else if (event.event_type === "benchmark_progress" && event.benchmark_name && event.data) {
        const data = event.data as { completed?: number; total?: number; current_accuracy?: number };
        benchmarkProgress[event.benchmark_name] = {
          ...benchmarkProgress[event.benchmark_name],
          completed: data.completed || 0,
          total: data.total || benchmarkProgress[event.benchmark_name]?.total || 0,
          score: data.current_accuracy,
          status: "running",
        };
      } else if (event.event_type === "benchmark_completed" && event.benchmark_name && event.data) {
        const data = event.data as { score?: number };
        if (benchmarkProgress[event.benchmark_name]) {
          benchmarkProgress[event.benchmark_name].status = "succeeded";
          if (data.score !== undefined) {
            benchmarkProgress[event.benchmark_name].score = data.score;
          }
        }
      } else if (event.event_type === "benchmark_failed" && event.benchmark_name) {
        if (benchmarkProgress[event.benchmark_name]) {
          benchmarkProgress[event.benchmark_name].status = "failed";
        }
      }
    }

    // Calculate overall progress from benchmark progress
    let totalCompleted = 0;
    let totalItems = 0;
    for (const bp of Object.values(benchmarkProgress)) {
      totalCompleted += bp.completed;
      totalItems += bp.total;
    }

    const overall = totalItems > 0 ? (totalCompleted / totalItems) * 100 : 0;
    return { overall, benchmarks: benchmarkProgress, status: runStatus };
  }, [currentRun, events]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-moss" />
      </div>
    );
  }

  const progress = progressData();
  const queueInfo = currentRun ? queueSchedule[currentRun.id] : undefined;
  const queueDelaySeconds = queueInfo?.startDelaySeconds ?? null;
  const now = new Date();
  const startedAt = currentRun?.started_at ? new Date(currentRun.started_at) : null;
  const elapsedMs =
    startedAt &&
    (currentRun?.completed_at
      ? new Date(currentRun.completed_at).getTime()
      : now.getTime()) -
      startedAt.getTime();
  const etaSeconds =
    currentRun && currentRun.status === "running"
      ? estimateRunRemainingSeconds(currentRun, now)
      : null;

  return (
    <div className="space-y-8">
      {/* Configuration Section */}
      <Card>
        <CardHeader>
          <CardTitle>Configure Benchmark Run</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {error && (
            <div className="flex items-center gap-2 rounded-lg bg-red-500/10 p-4 text-red-400">
              <AlertCircle className="h-5 w-5" />
              <span>{error}</span>
            </div>
          )}

          {/* Model Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-ink-200">Model</label>
            <Select value={selectedModel} onValueChange={setSelectedModel}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder="Select a model..." />
              </SelectTrigger>
              <SelectContent>
                {models
                  .filter((model) => model.is_active && model.instance_count > 0)
                  .map((model) => (
                    <SelectItem key={model.id} value={model.id}>
                      <span>{model.name}</span>
                    </SelectItem>
                  ))}
              </SelectContent>
            </Select>
          </div>

          {/* Subset Selection */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-ink-200">
              Subset Size
            </label>
            <Select value={subsetPct} onValueChange={setSubsetPct}>
              <SelectTrigger className="w-full max-w-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SUBSET_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-ink-400">
              Deterministic sampling ensures reproducible results. Minimum sample size is 1 item.
            </p>
          </div>

          {/* Benchmark Selection */}
          <div className="space-y-3">
            <label className="text-sm font-medium text-ink-200">
              Benchmarks
            </label>
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {benchmarks.map((benchmark) => {
                const totalItems = benchmark.total_items || 0;
                const sampledItems =
                  totalItems > 0
                    ? Math.max(1, Math.floor((totalItems * subsetValue) / 100))
                    : 0;
                const estSeconds =
                  benchmark.avg_item_latency_ms && sampledItems > 0
                    ? (benchmark.avg_item_latency_ms / 1000) * sampledItems
                    : null;

                return (
                  <label
                    key={benchmark.name}
                    className={cn(
                      "flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-colors",
                      selectedBenchmarks.has(benchmark.name)
                        ? "border-moss/50 bg-moss/5"
                        : "border-ink-500 bg-ink-700/50 hover:border-ink-400"
                    )}
                  >
                    <Checkbox
                      checked={selectedBenchmarks.has(benchmark.name)}
                      onCheckedChange={() => toggleBenchmark(benchmark.name)}
                      disabled={!benchmark.is_enabled}
                    />
                    <div className="flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-medium">{benchmark.display_name}</span>
                        {benchmark.requires_setup && (
                          <span className="rounded bg-yellow-500/20 px-1.5 py-0.5 text-xs text-yellow-400">
                            Setup required
                          </span>
                        )}
                      </div>
                      <p className="mt-1 text-xs text-ink-400">
                        {totalItems > 0
                          ? `${sampledItems}/${totalItems} tests`
                          : "Test count unavailable"}
                        {estSeconds !== null
                          ? ` · Est. ${formatDurationSeconds(estSeconds)}`
                          : ""}
                      </p>
                      {benchmark.description && (
                        <p className="mt-1 text-xs text-ink-400 line-clamp-2">
                          {benchmark.description}
                        </p>
                      )}
                      {benchmark.requires_setup && benchmark.setup_notes && (
                        <p className="mt-1 text-xs text-yellow-400/90">
                          {benchmark.setup_notes}
                        </p>
                      )}
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Start Button */}
          {queueStats && queueStats.running >= workerSlots && !currentRun && (
            <div className="rounded-lg bg-ink-800/70 p-3 text-sm text-ink-300">
              All worker slots are busy ({queueStats.running}/{workerSlots}). New runs
              will queue until a slot is free.
            </div>
          )}

          <div className="flex justify-end pt-4">
            <Button
              onClick={startRun}
              disabled={!selectedModel || selectedBenchmarks.size === 0 || running}
              size="lg"
            >
              {running ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Run Benchmarks
                </>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Progress Section */}
      {currentRun && (
        <Card ref={progressRef}>
          <CardHeader>
            <CardTitle className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <span>Run Progress</span>
                {(progress.status === "succeeded" || progress.status === "failed") && (
                  <Button variant="outline" size="sm" asChild>
                    <Link href={`/runs/${currentRun.id}`}>
                      View Details
                    </Link>
                  </Button>
                )}
              </div>
              <span className={cn("text-sm", getStatusColor(progress.status || currentRun.status))}>
                {(progress.status || currentRun.status).toUpperCase()}
              </span>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {currentRun.status === "queued" && (
              <div className="rounded-lg bg-ink-800/60 p-3 text-sm text-ink-300">
                Run is queued
                {queueInfo ? ` (position ${queueInfo.queuePosition})` : ""}.{" "}
                {queueDelaySeconds !== null
                  ? `Estimated start in ~${formatDurationSeconds(queueDelaySeconds)}.`
                  : "Waiting for a free worker slot."}
              </div>
            )}
            {currentRun.status === "running" && (
              <div className="rounded-lg bg-ink-800/60 p-3 text-sm text-ink-300">
                {elapsedMs !== null
                  ? `Elapsed ${formatDuration(elapsedMs)}`
                  : "Elapsed -"}{" "}
                {etaSeconds !== null
                  ? `· ETA ~ ${formatDurationSeconds(etaSeconds)}`
                  : "· ETA estimating..."}
              </div>
            )}

            {/* Overall Progress */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-ink-300">Overall Progress</span>
                <span className="text-ink-100">{Math.round(progress.overall)}%</span>
              </div>
              <Progress value={progress.overall} className={running ? "progress-animate" : ""} />
            </div>

            {/* Per-Benchmark Progress */}
            <div className="space-y-4">
              {currentRun.benchmarks.map((rb) => {
                const bp = progress.benchmarks[rb.benchmark_name];
                const pct = bp && bp.total > 0 ? (bp.completed / bp.total) * 100 : 0;
                const benchmarkStatus = bp?.status || rb.status;

                return (
                  <div key={rb.id} className="space-y-2">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {benchmarkStatus === "succeeded" && (
                          <CheckCircle2 className="h-4 w-4 text-moss" />
                        )}
                        {benchmarkStatus === "failed" && (
                          <XCircle className="h-4 w-4 text-red-400" />
                        )}
                        {benchmarkStatus === "running" && (
                          <Loader2 className="h-4 w-4 animate-spin text-moss" />
                        )}
                        {!["succeeded", "failed", "running"].includes(benchmarkStatus) && (
                          <div className="h-4 w-4 rounded-full bg-ink-500" />
                        )}
                        <span className="font-medium">{rb.benchmark_name}</span>
                      </div>
                      <div className="flex items-center gap-4 text-sm">
                        {bp?.score !== undefined && bp.score !== null && (
                          <span className="text-moss">{formatPercent(bp.score)}</span>
                        )}
                        <span className="text-ink-400">
                          {bp?.completed || 0}/{bp?.total || 0}
                        </span>
                      </div>
                    </div>
                    <Progress value={pct} className="h-1" />
                  </div>
                );
              })}
            </div>

            {/* Event Log */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-ink-200">Log</h4>
              <div className="max-h-48 overflow-y-auto rounded-lg bg-ink-900 p-3 font-mono text-xs">
                {events.slice(-20).map((event) => (
                  <div key={event.id} className="text-ink-400">
                    <span className="text-ink-500">
                      {new Date(event.created_at).toLocaleTimeString()}
                    </span>{" "}
                    <span className="text-moss">[{event.event_type}]</span>{" "}
                    {event.message}
                  </div>
                ))}
                {events.length === 0 && (
                  <span className="text-ink-500">Waiting for events...</span>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
