"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  getRun,
  getBenchmarkDetails,
  getExportUrl,
  type Run,
  type ItemResult,
} from "@/lib/api";
import {
  formatDate,
  formatPercent,
  formatDuration,
  getStatusColor,
  getStatusBgColor,
  cn,
} from "@/lib/utils";
import {
  Loader2,
  ArrowLeft,
  Download,
  FileText,
  CheckCircle2,
  XCircle,
  AlertCircle,
} from "lucide-react";

export default function RunDetailPage() {
  const params = useParams();
  const runId = params.id as string;

  const [run, setRun] = useState<Run | null>(null);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);
  const [items, setItems] = useState<ItemResult[]>([]);
  const [loading, setLoading] = useState(true);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await getRun(runId);
        setRun(data);
        if (data.benchmarks.length > 0) {
          setSelectedBenchmark(data.benchmarks[0].benchmark_name);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load run");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [runId]);

  useEffect(() => {
    if (!selectedBenchmark) return;

    async function loadItems() {
      setItemsLoading(true);
      try {
        const data = await getBenchmarkDetails(runId, selectedBenchmark!, 50, 0);
        setItems(data.items.items);
      } catch (e) {
        console.error("Failed to load items:", e);
      } finally {
        setItemsLoading(false);
      }
    }
    loadItems();
  }, [runId, selectedBenchmark]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="h-8 w-8 animate-spin text-moss" />
      </div>
    );
  }

  if (error || !run) {
    return (
      <div className="mx-auto max-w-screen-xl px-6 py-10">
        <div className="rounded-lg bg-red-500/10 p-6 text-center text-red-400">
          <AlertCircle className="mx-auto mb-2 h-8 w-8" />
          <p>{error || "Run not found"}</p>
          <Button variant="secondary" asChild className="mt-4">
            <Link href="/runs">Back to Runs</Link>
          </Button>
        </div>
      </div>
    );
  }

  const selectedRb = run.benchmarks.find(
    (rb) => rb.benchmark_name === selectedBenchmark
  );

  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <Link
          href="/runs"
          className="mb-4 inline-flex items-center text-sm text-ink-400 hover:text-ink-200"
        >
          <ArrowLeft className="mr-1.5 h-4 w-4" />
          Back to Runs
        </Link>
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-semibold text-ink-100">
              {run.model_slug}
            </h1>
            <p className="mt-1 text-ink-400">
              Run ID: {run.id.slice(0, 8)}... · {formatDate(run.created_at)}
            </p>
          </div>
          <div className="flex items-center gap-3">
            <span
              className={cn(
                "rounded-full px-3 py-1 text-sm font-medium",
                getStatusBgColor(run.status),
                getStatusColor(run.status)
              )}
            >
              {run.status}
            </span>
            {run.overall_score !== undefined && run.overall_score !== null && (
              <div className="rounded-lg bg-moss/10 px-4 py-2 text-center">
                <div className="text-2xl font-semibold text-moss">
                  {formatPercent(run.overall_score)}
                </div>
                <div className="text-xs text-ink-400">Overall</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Summary Card */}
      <Card className="mb-8">
        <CardContent className="grid gap-6 p-6 sm:grid-cols-4">
          <div>
            <div className="text-sm text-ink-400">Subset</div>
            <div className="text-xl font-medium">{run.subset_pct}%</div>
          </div>
          <div>
            <div className="text-sm text-ink-400">Benchmarks</div>
            <div className="text-xl font-medium">{run.benchmarks.length}</div>
          </div>
          <div>
            <div className="text-sm text-ink-400">Started</div>
            <div className="text-xl font-medium">
              {run.started_at ? formatDate(run.started_at) : "-"}
            </div>
          </div>
          <div>
            <div className="text-sm text-ink-400">Completed</div>
            <div className="text-xl font-medium">
              {run.completed_at ? formatDate(run.completed_at) : "-"}
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Export Buttons */}
      {(run.status === "succeeded" || run.status === "failed") && (
        <div className="mb-8 flex gap-3">
          <Button variant="secondary" asChild>
            <a href={getExportUrl(run.id, "csv")} download>
              <Download className="mr-2 h-4 w-4" />
              Export CSV
            </a>
          </Button>
          <Button variant="secondary" asChild>
            <a href={getExportUrl(run.id, "pdf")} download>
              <FileText className="mr-2 h-4 w-4" />
              Export PDF Report
            </a>
          </Button>
        </div>
      )}

      {/* Benchmark Results */}
      <div className="grid gap-6 lg:grid-cols-[1fr_2fr]">
        {/* Benchmark List */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Benchmarks</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {run.benchmarks.map((rb) => (
              <button
                key={rb.id}
                onClick={() => setSelectedBenchmark(rb.benchmark_name)}
                className={cn(
                  "flex w-full items-center justify-between rounded-lg p-3 text-left transition-colors",
                  selectedBenchmark === rb.benchmark_name
                    ? "bg-moss/10 border border-moss/30"
                    : "bg-ink-700/50 hover:bg-ink-600/50"
                )}
              >
                <div className="flex items-center gap-2">
                  {rb.status === "succeeded" && (
                    <CheckCircle2 className="h-4 w-4 text-moss" />
                  )}
                  {rb.status === "failed" && (
                    <XCircle className="h-4 w-4 text-red-400" />
                  )}
                  {rb.status === "running" && (
                    <Loader2 className="h-4 w-4 animate-spin text-moss" />
                  )}
                  {!["succeeded", "failed", "running"].includes(rb.status) && (
                    <div className="h-4 w-4 rounded-full bg-ink-500" />
                  )}
                  <span className="font-medium">{rb.benchmark_name}</span>
                </div>
                {rb.score !== undefined && rb.score !== null && (
                  <span className="text-sm text-moss">
                    {formatPercent(rb.score)}
                  </span>
                )}
              </button>
            ))}
          </CardContent>
        </Card>

        {/* Benchmark Details */}
        <Card className="border-2 border-red-500">
          <CardHeader className="bg-red-500/20">
            <CardTitle className="text-lg">
              {selectedBenchmark || "Select a benchmark"} - DEBUG
            </CardTitle>
          </CardHeader>
          <CardContent className="bg-blue-500/20">
            {selectedRb ? (
              <div className="space-y-6">
                {/* Metrics */}
                <div className="grid gap-4 sm:grid-cols-3">
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Score</div>
                    <div className="text-2xl font-semibold text-moss">
                      {formatPercent(selectedRb.score)}
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Items</div>
                    <div className="text-2xl font-semibold">
                      {selectedRb.completed_items}/{selectedRb.sampled_items}
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Status</div>
                    <div
                      className={cn(
                        "text-2xl font-semibold capitalize",
                        getStatusColor(selectedRb.status)
                      )}
                    >
                      {selectedRb.status}
                    </div>
                  </div>
                </div>

                {/* Item Results */}
                <div>
                  <h4 className="mb-3 font-medium text-ink-200">
                    Item Results
                  </h4>
                  {itemsLoading ? (
                    <div className="flex justify-center py-8">
                      <Loader2 className="h-6 w-6 animate-spin text-moss" />
                    </div>
                  ) : items.length === 0 ? (
                    <p className="py-8 text-center text-ink-400">
                      No items evaluated yet
                    </p>
                  ) : (
                    <div className="max-h-96 space-y-2 overflow-y-auto rounded-lg bg-ink-900 p-4">
                      {items.map((item) => (
                        <div
                          key={item.id}
                          className="flex items-start gap-3 rounded border border-ink-600 p-3"
                        >
                          <div className="mt-0.5">
                            {item.is_correct === true && (
                              <CheckCircle2 className="h-4 w-4 text-moss" />
                            )}
                            {item.is_correct === false && (
                              <XCircle className="h-4 w-4 text-red-400" />
                            )}
                            {item.is_correct === null && (
                              <div className="h-4 w-4 rounded-full bg-ink-500" />
                            )}
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 text-sm">
                              <span className="font-mono text-ink-400">
                                {item.item_id}
                              </span>
                              {item.latency_ms && (
                                <span className="text-ink-500">
                                  {item.latency_ms}ms
                                </span>
                              )}
                            </div>
                            {item.response && (
                              <div className="mt-1 truncate text-sm text-ink-300">
                                {item.response.slice(0, 100)}
                                {item.response.length > 100 && "..."}
                              </div>
                            )}
                            {item.error && (
                              <div className="mt-1 text-sm text-red-400">
                                Error: {item.error}
                              </div>
                            )}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="py-8 text-center text-ink-400">
                Select a benchmark to view details
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

