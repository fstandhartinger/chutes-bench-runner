"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  getRun,
  getBenchmarkDetails,
  getExportUrl,
  type Run,
  type ItemResult,
  type BenchmarkRunBenchmark,
} from "@/lib/api";
import {
  formatDate,
  formatPercent,
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
  ChevronDown,
  ChevronUp,
  Clock,
  Zap,
  Hash,
  MessageSquare,
  Target,
  AlertTriangle,
  Info,
  Code,
  Terminal,
  Package,
} from "lucide-react";

// Expandable Item Component
function ItemDetailCard({ item, index }: { item: ItemResult; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={cn(
        "rounded-lg border transition-all",
        item.is_correct === true
          ? "border-moss/30 bg-moss/5"
          : item.is_correct === false
          ? "border-red-400/30 bg-red-400/5"
          : item.error
          ? "border-yellow-400/30 bg-yellow-400/5"
          : "border-ink-600 bg-ink-800/50"
      )}
    >
      {/* Header - Always visible */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center justify-between p-4 text-left"
      >
        <div className="flex items-center gap-3">
          {/* Status Icon */}
          <div className="flex-shrink-0">
            {item.is_correct === true && (
              <CheckCircle2 className="h-5 w-5 text-moss" />
            )}
            {item.is_correct === false && (
              <XCircle className="h-5 w-5 text-red-400" />
            )}
            {item.is_correct === null && item.error && (
              <AlertTriangle className="h-5 w-5 text-yellow-400" />
            )}
            {item.is_correct === null && !item.error && (
              <div className="h-5 w-5 rounded-full bg-ink-500" />
            )}
          </div>

          {/* Item ID and Quick Stats */}
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-sm font-medium text-ink-200">
                #{index + 1} · {item.item_id}
              </span>
              {item.item_metadata?.category !== undefined && (
                <span className="rounded bg-ink-700 px-2 py-0.5 text-xs text-ink-400">
                  {String(item.item_metadata.category)}
                </span>
              )}
            </div>
            <div className="mt-1 flex items-center gap-3 text-xs text-ink-400">
              {item.latency_ms !== undefined && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {item.latency_ms}ms
                </span>
              )}
              {(item.input_tokens || item.output_tokens) && (
                <span className="flex items-center gap-1">
                  <Zap className="h-3 w-3" />
                  {item.input_tokens || 0} → {item.output_tokens || 0} tokens
                </span>
              )}
              {item.score !== undefined && item.score !== null && (
                <span className="flex items-center gap-1">
                  <Target className="h-3 w-3" />
                  Score: {item.score.toFixed(2)}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Expand/Collapse Icon */}
        <div className="flex items-center gap-2">
          {item.is_correct !== null && (
            <span
              className={cn(
                "text-sm font-medium",
                item.is_correct ? "text-moss" : "text-red-400"
              )}
            >
              {item.is_correct ? "Correct" : "Incorrect"}
            </span>
          )}
          {expanded ? (
            <ChevronUp className="h-5 w-5 text-ink-400" />
          ) : (
            <ChevronDown className="h-5 w-5 text-ink-400" />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-ink-700 p-4 space-y-4">
          {/* Error Message */}
          {item.error && (
            <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-3">
              <div className="flex items-center gap-2 text-red-400 font-medium text-sm mb-1">
                <AlertTriangle className="h-4 w-4" />
                Error
              </div>
              <p className="text-sm text-red-300 font-mono">{item.error}</p>
            </div>
          )}

          {/* Task/Prompt */}
          {item.prompt && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <MessageSquare className="h-4 w-4" />
                Task / Prompt
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                {item.prompt}
              </div>
            </div>
          )}

          {/* Model Response */}
          {(item.response !== undefined || item.error) && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Zap className="h-4 w-4" />
                Model Response
              </div>
              <div
                className={cn(
                  "rounded-lg p-3 text-sm font-mono whitespace-pre-wrap max-h-64 overflow-y-auto",
                  item.is_correct === true
                    ? "bg-moss/10 text-moss"
                    : item.is_correct === false
                    ? "bg-red-500/10 text-red-300"
                    : "bg-ink-900 text-ink-200"
                )}
              >
                {item.response && item.response.length > 0 ? item.response : "No response captured."}
              </div>
            </div>
          )}

          {/* Expected Answer */}
          {item.expected && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Target className="h-4 w-4" />
                Expected Answer
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-moss font-mono">
                {item.expected}
              </div>
            </div>
          )}

          {/* System Prompt */}
          {item.item_metadata?.system_prompt && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Info className="h-4 w-4" />
                System Prompt
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                {String(item.item_metadata.system_prompt)}
              </div>
            </div>
          )}

          {/* Metadata */}
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded bg-ink-800 p-2">
              <div className="text-xs text-ink-400">Latency</div>
              <div className="text-sm font-medium">
                {item.latency_ms ? `${item.latency_ms}ms` : "-"}
              </div>
            </div>
            <div className="rounded bg-ink-800 p-2">
              <div className="text-xs text-ink-400">Input Tokens</div>
              <div className="text-sm font-medium">
                {item.input_tokens ?? "-"}
              </div>
            </div>
            <div className="rounded bg-ink-800 p-2">
              <div className="text-xs text-ink-400">Output Tokens</div>
              <div className="text-sm font-medium">
                {item.output_tokens ?? "-"}
              </div>
            </div>
            <div className="rounded bg-ink-800 p-2">
              <div className="text-xs text-ink-400">Item Hash</div>
              <div className="text-sm font-medium font-mono truncate">
                {item.item_hash?.slice(0, 8) || "-"}
              </div>
            </div>
          </div>

          {/* Test Code if present */}
          {item.test_code && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Terminal className="h-4 w-4" />
                Test Code (Validation)
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200 font-mono whitespace-pre-wrap max-h-64 overflow-y-auto">
                {item.test_code}
              </div>
            </div>
          )}

          {/* Judge Output if present */}
          {item.judge_output && Object.keys(item.judge_output).length > 0 && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Info className="h-4 w-4" />
                Judge Output
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200 font-mono whitespace-pre-wrap">
                {JSON.stringify(item.judge_output, null, 2)}
              </div>
            </div>
          )}

          {/* Additional Metadata */}
          {item.item_metadata && Object.keys(item.item_metadata).length > 0 && (
            <div>
              <div className="flex items-center gap-2 text-sm font-medium text-ink-300 mb-2">
                <Hash className="h-4 w-4" />
                Item Metadata
              </div>
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200 font-mono whitespace-pre-wrap">
                {JSON.stringify(item.item_metadata, null, 2)}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// Error/Failure Info Component
function FailureInfo({
  run,
  benchmark,
}: {
  run: Run;
  benchmark?: BenchmarkRunBenchmark;
}) {
  const errorMessage = benchmark?.error_message || run.error_message;

  if (!errorMessage && run.status !== "failed" && benchmark?.status !== "failed") {
    return null;
  }

  return (
    <div className="rounded-lg bg-red-500/10 border border-red-500/30 p-4 mb-4">
      <div className="flex items-center gap-2 text-red-400 font-medium mb-2">
        <AlertTriangle className="h-5 w-5" />
        {benchmark ? `Benchmark Failed: ${benchmark.benchmark_name}` : "Run Failed"}
      </div>
      {errorMessage && (
        <p className="text-sm text-red-300">{errorMessage}</p>
      )}
      {!errorMessage && (
        <p className="text-sm text-red-300/70">
          No detailed error message available. Check the worker logs for more information.
        </p>
      )}
    </div>
  );
}

export default function RunDetailPage() {
  const params = useParams();
  const runId = params.id as string;

  const [run, setRun] = useState<Run | null>(null);
  const [selectedBenchmark, setSelectedBenchmark] = useState<string | null>(null);
  const [items, setItems] = useState<ItemResult[]>([]);
  const [totalItems, setTotalItems] = useState(0);
  const [loading, setLoading] = useState(true);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [itemsLimit, setItemsLimit] = useState(20);
  const [itemsMode, setItemsMode] = useState<"paged" | "all">("paged");
  const [showAllLoading, setShowAllLoading] = useState(false);
  const [exportingMarkdown, setExportingMarkdown] = useState(false);

  const fetchAllItems = useCallback(async () => {
    if (!selectedBenchmark) return [];
    const pageSize = 200;
    let offset = 0;
    let total = totalItems;
    const allItems: ItemResult[] = [];

    while (offset < total || total === 0) {
      const data = await getBenchmarkDetails(runId, selectedBenchmark, pageSize, offset);
      const batch = data.items.items;
      total = data.items.total;
      if (batch.length === 0) break;
      allItems.push(...batch);
      offset += batch.length;
    }

    return allItems;
  }, [runId, selectedBenchmark, totalItems]);

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
    setItemsMode("paged");
    setItemsLimit(20);
    setItems([]);
    setTotalItems(0);
  }, [selectedBenchmark]);

  useEffect(() => {
    if (!selectedBenchmark || itemsMode === "all") return;

    async function loadItems() {
      setItemsLoading(true);
      try {
        const data = await getBenchmarkDetails(runId, selectedBenchmark!, itemsLimit, 0);
        setItems(data.items.items);
        setTotalItems(data.items.total);
      } catch (e) {
        console.error("Failed to load items:", e);
      } finally {
        setItemsLoading(false);
      }
    }
    loadItems();
  }, [runId, selectedBenchmark, itemsLimit, itemsMode]);

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

  // Calculate stats
  const correctCount = items.filter((i) => i.is_correct === true).length;
  const incorrectCount = items.filter((i) => i.is_correct === false).length;
  const errorCount = items.filter((i) => i.error).length;
  const avgLatency =
    items.length > 0
      ? Math.round(
          items.reduce((sum, i) => sum + (i.latency_ms || 0), 0) / items.length
        )
      : 0;
  const totalTokens = items.reduce(
    (sum, i) => sum + (i.input_tokens || 0) + (i.output_tokens || 0),
    0
  );
  const sampledPct =
    selectedRb && selectedRb.total_items > 0
      ? (selectedRb.sampled_items / selectedRb.total_items) * 100
      : 0;

  const handleShowAll = async () => {
    if (!selectedBenchmark || items.length >= totalItems) return;
    setShowAllLoading(true);
    setItemsMode("all");
    try {
      const allItems = await fetchAllItems();
      setItems(allItems);
      setTotalItems(allItems.length || totalItems);
    } catch (e) {
      console.error("Failed to load all items:", e);
    } finally {
      setShowAllLoading(false);
    }
  };

  const exportMarkdown = async () => {
    if (!selectedBenchmark || !selectedRb) return;
    setExportingMarkdown(true);
    try {
      const exportItems =
        items.length === totalItems && items.length > 0 ? items : await fetchAllItems();
      const lines: string[] = [];

      lines.push(`# Benchmark Protocol`);
      lines.push(``);
      lines.push(`- Run ID: ${run.id}`);
      lines.push(`- Model: ${run.model_slug}`);
      lines.push(`- Benchmark: ${selectedBenchmark}`);
      lines.push(`- Subset: ${run.subset_pct}%`);
      lines.push(`- Sampled Items: ${selectedRb.sampled_items}/${selectedRb.total_items}`);
      lines.push(`- Status: ${selectedRb.status}`);
      lines.push(`- Generated: ${new Date().toISOString()}`);
      lines.push(``);

      exportItems.forEach((item, index) => {
        lines.push(`## Item ${index + 1}`);
        lines.push(`- Item ID: ${item.item_id}`);
        lines.push(`- Item Hash: ${item.item_hash || "-"}`);
        lines.push(`- Correct: ${item.is_correct === true ? "true" : item.is_correct === false ? "false" : "-"}`);
        lines.push(`- Score: ${item.score ?? "-"}`);
        lines.push(`- Error: ${item.error || "-"}`);
        lines.push(`- Latency: ${item.latency_ms ?? "-"} ms`);
        lines.push(`- Input Tokens: ${item.input_tokens ?? "-"}`);
        lines.push(`- Output Tokens: ${item.output_tokens ?? "-"}`);
        lines.push(``);

        lines.push(`### Prompt`);
        lines.push("```text");
        lines.push(item.prompt || "");
        lines.push("```");
        lines.push(``);

        lines.push(`### Response`);
        lines.push("```text");
        lines.push(item.response || "");
        lines.push("```");
        lines.push(``);

        if (item.expected) {
          lines.push(`### Expected`);
          lines.push("```text");
          lines.push(item.expected);
          lines.push("```");
          lines.push(``);
        }

        if (item.item_metadata?.system_prompt) {
          lines.push(`### System Prompt`);
          lines.push("```text");
          lines.push(String(item.item_metadata.system_prompt));
          lines.push("```");
          lines.push(``);
        }

        if (item.test_code) {
          lines.push(`### Test Code`);
          lines.push("```text");
          lines.push(item.test_code);
          lines.push("```");
          lines.push(``);
        }

        if (item.judge_output) {
          lines.push(`### Judge Output`);
          lines.push("```json");
          lines.push(JSON.stringify(item.judge_output, null, 2));
          lines.push("```");
          lines.push(``);
        }

        if (item.item_metadata && Object.keys(item.item_metadata).length > 0) {
          lines.push(`### Item Metadata`);
          lines.push("```json");
          lines.push(JSON.stringify(item.item_metadata, null, 2));
          lines.push("```");
          lines.push(``);
        }
      });

      const blob = new Blob([lines.join("\n")], { type: "text/markdown" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `benchmark_protocol_${run.id}_${selectedBenchmark}.md`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (e) {
      console.error("Failed to export markdown:", e);
    } finally {
      setExportingMarkdown(false);
    }
  };

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

      {/* Run-level Error Message */}
      {(run.status === "failed" || run.error_message) && (
        <FailureInfo run={run} />
      )}

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
          <Button variant="secondary" asChild>
            <a href={getExportUrl(run.id, "zip")} download>
              <Package className="mr-2 h-4 w-4" />
              Signed ZIP
            </a>
          </Button>
        </div>
      )}

      {/* Benchmark Results Section */}
      <div className="space-y-6">
        {/* Benchmark List Card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Benchmarks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-2">
              {run.benchmarks.map((rb) => (
                <button
                  key={rb.id}
                  onClick={() => setSelectedBenchmark(rb.benchmark_name)}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-4 py-2 transition-colors",
                    selectedBenchmark === rb.benchmark_name
                      ? "bg-moss/20 border border-moss/40 text-moss"
                      : "bg-ink-700/50 hover:bg-ink-600/50"
                  )}
                >
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
                  {rb.score !== undefined && rb.score !== null && (
                    <span className="text-sm opacity-80">
                      {formatPercent(rb.score)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Benchmark Details Card */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              {selectedBenchmark
                ? `${selectedBenchmark} Details`
                : "Select a benchmark"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {selectedRb ? (
              <div className="space-y-6">
                {/* Benchmark-level Error */}
                {selectedRb.status === "failed" && (
                  <FailureInfo run={run} benchmark={selectedRb} />
                )}

                {/* Metrics */}
                <div className="grid gap-4 sm:grid-cols-3 lg:grid-cols-6">
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Score</div>
                    <div className="text-2xl font-semibold text-moss">
                      {formatPercent(selectedRb.score)}
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Items</div>
                    <div className="text-2xl font-semibold">
                      {selectedRb.completed_items} / {selectedRb.sampled_items}
                    </div>
                    <div className="mt-1 text-xs text-ink-400">
                      Sampled {selectedRb.sampled_items} of {selectedRb.total_items} ({sampledPct.toFixed(2)}%)
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Correct</div>
                    <div className="text-2xl font-semibold text-moss">
                      {correctCount}
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Incorrect</div>
                    <div className="text-2xl font-semibold text-red-400">
                      {incorrectCount}
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Avg Latency</div>
                    <div className="text-2xl font-semibold">
                      {avgLatency}ms
                    </div>
                  </div>
                  <div className="rounded-lg bg-ink-700/50 p-4">
                    <div className="text-sm text-ink-400">Total Tokens</div>
                    <div className="text-2xl font-semibold">
                      {totalTokens.toLocaleString()}
                    </div>
                  </div>
                </div>

                {/* Benchmark Metrics if available */}
                {selectedRb.metrics &&
                  Object.keys(selectedRb.metrics).length > 0 && (
                    <div className="rounded-lg bg-ink-800 p-4">
                      <h4 className="font-medium text-ink-200 mb-2">
                        Benchmark Metrics
                      </h4>
                      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
                        {Object.entries(selectedRb.metrics).map(
                          ([key, value]) => (
                            <div key={key} className="rounded bg-ink-700 p-2">
                              <div className="text-xs text-ink-400 capitalize">
                                {key.replace(/_/g, " ")}
                              </div>
                              <div className="text-sm font-medium">
                                {typeof value === "number"
                                  ? value.toLocaleString(undefined, {
                                      maximumFractionDigits: 4,
                                    })
                                  : String(value)}
                              </div>
                            </div>
                          )
                        )}
                      </div>
                    </div>
                  )}

                {/* Item Results */}
                <div>
                  <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
                    <div>
                      <h4 className="font-medium text-ink-200">
                        Item Results ({totalItems} total
                        {items.length < totalItems
                          ? `, showing ${items.length}`
                          : ""}
                        )
                      </h4>
                      {errorCount > 0 && (
                        <span className="text-sm text-yellow-400">
                          {errorCount} items with errors
                        </span>
                      )}
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={handleShowAll}
                        disabled={
                          itemsLoading ||
                          showAllLoading ||
                          items.length >= totalItems
                        }
                      >
                        {showAllLoading ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : null}
                        Show All
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        onClick={exportMarkdown}
                        disabled={exportingMarkdown || itemsLoading}
                      >
                        {exportingMarkdown ? (
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : null}
                        Export Protocol (MD)
                      </Button>
                    </div>
                  </div>
                  {itemsLoading ? (
                    <div className="flex justify-center py-8">
                      <Loader2 className="h-6 w-6 animate-spin text-moss" />
                    </div>
                  ) : items.length === 0 ? (
                    <p className="py-8 text-center text-ink-400">
                      No items evaluated yet
                    </p>
                  ) : (
                    <div className="space-y-3">
                      {items.map((item, index) => (
                        <ItemDetailCard
                          key={item.id}
                          item={item}
                          index={index}
                        />
                      ))}

                      {/* Load More Button */}
                      {itemsMode === "paged" && items.length < totalItems && (
                        <div className="text-center pt-4">
                          <Button
                            variant="secondary"
                            onClick={() => setItemsLimit((prev) => prev + 20)}
                            disabled={itemsLoading}
                          >
                            {itemsLoading ? (
                              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                            ) : null}
                            Load More ({totalItems - items.length} remaining)
                          </Button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <p className="py-8 text-center text-ink-400">
                Select a benchmark above to view detailed results
              </p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
