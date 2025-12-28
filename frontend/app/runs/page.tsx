"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { getRuns, type Run } from "@/lib/api";
import {
  formatDate,
  formatPercent,
  getStatusColor,
  getStatusBgColor,
  cn,
} from "@/lib/utils";
import { Loader2, ExternalLink, Download } from "lucide-react";

export default function RunsPage() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const data = await getRuns();
        setRuns(data.runs);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load runs");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

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

      {error && (
        <div className="mb-6 rounded-lg bg-red-500/10 p-4 text-red-400">
          {error}
        </div>
      )}

      {runs.length === 0 ? (
        <Card>
          <CardContent className="py-12 text-center">
            <p className="text-ink-400">No benchmark runs yet</p>
            <Button asChild className="mt-4">
              <Link href="/">Start Your First Run</Link>
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {runs.map((run) => (
            <Card key={run.id} className="overflow-hidden">
              <div className="flex items-center justify-between p-6">
                <div className="space-y-1">
                  <div className="flex items-center gap-3">
                    <h3 className="text-lg font-medium">{run.model_slug}</h3>
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
                    {run.subset_pct}% subset · {run.benchmarks.length} benchmarks ·{" "}
                    {formatDate(run.created_at)}
                  </p>
                </div>

                <div className="flex items-center gap-6">
                  {run.overall_score !== undefined && run.overall_score !== null && (
                    <div className="text-right">
                      <div className="text-2xl font-semibold text-moss">
                        {formatPercent(run.overall_score)}
                      </div>
                      <div className="text-xs text-ink-400">Overall Score</div>
                    </div>
                  )}

                  <div className="flex gap-2">
                    <Button variant="secondary" size="sm" asChild>
                      <Link href={`/runs/${run.id}`}>
                        <ExternalLink className="mr-1.5 h-3.5 w-3.5" />
                        View
                      </Link>
                    </Button>
                    {(run.status === "succeeded" || run.status === "failed") && (
                      <Button variant="outline" size="sm" asChild>
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
              <div className="flex gap-4 border-t border-ink-500 bg-ink-800/30 px-6 py-3">
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
          ))}
        </div>
      )}
    </div>
  );
}






