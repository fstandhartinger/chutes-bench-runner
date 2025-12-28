/**
 * API client for Chutes Bench Runner backend
 */

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface Model {
  id: string;
  slug: string;
  name: string;
  tagline?: string;
  user?: string;
  logo?: string;
  instance_count: number;
  is_active: boolean;
}

export interface Benchmark {
  name: string;
  display_name: string;
  description?: string;
  is_enabled: boolean;
  supports_subset: boolean;
  requires_setup: boolean;
  setup_notes?: string;
  total_items: number;
}

export interface BenchmarkRunBenchmark {
  id: string;
  benchmark_name: string;
  status: string;
  total_items: number;
  completed_items: number;
  sampled_items: number;
  score?: number;
  metrics?: Record<string, unknown>;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
}

export interface Run {
  id: string;
  model_id: string;
  model_slug: string;
  subset_pct: number;
  status: string;
  selected_benchmarks?: string[];
  overall_score?: number;
  error_message?: string;
  started_at?: string;
  completed_at?: string;
  created_at: string;
  benchmarks: BenchmarkRunBenchmark[];
}

export interface RunEvent {
  id: string;
  event_type: string;
  benchmark_name?: string;
  message?: string;
  data?: Record<string, unknown>;
  created_at: string;
}

export interface ItemResult {
  id: string;
  item_id: string;
  prompt?: string;
  response?: string;
  expected?: string;
  is_correct?: boolean;
  score?: number;
  latency_ms?: number;
  error?: string;
  created_at: string;
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function getModels(search?: string): Promise<{ models: Model[]; total: number }> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  return fetchAPI(`/api/models?${params}`);
}

export async function getBenchmarks(): Promise<{ benchmarks: Benchmark[] }> {
  return fetchAPI("/api/benchmarks");
}

export async function getRuns(
  status?: string,
  limit = 50
): Promise<{ runs: Run[]; total: number }> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  params.set("limit", String(limit));
  return fetchAPI(`/api/runs?${params}`);
}

export async function getRun(runId: string): Promise<Run> {
  return fetchAPI(`/api/runs/${runId}`);
}

export async function createRun(
  modelId: string,
  subsetPct: number,
  selectedBenchmarks?: string[]
): Promise<Run> {
  return fetchAPI("/api/runs", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      subset_pct: subsetPct,
      selected_benchmarks: selectedBenchmarks,
    }),
  });
}

export async function cancelRun(runId: string): Promise<{ success: boolean; message: string }> {
  return fetchAPI(`/api/runs/${runId}/cancel`, { method: "POST" });
}

export async function getBenchmarkDetails(
  runId: string,
  benchmarkName: string,
  limit = 100,
  offset = 0
): Promise<{
  benchmark: BenchmarkRunBenchmark;
  items: { items: ItemResult[]; total: number };
}> {
  return fetchAPI(`/api/runs/${runId}/benchmarks/${benchmarkName}?limit=${limit}&offset=${offset}`);
}

export function getExportUrl(runId: string, format: "csv" | "pdf"): string {
  return `${API_BASE}/api/runs/${runId}/export?format=${format}`;
}

export function createEventSource(runId: string): EventSource {
  return new EventSource(`${API_BASE}/api/runs/${runId}/events`);
}


