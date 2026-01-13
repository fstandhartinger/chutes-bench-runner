/**
 * API client for Chutes Bench Runner backend
 */

const API_BASE = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

export interface Model {
  id: string;
  slug: string;
  name: string;
  provider: string;
  tagline?: string;
  user?: string;
  logo?: string;
  chute_id?: string;
  instance_count: number;
  is_active: boolean;
}

export interface Benchmark {
  name: string;
  display_name: string;
  description?: string;
  category?: string;
  is_enabled: boolean;
  supports_subset: boolean;
  requires_setup: boolean;
  setup_notes?: string;
  default_selected?: boolean;
  total_items: number;
  avg_item_latency_ms?: number;
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
  provider: string;
  subset_pct: number;
  subset_count?: number | null;
  subset_seed?: string | null;
  status: string;
  selected_benchmarks?: string[];
  overall_score?: number;
  error_message?: string;
  provider_metadata?: Record<string, unknown> | null;
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
  item_hash?: string;
  prompt?: string;
  response?: string;
  expected?: string;
  is_correct?: boolean;
  score?: number;
  judge_output?: Record<string, unknown>;
  latency_ms?: number;
  input_tokens?: number;
  output_tokens?: number;
  error?: string;
  item_metadata?: Record<string, unknown>;
  test_code?: string;
  created_at: string;
}

export interface BenchmarkSummary {
  total_items: number;
  correct: number;
  incorrect: number;
  errors: number;
  avg_latency_ms?: number | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  input_cost_usd?: number | null;
  output_cost_usd?: number | null;
  total_cost_usd?: number | null;
  pricing_input_per_million_usd?: number | null;
  pricing_output_per_million_usd?: number | null;
  error_breakdown?: { message: string; count: number }[];
}

export interface SignedExportVerification {
  valid: boolean;
  signature_valid: boolean;
  hash_match: boolean;
  errors: string[];
  run_id?: string;
  model_slug?: string;
  subset_pct?: number;
  subset_count?: number;
  subset_seed?: string;
  overall_score?: number;
  exported_at?: string;
  benchmark_count?: number;
  public_key_fingerprint?: string;
}

export interface PublicKeyInfo {
  algorithm: string;
  public_key: string;
  public_key_fingerprint: string;
}

export interface MaintenanceStatus {
  maintenance_mode: boolean;
  message: string;
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

export async function getModels(
  search?: string,
  provider?: string
): Promise<{ models: Model[]; total: number }> {
  const params = new URLSearchParams();
  if (search) params.set("search", search);
  if (provider) params.set("provider", provider);
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
  selectedBenchmarks?: string[],
  subsetCount?: number | null,
  subsetSeed?: string | null,
  provider?: string
): Promise<Run> {
  return fetchAPI("/api/runs", {
    method: "POST",
    body: JSON.stringify({
      model_id: modelId,
      subset_pct: subsetPct,
      subset_count: subsetCount ?? undefined,
      subset_seed: subsetSeed ?? undefined,
      selected_benchmarks: selectedBenchmarks,
      provider,
    }),
  });
}

export interface WorkerHeartbeatInfo {
  worker_id: string;
  hostname?: string | null;
  running_runs: number;
  max_concurrent_runs: number;
  item_concurrency: number;
  last_seen: string;
}

export interface WorkerTimeseriesPoint {
  timestamp: string;
  worker_count: number;
  running_runs: number;
  queued_runs: number;
}

export interface RunSummary {
  id: string;
  model_slug: string;
  provider: string;
  subset_pct: number;
  subset_count?: number | null;
  subset_seed?: string | null;
  status: string;
  created_at: string;
  started_at?: string | null;
  completed_at?: string | null;
  benchmarks?: string[] | null;
}

export interface TokenUsageWindow {
  window_hours: number;
  input_tokens: number;
  output_tokens: number;
}

export interface TokenUsageStats {
  last_24h: TokenUsageWindow;
  last_7d: TokenUsageWindow;
}

export interface SandyMetricsPoint {
  timestamp: string;
  cpu_ratio?: number | null;
  memory_ratio?: number | null;
  disk_ratio?: number | null;
}

export interface SandySandboxStats {
  sandbox_id: string;
  container_id?: string | null;
  cpu_ratio?: number | null;
  cpu_cores_used?: number | null;
  cpu_cores_total?: number | null;
  memory_usage_bytes?: number | null;
  memory_limit_bytes?: number | null;
  memory_ratio?: number | null;
  disk_bytes?: number | null;
  updated_at?: string | null;
}

export interface SandyResourcesResponse {
  canCreateSandbox: boolean;
  rejectReason?: string | null;
  limits: Record<string, number | null>;
  priorityBreakdown: Record<string, number>;
  cpu_percent?: number | null;
  memory_percent?: number | null;
  memory_total_gb?: number | null;
  memory_available_gb?: number | null;
  disk_used_ratio?: number | null;
  cpuCount?: number | null;
  load1?: number | null;
  load5?: number | null;
  load15?: number | null;
  memoryTotalBytes?: number | null;
  memoryUsedBytes?: number | null;
  memoryAvailableBytes?: number | null;
  diskTotalBytes?: number | null;
  diskUsedBytes?: number | null;
  diskFreeBytes?: number | null;
  diskUsedRatio?: number | null;
}

export interface OpsOverview {
  workers: WorkerHeartbeatInfo[];
  timeseries: WorkerTimeseriesPoint[];
  queue_counts: Record<string, number>;
  queued_runs: RunSummary[];
  running_runs: RunSummary[];
  completed_runs: RunSummary[];
  worker_config: {
    worker_max_concurrent: number;
    worker_item_concurrency: number;
  };
  token_stats?: TokenUsageStats | null;
}

export async function getOpsOverview(minutes?: number): Promise<OpsOverview> {
  const params = new URLSearchParams();
  if (minutes) params.set("minutes", String(minutes));
  const query = params.toString();
  return fetchAPI(`/api/ops/overview${query ? `?${query}` : ""}`);
}

export async function getSandyMetrics(hours = 12): Promise<SandyMetricsPoint[]> {
  return fetchAPI(`/api/ops/sandy/metrics?hours=${hours}`);
}

export async function getSandyResources(): Promise<SandyResourcesResponse> {
  return fetchAPI("/api/ops/sandy/resources");
}

export async function getSandySandboxStats(ids?: string[]): Promise<SandySandboxStats[]> {
  const params = new URLSearchParams();
  if (ids && ids.length > 0) {
    params.set("ids", ids.join(","));
  }
  const query = params.toString();
  return fetchAPI(`/api/ops/sandy/sandboxes${query ? `?${query}` : ""}`);
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
  summary: BenchmarkSummary;
}> {
  return fetchAPI(`/api/runs/${runId}/benchmarks/${benchmarkName}?limit=${limit}&offset=${offset}`);
}

export function getExportUrl(runId: string, format: "csv" | "pdf" | "zip"): string {
  return `${API_BASE}/api/runs/${runId}/export?format=${format}`;
}

export function createEventSource(runId: string, lastEventId?: string): EventSource {
  const url = new URL(`${API_BASE}/api/runs/${runId}/events`);
  if (lastEventId) {
    url.searchParams.set("Last-Event-ID", lastEventId);
  }
  return new EventSource(url.toString(), { withCredentials: true });
}

export async function verifySignedExport(file: File): Promise<SignedExportVerification> {
  const url = `${API_BASE}/api/exports/verify`;
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Request failed" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

export async function getPublicKeyInfo(): Promise<PublicKeyInfo> {
  return fetchAPI("/api/exports/public-key");
}

export async function getServiceStatus(): Promise<MaintenanceStatus> {
  return fetchAPI("/api/status");
}
