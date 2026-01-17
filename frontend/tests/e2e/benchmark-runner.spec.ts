import { test, expect } from "@playwright/test";

const nowIso = new Date().toISOString();

const stubModel = {
  id: "model-1",
  slug: "stub-model",
  name: "Stub Model",
  provider: "chutes",
  instance_count: 1,
  is_active: true,
};

const stubBenchmarks = [
  {
    name: "mmlu",
    display_name: "MMLU",
    description: "General knowledge",
    category: "Core Benchmarks",
    is_enabled: true,
    supports_subset: true,
    requires_setup: false,
    total_items: 50,
    default_selected: true,
  },
];

const stubOpsOverview = {
  workers: [
    {
      worker_id: "worker-1",
      hostname: "test-worker",
      running_runs: 0,
      max_concurrent_runs: 2,
      item_concurrency: 2,
      last_seen: nowIso,
    },
  ],
  timeseries: [],
  queue_counts: {
    queued: 0,
    running: 0,
    succeeded: 0,
    failed: 0,
    canceled: 0,
  },
  queued_runs: [],
  running_runs: [],
  completed_runs: [],
  worker_config: {
    worker_max_concurrent: 2,
    worker_item_concurrency: 2,
  },
  token_stats: null,
};

test.beforeEach(async ({ page }) => {
  await page.route("**/api/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();

    if (path === "/api/models") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ models: [stubModel], total: 1 }),
      });
    }

    if (path === "/api/benchmarks") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ benchmarks: stubBenchmarks }),
      });
    }

    if (path === "/api/runs") {
      if (method === "POST") {
        return route.fulfill({
          status: 200,
          contentType: "application/json",
          body: JSON.stringify({
            id: "run-1",
            model_id: stubModel.id,
            model_slug: stubModel.slug,
            provider: stubModel.provider,
            subset_pct: 10,
            status: "queued",
            created_at: nowIso,
            benchmarks: [],
          }),
        });
      }
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ runs: [], total: 0 }),
      });
    }

    if (path === "/api/ops/overview") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify(stubOpsOverview),
      });
    }

    if (path === "/api/status") {
      return route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ maintenance_mode: false, message: "" }),
      });
    }

    return route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({}),
    });
  });
});

test("loads the benchmark runner with stubbed data", async ({ page }) => {
  const consoleErrors: string[] = [];
  const requestFailures: string[] = [];

  page.on("console", (msg) => {
    if (msg.type() === "error") {
      consoleErrors.push(msg.text());
    }
  });
  page.on("requestfailed", (request) => {
    requestFailures.push(request.url());
  });

  await page.goto("/");
  await page.waitForLoadState("networkidle");

  await expect(page.getByRole("heading", { name: "Run Benchmarks" })).toBeVisible();
  await expect(page.getByText("Configure Benchmark Run")).toBeVisible();

  await expect(page.getByText("Stub Model")).toBeVisible();
  await expect(page.getByText("No models available for this provider.")).toHaveCount(0);

  expect(consoleErrors).toEqual([]);
  expect(requestFailures).toEqual([]);
});
