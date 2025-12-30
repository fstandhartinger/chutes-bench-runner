import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8000";

function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="rounded-lg bg-ink-900 p-4 text-sm text-ink-200 overflow-x-auto">
      <code>{children}</code>
    </pre>
  );
}

export default function ApiDocsPage() {
  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-4xl font-semibold text-ink-100">API Usage</h1>
        <p className="mt-2 text-lg text-ink-300">
          Start runs with a bearer API key, track progress, and download signed results.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>1) Start a Run (Bearer API Key)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-ink-300">
          <p>
            Send your Chutes API key in the Authorization header. The run will execute using
            that key, not the system key.
          </p>
          <div className="space-y-2 text-sm text-ink-300">
            <p className="font-medium text-ink-200">Request fields</p>
            <ul className="space-y-1">
              <li>
                <span className="text-ink-100">model_id</span>: bench runner model UUID,
                Chutes <span className="text-ink-100">chute_id</span>, or the model slug/name
                (e.g. <span className="text-ink-100">zai-org/GLM-4.7-TEE</span>).
              </li>
              <li>
                <span className="text-ink-100">subset_pct</span>: integer 1–100 (common values:
                1, 5, 10, 25, 50, 100). Minimum sample size is 1 item.
              </li>
              <li>
                <span className="text-ink-100">selected_benchmarks</span>: optional list of
                benchmark names (use <span className="text-ink-100">/api/benchmarks</span> to
                discover valid values).
              </li>
            </ul>
          </div>
          <CodeBlock>{`curl -X POST ${BACKEND_URL}/api/runs/api \\
  -H "Authorization: Bearer <CHUTES_API_KEY>" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model_id": "<model-uuid | chute_id | model-slug>",
    "subset_pct": 1,
    "selected_benchmarks": ["mmlu_pro", "ifbench"]
  }'`}</CodeBlock>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2) Track Progress</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-ink-300">
          <p>Use the run ID returned by the create call.</p>
          <CodeBlock>{`# Poll run status
curl ${BACKEND_URL}/api/runs/<run-id>`}</CodeBlock>
          <CodeBlock>{`# Stream events (SSE)
curl -N ${BACKEND_URL}/api/runs/<run-id>/events`}</CodeBlock>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>3) Download Results</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-ink-300">
          <p>Exports are available after the run completes (succeeded or failed).</p>
          <CodeBlock>{`# CSV
curl -O ${BACKEND_URL}/api/runs/<run-id>/export?format=csv

# PDF
curl -O ${BACKEND_URL}/api/runs/<run-id>/export?format=pdf

# Signed ZIP (JSON + signature)
curl -O ${BACKEND_URL}/api/runs/<run-id>/export?format=zip`}</CodeBlock>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>4) Verify a Signed ZIP</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-ink-300">
          <p>Upload a signed zip to confirm it was produced by the official bench runner.</p>
          <CodeBlock>{`curl -X POST ${BACKEND_URL}/api/exports/verify \\
  -F "file=@benchmark_results.zip"`}</CodeBlock>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>5) Public Key</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-ink-300">
          <p>Fetch the public key for offline verification.</p>
          <CodeBlock>{`curl ${BACKEND_URL}/api/exports/public-key`}</CodeBlock>
        </CardContent>
      </Card>
    </div>
  );
}
