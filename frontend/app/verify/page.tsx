"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { verifySignedExport, getPublicKeyInfo, type SignedExportVerification, type PublicKeyInfo } from "@/lib/api";
import { cn } from "@/lib/utils";

export default function VerifyPage() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<SignedExportVerification | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [publicKey, setPublicKey] = useState<PublicKeyInfo | null>(null);

  useEffect(() => {
    async function loadKey() {
      try {
        const info = await getPublicKeyInfo();
        setPublicKey(info);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load public key");
      }
    }
    loadKey();
  }, []);

  const handleVerify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const verification = await verifySignedExport(file);
      setResult(verification);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10 space-y-8">
      <div>
        <h1 className="text-4xl font-semibold text-ink-100">Verify Results</h1>
        <p className="mt-2 text-lg text-ink-300">
          Upload a signed benchmark zip to confirm authenticity and integrity.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Upload Signed ZIP</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <input
            type="file"
            accept=".zip"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            className="block w-full text-sm text-ink-200"
          />
          <Button onClick={handleVerify} disabled={!file || loading}>
            {loading ? "Verifying..." : "Verify"}
          </Button>
        </CardContent>
      </Card>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-4 text-red-300">
          {error}
        </div>
      )}

      {result && (
        <Card className="border-ink-600">
          <CardHeader>
            <CardTitle>Verification Result</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className={cn(
                "rounded-lg border p-4 text-sm",
                result.valid ? "border-moss/40 bg-moss/10 text-moss" : "border-red-500/40 bg-red-500/10 text-red-300"
              )}
            >
              {result.valid ? "Signature valid. Results are authentic." : "Signature invalid or data tampered."}
            </div>

            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 text-sm text-ink-200">
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Run ID</div>
                <div className="font-mono break-all">{result.run_id || "-"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Model</div>
                <div>{result.model_slug || "-"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Subset</div>
                <div>
                  {result.subset_count
                    ? `${result.subset_count} items`
                    : result.subset_pct ?? "-"}
                </div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Subset Seed</div>
                <div>{result.subset_seed || "-"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Exported At</div>
                <div>{result.exported_at || "-"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Benchmarks</div>
                <div>{result.benchmark_count ?? "-"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Overall Score</div>
                <div>{result.overall_score ?? "-"}</div>
              </div>
            </div>

            <div className="grid gap-3 sm:grid-cols-2 text-sm text-ink-200">
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Signature</div>
                <div>{result.signature_valid ? "Valid" : "Invalid"}</div>
              </div>
              <div className="rounded bg-ink-900 p-3">
                <div className="text-xs text-ink-400">Hash Match</div>
                <div>{result.hash_match ? "Yes" : "No"}</div>
              </div>
            </div>

            {result.errors?.length > 0 && (
              <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300">
                <div className="font-medium mb-2">Errors</div>
                <ul className="list-disc pl-5 space-y-1">
                  {result.errors.map((err, idx) => (
                    <li key={`${err}-${idx}`}>{err}</li>
                  ))}
                </ul>
              </div>
            )}

            {publicKey && (
              <div className="rounded-lg bg-ink-900 p-3 text-sm text-ink-200">
                <div className="text-xs text-ink-400">Public Key Fingerprint</div>
                <div className="font-mono break-all">{publicKey.public_key_fingerprint}</div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
