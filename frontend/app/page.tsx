import { BenchmarkRunner } from "@/components/benchmark-runner";

export default function HomePage() {
  return (
    <div className="mx-auto max-w-screen-xl px-6 py-10">
      <div className="mb-10">
        <h1 className="text-4xl font-semibold text-ink-100">Run Benchmarks</h1>
        <p className="mt-2 text-lg text-ink-300">
          Evaluate Chutes-hosted LLMs across industry-standard benchmarks with
          one click
        </p>
      </div>
      <BenchmarkRunner />
    </div>
  );
}
















