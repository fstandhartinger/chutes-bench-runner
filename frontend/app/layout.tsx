import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Chutes Bench Runner",
  description: "Run LLM benchmarks against Chutes-hosted models with one click",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-ink-900 text-ink-100 antialiased">
        <div className="flex min-h-screen flex-col">
          <header className="sticky top-0 z-50 border-b border-ink-500 bg-ink-900/80 backdrop-blur-xl">
            <div className="mx-auto flex h-16 max-w-screen-xl items-center justify-between px-6">
              <a href="/" className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-moss/10">
                  <svg
                    className="h-5 w-5 text-moss"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
                <span className="text-xl font-semibold">Bench Runner</span>
              </a>
              <nav className="flex items-center gap-6">
                <a
                  href="/"
                  className="text-sm text-ink-300 transition-colors hover:text-ink-100"
                >
                  Run Benchmarks
                </a>
                <a
                  href="/runs"
                  className="text-sm text-ink-300 transition-colors hover:text-ink-100"
                >
                  History
                </a>
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="border-t border-ink-500 py-6">
            <div className="mx-auto max-w-screen-xl px-6 text-center text-sm text-ink-400">
              Chutes Bench Runner · Powered by{" "}
              <a
                href="https://chutes.ai"
                className="text-moss hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                Chutes
              </a>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}

