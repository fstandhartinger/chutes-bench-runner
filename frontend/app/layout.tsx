import type { Metadata } from "next";
import "./globals.css";
import { AuthProvider } from "@/contexts/auth-context";
import { Header } from "@/components/header";

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
        <AuthProvider>
          <div className="flex min-h-screen flex-col">
            <Header />
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
        </AuthProvider>
      </body>
    </html>
  );
}

