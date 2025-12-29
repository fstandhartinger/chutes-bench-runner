'use client';

import Link from 'next/link';
import { useAuth } from '@/contexts/auth-context';

export function Header() {
  const { isLoading, isAuthenticated, idpConfigured, user, hasInvokeScope, login, logout } = useAuth();

  return (
    <header className="sticky top-0 z-50 border-b border-ink-500 bg-ink-900/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-screen-xl items-center justify-between px-6">
        <Link href="/" className="flex items-center gap-3">
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
        </Link>
        
        <div className="flex items-center gap-6">
          <nav className="flex items-center gap-6">
            <Link
              href="/"
              className="text-sm text-ink-300 transition-colors hover:text-ink-100"
            >
              Run Benchmarks
            </Link>
            <Link
              href="/runs"
              className="text-sm text-ink-300 transition-colors hover:text-ink-100"
            >
              History
            </Link>
            <Link
              href="/api-docs"
              className="text-sm text-ink-300 transition-colors hover:text-ink-100"
            >
              API
            </Link>
            <Link
              href="/verify"
              className="text-sm text-ink-300 transition-colors hover:text-ink-100"
            >
              Verify
            </Link>
          </nav>
          
          <div className="h-6 w-px bg-ink-500" />
          
          {isLoading ? (
            <div className="h-8 w-20 animate-pulse rounded bg-ink-700" />
          ) : isAuthenticated && user ? (
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2">
                <div className="flex h-8 w-8 items-center justify-center rounded-full bg-moss/20 text-moss text-sm font-medium">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <div className="flex flex-col">
                  <span className="text-sm font-medium text-ink-100">
                    {user.username}
                  </span>
                  {hasInvokeScope && (
                    <span className="text-xs text-moss">Using your Chutes account</span>
                  )}
                </div>
              </div>
              <button
                onClick={() => logout()}
                className="text-sm text-ink-400 hover:text-ink-100 transition-colors"
              >
                Sign out
              </button>
            </div>
          ) : idpConfigured ? (
            <button
              onClick={() => login(window.location.pathname)}
              className="flex items-center gap-2 rounded-lg bg-moss/10 px-4 py-2 text-sm font-medium text-moss transition-colors hover:bg-moss/20"
            >
              <svg
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
              Sign in with Chutes
            </button>
          ) : (
            <span className="text-sm text-ink-400">Using API Key</span>
          )}
        </div>
      </div>
    </header>
  );
}






