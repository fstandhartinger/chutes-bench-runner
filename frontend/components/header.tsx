'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useAuth } from '@/contexts/auth-context';

export function Header() {
  const { isLoading, isAuthenticated, idpConfigured, user, hasInvokeScope, login, logout } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const navItems = [
    { href: '/', label: 'Run Benchmarks' },
    { href: '/runs', label: 'History' },
    { href: '/api-docs', label: 'API' },
    { href: '/ops', label: 'Ops' },
    { href: '/verify', label: 'Verify' },
  ];

  const authControl = isLoading ? (
    <div className="h-8 w-20 animate-pulse rounded bg-ink-700" />
  ) : isAuthenticated && user ? (
    <div className="flex items-center gap-3">
      <div className="flex items-center gap-2">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-moss/20 text-sm font-medium text-moss">
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
        className="text-sm text-ink-400 transition-colors hover:text-ink-100"
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
  );

  return (
    <header className="sticky top-0 z-50 border-b border-ink-500 bg-ink-900/80 backdrop-blur-xl">
      <div className="mx-auto flex h-16 w-full max-w-screen-xl items-center justify-between px-6">
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
        
        <div className="flex items-center gap-3 md:gap-6">
          <button
            type="button"
            className="rounded-lg border border-ink-500/60 p-2 text-ink-200 transition hover:text-ink-50 md:hidden"
            aria-expanded={isMenuOpen}
            aria-controls="mobile-nav"
            onClick={() => setIsMenuOpen((prev) => !prev)}
          >
            <span className="sr-only">Toggle navigation</span>
            <svg
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              strokeWidth={2}
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>

          <div className="desktop-nav hidden items-center gap-6 md:flex">
            <nav className="flex items-center gap-6">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="text-sm text-ink-300 transition-colors hover:text-ink-100"
                >
                  {item.label}
                </Link>
              ))}
            </nav>

            <div className="h-6 w-px bg-ink-500" />

            {authControl}
          </div>
        </div>
      </div>

      {isMenuOpen && (
        <div id="mobile-nav" className="border-t border-ink-500/60 bg-ink-900/95 md:hidden">
          <div className="mx-auto flex w-full max-w-screen-xl flex-col gap-4 px-6 py-4">
            <nav className="flex flex-col gap-3">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="text-sm text-ink-200 transition-colors hover:text-ink-50"
                  onClick={() => setIsMenuOpen(false)}
                >
                  {item.label}
                </Link>
              ))}
            </nav>
            <div className="h-px bg-ink-500/60" />
            <div className="flex flex-col gap-3">
              {authControl}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}




























