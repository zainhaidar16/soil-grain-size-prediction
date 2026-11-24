"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { Layers, Moon, Sun, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import type { HealthResponse } from "@/lib/types";

export function Navbar() {
  const pathname = usePathname();
  const [dark, setDark] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    const stored = window.localStorage.getItem("soilvision-theme");
    const prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = stored ? stored === "dark" : prefersDark;
    document.documentElement.classList.toggle("dark", isDark);
    const frame = window.requestAnimationFrame(() => setDark(isDark));
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    let active = true;
    fetch("/api/health")
      .then((r) => r.json())
      .then((data: HealthResponse) => {
        if (active) setHealth(data);
      })
      .catch(() => {
        if (active) {
          setHealth({ status: "offline", model_ready: false, model_version: null, error: "Offline" });
        }
      });
    return () => {
      active = false;
    };
  }, []);

  const toggleTheme = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle("dark", next);
    window.localStorage.setItem("soilvision-theme", next ? "dark" : "light");
  };

  return (
    <header className="sticky top-0 z-40 border-b border-[var(--border)] bg-[var(--bg)]/95 backdrop-blur">
      <div className="mx-auto max-w-6xl px-4 sm:px-6 h-16 flex items-center justify-between gap-4">
        <div className="flex items-center gap-8">
          <Link href="/" className="flex items-center gap-2.5 group">
            <span className="flex h-8 w-8 items-center justify-center rounded-md bg-[#143C35] text-white">
              <Layers className="h-4 w-4" />
            </span>
            <span className="font-semibold tracking-tight text-[var(--fg)]">
              SoilVision <span className="text-[var(--accent)]">AI</span>
            </span>
          </Link>

          <nav className="hidden sm:flex items-center gap-1" aria-label="Main navigation">
            <Link
              href="/"
              className={cn(
                "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                pathname === "/"
                  ? "bg-[var(--accent-soft)] text-[var(--primary)] dark:text-[var(--accent)]"
                  : "text-[var(--fg-muted)] hover:text-[var(--fg)]"
              )}
            >
              Overview
            </Link>
            <Link
              href="/analyze"
              className={cn(
                "px-3 py-1.5 rounded-md text-sm font-medium transition-colors",
                pathname === "/analyze"
                  ? "bg-[var(--accent-soft)] text-[var(--primary)] dark:text-[var(--accent)]"
                  : "text-[var(--fg-muted)] hover:text-[var(--fg)]"
              )}
            >
              Analyze Soil
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          {health && (
            <div
              className="hidden md:flex items-center gap-2 rounded-md border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1 text-xs text-[var(--fg-muted)]"
              title={health.model_ready ? `Inference engine ready (${health.model_version})` : health.error || "Backend offline"}
            >
              <span
                className={cn(
                  "h-2 w-2 rounded-full",
                  health.model_ready ? "bg-[#148E79]" : "bg-amber-500"
                )}
              />
              <span className="tabular-nums">
                {health.model_ready ? "Model Ready" : "Engine Offline"}
              </span>
            </div>
          )}

          <a
            href="https://github.com/zainhaidar16/soil-grain-size-prediction"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden sm:inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] px-2.5 py-1.5 text-xs font-medium text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors"
          >
            GitHub
            <ExternalLink className="h-3 w-3" />
          </a>

          <button
            type="button"
            onClick={toggleTheme}
            aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[var(--border)] text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>

          <Link
            href="/analyze"
            className="sm:hidden inline-flex items-center rounded-md bg-[#143C35] px-3 py-1.5 text-xs font-medium text-white"
          >
            Analyze
          </Link>
        </div>
      </div>
    </header>
  );
}
