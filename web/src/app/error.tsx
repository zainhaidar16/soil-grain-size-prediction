"use client";

import Link from "next/link";
import { AlertCircle } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <div className="mx-auto max-w-xl px-4 py-20 text-center space-y-4">
      <AlertCircle className="mx-auto h-10 w-10 text-red-600" />
      <h1 className="text-2xl font-semibold text-[var(--fg)]">Something went wrong</h1>
      <p className="text-sm text-[var(--fg-muted)]">
        {error.message || "An unexpected application error occurred."}
      </p>
      <div className="flex items-center justify-center gap-3 pt-2">
        <button
          type="button"
          onClick={reset}
          className="rounded-md bg-[#143C35] px-4 py-2 text-sm font-medium text-white"
        >
          Try again
        </button>
        <Link
          href="/"
          className="rounded-md border border-[var(--border)] px-4 py-2 text-sm font-medium text-[var(--fg)]"
        >
          Return Home
        </Link>
      </div>
    </div>
  );
}
