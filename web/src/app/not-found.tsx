import Link from "next/link";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 py-24 text-center space-y-4">
      <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">404</p>
      <h1 className="text-2xl font-semibold text-[var(--fg)]">Page not found</h1>
      <p className="text-sm text-[var(--fg-muted)]">
        The requested page does not exist in SoilVision AI.
      </p>
      <div className="pt-2">
        <Link
          href="/"
          className="inline-flex rounded-md bg-[#143C35] px-4 py-2 text-sm font-medium text-white"
        >
          Return to Overview
        </Link>
      </div>
    </div>
  );
}
