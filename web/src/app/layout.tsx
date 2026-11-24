import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Navbar } from "@/components/navbar";

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "SoilVision AI — Intelligent Soil Analysis from Images",
  description:
    "Physically calibrated computer-vision soil grain-size distribution analysis using frozen DINOv2 embeddings, scale-space texture features, and monotonic sieve-curve estimation.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <body className="min-h-screen flex flex-col bg-[var(--bg)] text-[var(--fg)]">
        <Navbar />
        <main className="flex-1">{children}</main>
        <footer className="border-t border-[var(--border)] bg-[var(--surface)]">
          <div className="mx-auto max-w-6xl px-4 sm:px-6 py-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-sm text-[var(--fg-muted)]">
            <div>
              <p className="font-semibold text-[var(--fg)]">SoilVision AI</p>
              <p className="mt-1">
                Intelligent Soil Analysis from Images — AI estimates only; not a substitute for certified laboratory sieve testing.
              </p>
            </div>
            <div className="flex items-center gap-6 shrink-0">
              <a
                href="https://github.com/zainhaidar16/soil-grain-size-prediction"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-[var(--accent)] transition-colors underline underline-offset-4"
              >
                GitHub Repository
              </a>
              <a
                href="https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-[var(--accent)] transition-colors underline underline-offset-4"
              >
                Kaggle Competition
              </a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
