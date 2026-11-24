import Link from "next/link";
import { ArrowRight, Camera, Cpu, FileCheck2, Ruler, Scale, ShieldCheck } from "lucide-react";
import { GrainSizeChart, type GrainSizePoint } from "@/components/grain-size-chart";

// Illustrative curve from actual training validation sample (H616 clean sand, lab-sieved)
const SAMPLE_PREVIEW_CURVE: GrainSizePoint[] = [
  { diameter: 0.002, label: "0.002", passing: 0.4 },
  { diameter: 0.0063, label: "0.0063", passing: 0.9 },
  { diameter: 0.02, label: "0.02", passing: 1.6 },
  { diameter: 0.063, label: "0.063", passing: 2.8 },
  { diameter: 0.2, label: "0.2", passing: 47.7 },
  { diameter: 0.63, label: "0.63", passing: 97.6 },
  { diameter: 2, label: "2", passing: 99.8 },
  { diameter: 6.3, label: "6.3", passing: 100.0 },
  { diameter: 20, label: "20", passing: 100.0 },
  { diameter: 63, label: "63", passing: 100.0 },
  { diameter: 200, label: "200", passing: 100.0 },
];

export default function HomePage() {
  return (
    <div className="divide-y divide-[var(--border)]">
      {/* Hero Section */}
      <section className="mx-auto max-w-6xl px-4 sm:px-6 py-16 sm:py-24">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 items-center">
          <div className="lg:col-span-6 space-y-6">
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">
              Computer-Vision Soil Characterization
            </p>
            <h1 className="text-3xl sm:text-4xl lg:text-[42px] font-semibold tracking-tight text-[var(--fg)] leading-[1.15]">
              Intelligent Soil Analysis from Images
            </h1>
            <p className="text-base text-[var(--fg-muted)] leading-relaxed max-w-xl">
              SoilVision AI estimates 11-sieve cumulative grain-size distributions (0.002 mm to
              200 mm) directly from physically calibrated soil photographs using frozen DINOv2
              representations, multi-scale photogranulometry, and monotonic curve constraints.
            </p>
            <div className="flex flex-wrap items-center gap-3 pt-2">
              <Link
                href="/analyze"
                className="inline-flex items-center gap-2 rounded-md bg-[#143C35] px-5 py-3 text-sm font-medium text-white hover:bg-[#194c43] transition-colors"
              >
                Analyze Soil
                <ArrowRight className="h-4 w-4" />
              </Link>
              <a
                href="#methodology"
                className="inline-flex items-center rounded-md border border-[var(--border)] px-5 py-3 text-sm font-medium text-[var(--fg)] hover:border-[var(--accent)] transition-colors"
              >
                Methodology &amp; Limits
              </a>
            </div>
            <div className="grid grid-cols-3 gap-4 pt-4 border-t border-[var(--border)] tabular-nums">
              <div>
                <p className="text-xs text-[var(--fg-muted)]">Sieve Points</p>
                <p className="mt-0.5 text-lg font-semibold text-[var(--fg)]">11 (0.002–200 mm)</p>
              </div>
              <div>
                <p className="text-xs text-[var(--fg-muted)]">Kaggle Public EMD</p>
                <p className="mt-0.5 text-lg font-semibold text-[var(--fg)]">37.32</p>
              </div>
              <div>
                <p className="text-xs text-[var(--fg-muted)]">Training Soils</p>
                <p className="mt-0.5 text-lg font-semibold text-[var(--fg)]">24 (127 photos)</p>
              </div>
            </div>
          </div>

          <div className="lg:col-span-6 space-y-2.5">
            <div className="flex items-center justify-between text-xs text-[var(--fg-muted)]">
              <span className="font-medium text-[var(--fg)]">
                Grain-Size Curve Preview (Reference Illustration)
              </span>
              <span className="tabular-nums">ISO 14688-1 log-scale</span>
            </div>
            <GrainSizeChart data={SAMPLE_PREVIEW_CURVE} height={320} showDownload={false} />
            <p className="text-xs text-[var(--fg-muted)]">
              Illustrative laboratory-sieved curve (clean sand reference) showing the 11 standard
              output sieves. Upload a photograph in the workspace to run real model inference.
            </p>
          </div>
        </div>
      </section>

      {/* How the Hybrid Model Works */}
      <section id="methodology" className="bg-[var(--surface)]">
        <div className="mx-auto max-w-6xl px-4 sm:px-6 py-16">
          <div className="max-w-2xl">
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">
              Inference Architecture
            </p>
            <h2 className="mt-2 text-2xl font-semibold tracking-tight text-[var(--fg)]">
              How SoilVision AI Processes Photographs
            </h2>
            <p className="mt-2 text-sm text-[var(--fg-muted)] leading-relaxed">
              Built directly on our verified competition pipeline (commit{" "}
              <code className="text-xs font-mono">3810025</code>, Public EMD 37.32208), combining
              physically calibrated texture analysis with self-supervised vision features.
            </p>
          </div>

          <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              {
                icon: Ruler,
                title: "1. Physical Scale & JPEG Normalization",
                body: "Rescales each uploaded photograph to a standardized physical grid (4.5 px/mm) using explicit user-provided scale calibration and train-like JPEG normalization to reduce camera-domain shift.",
              },
              {
                icon: Cpu,
                title: "2. Hybrid Multi-Scale Feature Extraction",
                body: "Extracts 29 DoG, Lab, and coarse-granulometry features, 14 scale-only DoG bands (Scale14), and frozen DINOv2-S RGB & GrayNorm embeddings across 90 mm and 150 mm physical crops.",
              },
              {
                icon: Scale,
                title: "3. Soft-Routed Specialists & Stone Gate",
                body: "Blends Logit-PCA Ridge, Weibull Ridge, and DINOv2 Latent-Kernel-Median with a scale-aware Weibull sand specialist via sigmoid routing, followed by large-stone capping (>=20 mm) and isotonic repair.",
              },
            ].map((step) => (
              <div
                key={step.title}
                className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-6 space-y-3"
              >
                <step.icon className="h-5 w-5 text-[var(--accent)]" />
                <h3 className="text-base font-semibold text-[var(--fg)]">{step.title}</h3>
                <p className="text-sm text-[var(--fg-muted)] leading-relaxed">{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Scientific Methodology & Important Limitations */}
      <section className="mx-auto max-w-6xl px-4 sm:px-6 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
          <div className="space-y-4">
            <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">
              Engineering Outputs
            </p>
            <h2 className="text-2xl font-semibold tracking-tight text-[var(--fg)]">
              Transparent Outputs for Technical Review
            </h2>
            <ul className="space-y-3 text-sm text-[var(--fg-muted)]">
              <li className="flex items-start gap-2.5">
                <FileCheck2 className="h-4 w-4 text-[var(--accent)] mt-0.5 shrink-0" />
                <span>
                  <strong className="text-[var(--fg)]">Monotonic 11-sieve curves:</strong> Every
                  prediction is projected via log-weighted isotonic regression onto [0, 100]% with
                  200 mm pinned to 100%.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <Camera className="h-4 w-4 text-[var(--accent)] mt-0.5 shrink-0" />
                <span>
                  <strong className="text-[var(--fg)]">Multi-photo aggregation:</strong> Upload 1 to
                  12 photos of the same sample; multi-photo inputs average features and expose
                  photo-to-photo dispersion.
                </span>
              </li>
              <li className="flex items-start gap-2.5">
                <ShieldCheck className="h-4 w-4 text-[var(--accent)] mt-0.5 shrink-0" />
                <span>
                  <strong className="text-[var(--fg)]">Honest parameter calculation:</strong> D10,
                  D30, D50, D60, Cu, and Cc are log-interpolated only when supported by the curve,
                  otherwise explicitly marked Not available.
                </span>
              </li>
            </ul>
          </div>

          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-6 space-y-3">
            <h3 className="text-base font-semibold text-[var(--fg)]">
              Important Scientific Limitations
            </h3>
            <ul className="list-disc pl-5 space-y-2 text-sm text-[var(--fg-muted)] leading-relaxed">
              <li>
                <strong className="text-[var(--fg)]">Small training set:</strong> Fitted on 24
                laboratory-sieved Austrian soils (127 photographs); unseen regional geology,
                moisture, or lighting increases prediction error.
              </li>
              <li>
                <strong className="text-[var(--fg)]">Optical resolution limit:</strong> Silt and
                clay particles (&lt;0.063 mm) are smaller than a single pixel at 4.5 px/mm and
                cannot be directly resolved or separated by mineralogy.
              </li>
              <li>
                <strong className="text-[var(--fg)]">Scale sensitivity:</strong> Accurate
                grain-size estimation requires physical camera calibration (known px/mm or
                field-of-view width). Uncalibrated photos carry explicit reliability warnings.
              </li>
              <li>
                <strong className="text-[var(--fg)]">Not a laboratory substitute:</strong> Results
                are AI estimates derived from photographs and cannot replace certified sieve
                analysis, hydrometer testing, or Atterberg plasticity limits.
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
