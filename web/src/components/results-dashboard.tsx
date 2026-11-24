"use client";

import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FileSpreadsheet,
  FileText,
  Info,
  RotateCcw,
  ShieldAlert,
} from "lucide-react";
import { GrainSizeChart, type GrainSizePoint } from "@/components/grain-size-chart";
import type { PredictionResponse } from "@/lib/types";
import {
  buildSieveCsv,
  formatCoeff,
  formatMm,
  generateSoilPdfReport,
  triggerFileDownload,
} from "@/lib/utils";

interface ResultsDashboardProps {
  result: PredictionResponse;
  previewUrls: string[];
  onReset: () => void;
}

export function ResultsDashboard({ result, previewUrls, onReset }: ResultsDashboardProps) {
  const [exportingPdf, setExportingPdf] = useState(false);

  const chartData: GrainSizePoint[] = result.sieve_diameters_mm.map((diameter, i) => ({
    diameter,
    label: result.sieve_labels[i],
    passing: result.cumulative_passing_pct[i],
  }));

  const handleCsvDownload = () => {
    const csv = buildSieveCsv(result);
    const safeName = result.sample_id.replace(/[^A-Za-z0-9._-]/g, "_") || "soil_sample";
    triggerFileDownload(csv, `SoilVision_Sieve_${safeName}.csv`, "text/csv;charset=utf-8");
  };

  const handlePdfDownload = async () => {
    setExportingPdf(true);
    try {
      await generateSoilPdfReport(result, previewUrls[0] ?? null);
    } finally {
      setExportingPdf(false);
    }
  };

  const ep = result.engineering_parameters;
  const comp = result.composition;
  const rel = result.reliability;

  const calBadge =
    result.calibration.status === "calibrated"
      ? "bg-[var(--accent-soft)] text-[var(--primary)] dark:text-[var(--accent)] border-[var(--accent)]/30"
      : result.calibration.status === "preset_calibrated"
      ? "bg-[var(--surface)] text-[var(--fg)] border-[var(--border)]"
      : "bg-[var(--warning-bg)] text-[var(--warning-fg)] border-[var(--warning-border)]";

  return (
    <div className="space-y-8">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-[var(--border)] pb-5">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-[var(--accent)]">
            Completed Analysis
          </p>
          <h2 className="mt-1 text-2xl font-semibold tracking-tight text-[var(--fg)]">
            {result.sample_id}
          </h2>
          <p className="mt-1 text-xs text-[var(--fg-muted)] tabular-nums">
            Analyzed {new Date(result.analyzed_at_utc).toLocaleString()} · Model {result.model_version} · Schema v{result.schema_version}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            type="button"
            onClick={handleCsvDownload}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-elevated)] px-3.5 py-2 text-sm font-medium text-[var(--fg)] hover:border-[var(--accent)] transition-colors"
          >
            <FileSpreadsheet className="h-4 w-4 text-[var(--accent)]" />
            Export CSV
          </button>
          <button
            type="button"
            onClick={handlePdfDownload}
            disabled={exportingPdf}
            className="inline-flex items-center gap-1.5 rounded-md bg-[#143C35] px-3.5 py-2 text-sm font-medium text-white hover:bg-[#194c43] disabled:opacity-60 transition-colors"
          >
            <FileText className="h-4 w-4" />
            {exportingPdf ? "Generating PDF..." : "Download PDF Report"}
          </button>
          <button
            type="button"
            onClick={onReset}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] px-3 py-2 text-sm font-medium text-[var(--fg-muted)] hover:text-[var(--fg)] transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
            New Analysis
          </button>
        </div>
      </div>

      {/* Warnings Banner (Calibration / Single Image / OOD / Large Stone) */}
      {result.warnings.length > 0 && (
        <div
          role="status"
          className="rounded-lg border border-[var(--warning-border)] bg-[var(--warning-bg)] p-4 text-sm text-[var(--warning-fg)]"
        >
          <div className="flex items-start gap-2.5">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <div className="space-y-1">
              <p className="font-semibold">Analysis Notices</p>
              <ul className="list-disc pl-4 space-y-1 text-xs leading-relaxed">
                {result.warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* A. Sample Overview */}
      <section aria-labelledby="sec-overview" className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-4 rounded-lg border border-[var(--border)] bg-[var(--surface)] p-4">
          <p className="text-xs font-medium uppercase tracking-wider text-[var(--fg-muted)] mb-3">
            Uploaded Soil Photograph{result.num_images > 1 ? `s (${result.num_images})` : ""}
          </p>
          {previewUrls.length > 0 ? (
            <div className="space-y-2">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src={previewUrls[0]}
                alt={`Uploaded soil sample ${result.sample_id}`}
                className="w-full h-48 object-cover rounded-md border border-[var(--border)]"
              />
              {previewUrls.length > 1 && (
                <div className="grid grid-cols-4 gap-1.5">
                  {previewUrls.slice(1, 5).map((u, i) => (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      key={i}
                      src={u}
                      alt={`Additional view ${i + 2}`}
                      className="h-12 w-full object-cover rounded border border-[var(--border)]"
                    />
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="h-48 rounded-md border border-dashed border-[var(--border)] flex items-center justify-center text-xs text-[var(--fg-muted)]">
              Preview unavailable
            </div>
          )}
        </div>

        <div className="lg:col-span-8 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5 flex flex-col justify-between gap-4">
          <div>
            <h3 id="sec-overview" className="text-base font-semibold text-[var(--fg)]">
              Sample Overview
            </h3>
            <dl className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-4 text-sm">
              <div>
                <dt className="text-xs text-[var(--fg-muted)]">Indicative Soil Category</dt>
                <dd className="mt-0.5 font-semibold text-[var(--fg)]">
                  {result.classification.category}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--fg-muted)]">Calibration Status</dt>
                <dd className="mt-1">
                  <span
                    className={`inline-flex items-center rounded border px-2 py-0.5 text-xs font-medium tabular-nums ${calBadge}`}
                  >
                    {result.calibration.status.replace("_", " ")} · {result.calibration.effective_ppm} px/mm
                  </span>
                </dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--fg-muted)]">Photographs Analyzed</dt>
                <dd className="mt-0.5 font-medium tabular-nums text-[var(--fg)]">
                  {result.num_images} {result.num_images === 1 ? "image" : "images (sample-level ensemble)"}
                </dd>
              </div>
              <div>
                <dt className="text-xs text-[var(--fg-muted)]">Model Version</dt>
                <dd className="mt-0.5 font-medium tabular-nums text-[var(--fg)]">
                  {result.model_version}
                </dd>
              </div>
              <div className="sm:col-span-2">
                <dt className="text-xs text-[var(--fg-muted)]">Calibration Source</dt>
                <dd className="mt-0.5 text-xs text-[var(--fg)]">{result.calibration.source}</dd>
              </div>
            </dl>
          </div>

          <div className="pt-3 border-t border-[var(--border)] flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--fg-muted)] tabular-nums">
            <span>Soft sand-specialist routing weight: {(rel.sand_routing_weight * 100).toFixed(1)}%</span>
            <span>Median diameter D50: {formatMm(ep.d50_mm)}</span>
          </div>
        </div>
      </section>

      {/* B. Interactive Grain-Size Distribution Chart */}
      <section aria-labelledby="sec-chart" className="space-y-3">
        <h3 id="sec-chart" className="text-lg font-semibold text-[var(--fg)]">
          Grain-Size Distribution Curve
        </h3>
        <GrainSizeChart data={chartData} sampleId={result.sample_id} height={380} />
      </section>

      {/* C. Sieve Table + D. Soil Composition */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <section
          aria-labelledby="sec-table"
          className="lg:col-span-6 rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5"
        >
          <div className="flex items-center justify-between gap-2 mb-4">
            <h3 id="sec-table" className="text-base font-semibold text-[var(--fg)]">
              Sieve Distribution (11 Sieves)
            </h3>
            <button
              type="button"
              onClick={handleCsvDownload}
              className="inline-flex items-center gap-1 text-xs font-medium text-[var(--accent)] hover:underline"
            >
              <Download className="h-3.5 w-3.5" />
              CSV
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-[var(--border)] text-left text-xs uppercase tracking-wider text-[var(--fg-muted)]">
                  <th className="py-2 pr-4 font-medium">Sieve Diameter</th>
                  <th className="py-2 px-4 text-right font-medium">Passing (%)</th>
                  <th className="py-2 pl-4 text-right font-medium">Retained (%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border)] tabular-nums">
                {chartData.map((row) => (
                  <tr key={row.label} className="hover:bg-[var(--surface)]/70">
                    <td className="py-2 pr-4 font-medium text-[var(--fg)]">{row.label} mm</td>
                    <td className="py-2 px-4 text-right font-semibold text-[var(--fg)]">
                      {row.passing.toFixed(2)}%
                    </td>
                    <td className="py-2 pl-4 text-right text-[var(--fg-muted)]">
                      {(100 - row.passing).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <div className="lg:col-span-6 space-y-6">
          {/* D. Soil Composition */}
          <section
            aria-labelledby="sec-comp"
            className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5"
          >
            <h3 id="sec-comp" className="text-base font-semibold text-[var(--fg)]">
              Estimated Soil Composition
            </h3>
            <p className="mt-1 text-xs text-[var(--fg-muted)]">{comp.standard}</p>

            {/* Stacked Horizontal Bar */}
            <div className="mt-4 h-7 w-full overflow-hidden rounded-md border border-[var(--border)] flex">
              {comp.fines_pct > 0 && (
                <div
                  style={{ width: `${comp.fines_pct}%` }}
                  className="bg-[#5C7C75] transition-all"
                  title={`Fine fraction (<0.063 mm): ${comp.fines_pct.toFixed(1)}%`}
                />
              )}
              {comp.sand_pct > 0 && (
                <div
                  style={{ width: `${comp.sand_pct}%` }}
                  className="bg-[#148E79] transition-all"
                  title={`Sand-sized (0.063–2.0 mm): ${comp.sand_pct.toFixed(1)}%`}
                />
              )}
              {comp.gravel_and_coarser_pct > 0 && (
                <div
                  style={{ width: `${comp.gravel_and_coarser_pct}%` }}
                  className="bg-[#143C35] transition-all"
                  title={`Gravel & coarser (>2.0 mm): ${comp.gravel_and_coarser_pct.toFixed(1)}%`}
                />
              )}
            </div>

            <div className="mt-4 grid grid-cols-3 gap-3 text-sm tabular-nums">
              <div className="rounded-md bg-[var(--surface)] p-3 border border-[var(--border)]">
                <div className="flex items-center gap-1.5 text-xs text-[var(--fg-muted)]">
                  <span className="h-2.5 w-2.5 rounded-sm bg-[#5C7C75]" />
                  Fines (&lt;0.063 mm)
                </div>
                <p className="mt-1 text-lg font-semibold text-[var(--fg)]">
                  {comp.fines_pct.toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md bg-[var(--surface)] p-3 border border-[var(--border)]">
                <div className="flex items-center gap-1.5 text-xs text-[var(--fg-muted)]">
                  <span className="h-2.5 w-2.5 rounded-sm bg-[#148E79]" />
                  Sand (0.063–2 mm)
                </div>
                <p className="mt-1 text-lg font-semibold text-[var(--fg)]">
                  {comp.sand_pct.toFixed(1)}%
                </p>
              </div>
              <div className="rounded-md bg-[var(--surface)] p-3 border border-[var(--border)]">
                <div className="flex items-center gap-1.5 text-xs text-[var(--fg-muted)]">
                  <span className="h-2.5 w-2.5 rounded-sm bg-[#143C35]" />
                  Gravel+ (&gt;2 mm)
                </div>
                <p className="mt-1 text-lg font-semibold text-[var(--fg)]">
                  {comp.gravel_and_coarser_pct.toFixed(1)}%
                </p>
              </div>
            </div>
            <p className="mt-3 text-xs text-[var(--fg-muted)] leading-relaxed">{comp.note}</p>
          </section>

          {/* E. Engineering Parameters */}
          <section
            aria-labelledby="sec-eng"
            className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5"
          >
            <h3 id="sec-eng" className="text-base font-semibold text-[var(--fg)]">
              Engineering Parameters
            </h3>
            <p className="mt-1 text-xs text-[var(--fg-muted)]">
              Logarithmically interpolated from the predicted 11-sieve curve.
            </p>
            <dl className="mt-4 grid grid-cols-2 sm:grid-cols-3 gap-3 tabular-nums">
              {[
                { label: "D10 (Effective Size)", value: formatMm(ep.d10_mm) },
                { label: "D30", value: formatMm(ep.d30_mm) },
                { label: "D50 (Median Size)", value: formatMm(ep.d50_mm) },
                { label: "D60", value: formatMm(ep.d60_mm) },
                { label: "Cu = D60 / D10", value: formatCoeff(ep.cu) },
                { label: "Cc = D30² / (D10·D60)", value: formatCoeff(ep.cc) },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3"
                >
                  <dt className="text-[11px] text-[var(--fg-muted)]">{item.label}</dt>
                  <dd className="mt-1 text-sm font-semibold text-[var(--fg)]">{item.value}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>

      {/* F. Indicative Soil Classification & G. Prediction Reliability */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <section
          aria-labelledby="sec-class"
          className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5"
        >
          <div className="flex items-start justify-between gap-2">
            <div>
              <h3 id="sec-class" className="text-base font-semibold text-[var(--fg)]">
                Indicative Soil Classification
              </h3>
              <p className="mt-0.5 text-xs text-[var(--fg-muted)]">
                {result.classification.standard}
              </p>
            </div>
            <span className="rounded border border-[var(--border)] bg-[var(--surface)] px-2 py-0.5 text-xs font-medium text-[var(--fg-muted)]">
              Non-definitive
            </span>
          </div>

          <div className="mt-4 rounded-md border border-[var(--border)] bg-[var(--surface)] p-3.5">
            <p className="text-sm font-semibold text-[var(--fg)]">
              {result.classification.symbol}
            </p>
            <p className="mt-1 text-xs text-[var(--fg-muted)] leading-relaxed">
              {result.classification.rationale}
            </p>
          </div>

          <div className="mt-4">
            <p className="text-xs font-medium text-[var(--fg)]">
              Required laboratory inputs unavailable from photographs:
            </p>
            <ul className="mt-1.5 list-disc pl-4 space-y-1 text-xs text-[var(--fg-muted)]">
              {result.classification.missing_laboratory_inputs.map((m) => (
                <li key={m}>{m}</li>
              ))}
            </ul>
          </div>
        </section>

        <section
          aria-labelledby="sec-rel"
          className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-5 flex flex-col justify-between gap-4"
        >
          <div>
            <h3 id="sec-rel" className="text-base font-semibold text-[var(--fg)]">
              Prediction Reliability &amp; Uncertainty
            </h3>
            <dl className="mt-4 grid grid-cols-2 gap-3 text-xs tabular-nums">
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
                <dt className="text-[var(--fg-muted)]">Bayesian Leverage (OOD)</dt>
                <dd className="mt-1 flex items-center gap-1.5 font-semibold text-[var(--fg)]">
                  {rel.ood_warning ? (
                    <ShieldAlert className="h-3.5 w-3.5 text-amber-600" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-[#148E79]" />
                  )}
                  {rel.bayesian_leverage.toFixed(2)}{" "}
                  <span className="font-normal text-[var(--fg-muted)]">
                    ({rel.ood_warning ? "Elevated" : "Within envelope"})
                  </span>
                </dd>
              </div>
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
                <dt className="text-[var(--fg-muted)]">Multi-Photo Dispersion</dt>
                <dd className="mt-1 font-semibold text-[var(--fg)]">
                  {rel.photo_dispersion_pct !== null
                    ? `±${rel.photo_dispersion_pct.toFixed(1)} pp (0.2–0.63 mm)`
                    : "Single photo (N/A)"}
                </dd>
              </div>
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
                <dt className="text-[var(--fg-muted)]">Detected Stone Area ≥20 mm</dt>
                <dd className="mt-1 font-semibold text-[var(--fg)]">
                  {(rel.coarse_stone_area_20mm * 100).toFixed(1)}%{" "}
                  <span className="font-normal text-[var(--fg-muted)]">
                    ({rel.large_stone_correction_applied ? "Cap applied" : "Below 10% gate"})
                  </span>
                </dd>
              </div>
              <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3">
                <dt className="text-[var(--fg-muted)]">Calibration Mode</dt>
                <dd className="mt-1 font-semibold text-[var(--fg)] capitalize">
                  {rel.calibration_status.replace("_", " ")}
                </dd>
              </div>
            </dl>
          </div>

          <div className="rounded-md border border-[var(--border)] bg-[var(--surface)] p-3 text-xs text-[var(--fg-muted)] flex items-start gap-2 leading-relaxed">
            <Info className="h-4 w-4 text-[var(--accent)] shrink-0 mt-0.5" />
            <p>{rel.disclaimer}</p>
          </div>
        </section>
      </div>
    </div>
  );
}
