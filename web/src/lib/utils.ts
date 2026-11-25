import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { PredictionResponse } from "./types";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatMm(value: number | null, digits = 3): string {
  if (value === null || !Number.isFinite(value)) return "Not available";
  if (value < 0.01) return `${value.toFixed(4)} mm`;
  if (value < 1) return `${value.toFixed(digits)} mm`;
  return `${value.toFixed(2)} mm`;
}

export function formatCoeff(value: number | null): string {
  if (value === null || !Number.isFinite(value)) return "Not available";
  return value.toFixed(2);
}

export function buildSieveCsv(result: PredictionResponse): string {
  const lines: string[] = [
    `# SoilVision AI Grain-Size Distribution Report`,
    `# Sample ID,${result.sample_id}`,
    `# Analyzed At (UTC),${result.analyzed_at_utc}`,
    `# Model Version,${result.model_version}`,
    `# Images Analyzed,${result.num_images}`,
    `# Calibration Mode,${result.calibration.mode} (${result.calibration.effective_ppm} px/mm)`,
    `# Disclaimer,"${result.reliability.disclaimer}"`,
    `sieve_diameter_mm,cumulative_passing_pct`,
  ];
  for (let i = 0; i < result.sieve_diameters_mm.length; i++) {
    lines.push(`${result.sieve_labels[i]},${result.cumulative_passing_pct[i].toFixed(2)}`);
  }
  return lines.join("\n") + "\n";
}

export function triggerFileDownload(content: Blob | string, filename: string, mime = "text/csv") {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export async function generateSoilPdfReport(
  result: PredictionResponse,
  previewDataUrl: string | null
): Promise<void> {
  const { jsPDF } = await import("jspdf");
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageW = 210;
  const margin = 16;
  const contentW = pageW - margin * 2;
  let y = 18;

  // Header band
  doc.setFillColor(20, 60, 53); // #143C35
  doc.rect(0, 0, pageW, 26, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.text("SoilVision AI — Soil Grain-Size Analysis Report", margin, 12);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.text(
    `Generated: ${new Date(result.analyzed_at_utc).toUTCString()}   |   Model: ${result.model_version}`,
    margin,
    19
  );
  y = 34;

  // Sample overview + optional uploaded photo
  doc.setTextColor(23, 33, 43);
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.text("1. Sample Information & Calibration", margin, y);
  y += 5;

  const infoW = previewDataUrl ? contentW - 46 : contentW;
  doc.setDrawColor(218, 224, 221);
  doc.setFillColor(246, 248, 247);
  doc.roundedRect(margin, y, infoW, 34, 2, 2, "FD");
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  const metaRows: [string, string][] = [
    ["Sample ID:", result.sample_id],
    ["Photographs Analyzed:", String(result.num_images)],
    ["Calibration Status:", `${result.calibration.status} (${result.calibration.effective_ppm} px/mm)`],
    ["Calibration Source:", result.calibration.source.slice(0, 56)],
    ["Indicative Category:", result.classification.category],
  ];
  let my = y + 6;
  for (const [k, v] of metaRows) {
    doc.setFont("helvetica", "bold");
    doc.text(k, margin + 3, my);
    doc.setFont("helvetica", "normal");
    doc.text(v, margin + 40, my);
    my += 6;
  }

  if (previewDataUrl) {
    try {
      const fmt = previewDataUrl.startsWith("data:image/png") ? "PNG" : "JPEG";
      doc.addImage(previewDataUrl, fmt, margin + infoW + 4, y, 42, 34);
    } catch {
      // Skip image if format cannot be embedded
    }
  }
  y += 42;

  // 2. Logarithmic Grain-Size Distribution Chart (vector drawn)
  doc.setFont("helvetica", "bold");
  doc.setFontSize(11);
  doc.text("2. Cumulative Grain-Size Distribution Curve (Logarithmic Scale)", margin, y);
  y += 4;

  const chartX = margin + 12;
  const chartY = y + 3;
  const chartW = contentW - 16;
  const chartH = 54;
  doc.setDrawColor(200, 208, 205);
  doc.setFillColor(255, 255, 255);
  doc.rect(chartX, chartY, chartW, chartH, "FD");

  const logMin = Math.log10(0.002);
  const logMax = Math.log10(200);
  const mapX = (d: number) => chartX + ((Math.log10(d) - logMin) / (logMax - logMin)) * chartW;
  const mapY = (p: number) => chartY + chartH - (p / 100) * chartH;

  // Gridlines
  doc.setDrawColor(232, 236, 234);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(7.5);
  doc.setTextColor(95, 108, 114);
  for (const pct of [0, 20, 40, 60, 80, 100]) {
    const gy = mapY(pct);
    doc.line(chartX, gy, chartX + chartW, gy);
    doc.text(`${pct}%`, chartX - 10, gy + 1.5);
  }
  for (const d of [0.002, 0.02, 0.063, 0.2, 0.63, 2, 20, 200]) {
    const gx = mapX(d);
    doc.line(gx, chartY, gx, chartY + chartH);
    doc.text(String(d), gx - 3, chartY + chartH + 4);
  }

  // Curve polyline
  doc.setDrawColor(20, 142, 121); // #148E79
  doc.setLineWidth(0.9);
  for (let i = 0; i < result.sieve_diameters_mm.length - 1; i++) {
    doc.line(
      mapX(result.sieve_diameters_mm[i]),
      mapY(result.cumulative_passing_pct[i]),
      mapX(result.sieve_diameters_mm[i + 1]),
      mapY(result.cumulative_passing_pct[i + 1])
    );
  }
  doc.setFillColor(20, 60, 53);
  for (let i = 0; i < result.sieve_diameters_mm.length; i++) {
    doc.circle(
      mapX(result.sieve_diameters_mm[i]),
      mapY(result.cumulative_passing_pct[i]),
      1.1,
      "F"
    );
  }
  doc.setLineWidth(0.2);
  doc.setTextColor(23, 33, 43);
  doc.setFontSize(8);
  doc.text("Particle Diameter (mm, log10 scale)", chartX + chartW / 2 - 22, chartY + chartH + 9);
  y = chartY + chartH + 15;

  // 3. Full Sieve Distribution Table + 4. Composition & Engineering Parameters side-by-side
  const leftW = 80;
  const rightX = margin + leftW + 8;
  const rightW = contentW - leftW - 8;

  doc.setFont("helvetica", "bold");
  doc.setFontSize(10.5);
  doc.text("3. Sieve Distribution (11 Sieves)", margin, y);
  doc.text("4. Composition & Engineering Parameters", rightX, y);
  y += 4;

  // Left: sieve table
  doc.setFillColor(20, 60, 53);
  doc.rect(margin, y, leftW, 6.5, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(8.5);
  doc.text("Diameter (mm)", margin + 3, y + 4.5);
  doc.text("Passing (%)", margin + 48, y + 4.5);

  let ty = y + 6.5;
  doc.setTextColor(23, 33, 43);
  doc.setFont("helvetica", "normal");
  for (let i = 0; i < result.sieve_labels.length; i++) {
    if (i % 2 === 1) {
      doc.setFillColor(246, 248, 247);
      doc.rect(margin, ty, leftW, 5.8, "F");
    }
    doc.text(`${result.sieve_labels[i]} mm`, margin + 3, ty + 4.2);
    doc.text(`${result.cumulative_passing_pct[i].toFixed(2)}%`, margin + 48, ty + 4.2);
    ty += 5.8;
  }

  // Right: Composition + Engineering + Indicative Classification
  let ry = y + 4;
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8.5);
  doc.text(`Estimated Fractions (${result.composition.standard}):`, rightX, ry);
  ry += 5;
  doc.setFont("helvetica", "normal");
  doc.text(`Fine fraction (<0.063 mm): ${result.composition.fines_pct.toFixed(1)}%`, rightX, ry);
  ry += 4.8;
  doc.text(`Sand-sized (0.063–2.0 mm): ${result.composition.sand_pct.toFixed(1)}%`, rightX, ry);
  ry += 4.8;
  doc.text(`Gravel & coarser (>2.0 mm): ${result.composition.gravel_and_coarser_pct.toFixed(1)}%`, rightX, ry);
  ry += 7;

  doc.setFont("helvetica", "bold");
  doc.text("Engineering Parameters (Log-Interpolated):", rightX, ry);
  ry += 5;
  doc.setFont("helvetica", "normal");
  const ep = result.engineering_parameters;
  const epLines = [
    `D10: ${formatMm(ep.d10_mm)}    |    D30: ${formatMm(ep.d30_mm)}`,
    `D50: ${formatMm(ep.d50_mm)}    |    D60: ${formatMm(ep.d60_mm)}`,
    `Cu (D60/D10): ${formatCoeff(ep.cu)}    |    Cc: ${formatCoeff(ep.cc)}`,
  ];
  for (const line of epLines) {
    doc.text(line, rightX, ry);
    ry += 4.8;
  }
  ry += 4;
  doc.setFont("helvetica", "bold");
  doc.text("Indicative Classification (Grain-Size Only):", rightX, ry);
  ry += 4.8;
  doc.setFont("helvetica", "normal");
  const clsLines = doc.splitTextToSize(`${result.classification.symbol} — ${result.classification.rationale}`, rightW);
  doc.text(clsLines, rightX, ry);

  // 5. Limitations & Scientific Disclaimer at bottom
  const footY = Math.max(ty + 8, 245);
  doc.setDrawColor(218, 224, 221);
  doc.setFillColor(246, 248, 247);
  doc.roundedRect(margin, footY, contentW, 28, 2, 2, "FD");
  doc.setFont("helvetica", "bold");
  doc.setFontSize(8.5);
  doc.setTextColor(20, 60, 53);
  doc.text("Model Limitations & Scientific Disclaimer", margin + 3, footY + 5.5);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(8);
  doc.setTextColor(55, 65, 72);
  const discLines = doc.splitTextToSize(
    `${result.reliability.disclaimer} Trained on 24 laboratory-sieved Austrian soils (127 photos). ` +
      `Atterberg plasticity limits, organic content, and mineralogy are not measured from photographs.`,
    contentW - 6
  );
  doc.text(discLines, margin + 3, footY + 11);

  const safeName = result.sample_id.replace(/[^A-Za-z0-9._-]/g, "_") || "soil_sample";
  doc.save(`SoilVision_Report_${safeName}.pdf`);
}
