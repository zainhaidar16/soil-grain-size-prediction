"use client";

import { useRef } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Download } from "lucide-react";

export interface GrainSizePoint {
  diameter: number;
  label: string;
  passing: number;
}

interface GrainSizeChartProps {
  data: GrainSizePoint[];
  sampleId?: string;
  height?: number;
  showDownload?: boolean;
}

const LOG_TICKS = [0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200];

export function GrainSizeChart({
  data,
  sampleId = "soil_sample",
  height = 360,
  showDownload = true,
}: GrainSizeChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const handleDownloadPng = () => {
    const svg = containerRef.current?.querySelector("svg");
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const scale = 2;
    const canvas = document.createElement("canvas");
    canvas.width = Math.max(800, rect.width * scale);
    canvas.height = Math.max(420, rect.height * scale);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const clone = svg.cloneNode(true) as SVGSVGElement;
    clone.setAttribute("width", String(rect.width));
    clone.setAttribute("height", String(rect.height));
    const xml = new XMLSerializer().serializeToString(clone);
    const blob = new Blob([xml], { type: "image/svg+xml;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      const a = document.createElement("a");
      a.download = `SoilVision_Curve_${sampleId.replace(/[^A-Za-z0-9._-]/g, "_")}.png`;
      a.href = canvas.toDataURL("image/png");
      a.click();
    };
    img.src = url;
  };

  return (
    <div className="w-full">
      {showDownload && (
        <div className="mb-3 flex flex-wrap items-center justify-between gap-2 text-xs text-[var(--fg-muted)]">
          <div className="flex flex-wrap items-center gap-4">
            <span>ISO 14688-1 boundaries: Fines &lt; 0.063 mm | Sand 0.063–2 mm | Gravel &gt; 2 mm</span>
          </div>
          <button
            type="button"
            onClick={handleDownloadPng}
            className="inline-flex items-center gap-1.5 rounded-md border border-[var(--border)] bg-[var(--surface-elevated)] px-2.5 py-1.5 text-xs font-medium text-[var(--fg)] hover:border-[var(--accent)] transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            Download PNG
          </button>
        </div>
      )}

      <div
        ref={containerRef}
        className="w-full rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-3 sm:p-4"
        style={{ height }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 14, right: 20, left: 4, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
            <XAxis
              dataKey="diameter"
              type="number"
              scale="log"
              domain={[0.002, 200]}
              ticks={LOG_TICKS}
              tickFormatter={(v: number) => (v < 0.01 ? v.toString() : v < 1 ? String(v) : String(v))}
              stroke="var(--fg-muted)"
              tick={{ fontSize: 11, fill: "var(--fg-muted)" }}
              label={{
                value: "Particle Diameter (mm, logarithmic scale)",
                position: "insideBottom",
                offset: -14,
                style: { fontSize: 12, fill: "var(--fg-muted)" },
              }}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 20, 40, 60, 80, 100]}
              tickFormatter={(v: number) => `${v}%`}
              stroke="var(--fg-muted)"
              tick={{ fontSize: 11, fill: "var(--fg-muted)" }}
              width={46}
            />
            <ReferenceLine
              x={0.063}
              stroke="#64748b"
              strokeDasharray="4 4"
              label={{ value: "0.063", position: "top", fontSize: 10, fill: "#64748b" }}
            />
            <ReferenceLine
              x={2}
              stroke="#64748b"
              strokeDasharray="4 4"
              label={{ value: "2 mm", position: "top", fontSize: 10, fill: "#64748b" }}
            />
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const pt = payload[0].payload as GrainSizePoint;
                return (
                  <div className="rounded-md border border-[var(--border)] bg-[var(--surface-elevated)] px-3 py-2 text-xs shadow-sm">
                    <p className="font-semibold text-[var(--fg)] tabular-nums">
                      Sieve {pt.label} mm
                    </p>
                    <p className="mt-0.5 text-[var(--accent)] font-medium tabular-nums">
                      Passing: {pt.passing.toFixed(2)}%
                    </p>
                    <p className="text-[var(--fg-muted)] tabular-nums">
                      Retained: {(100 - pt.passing).toFixed(2)}%
                    </p>
                  </div>
                );
              }}
            />
            <Line
              type="monotone"
              dataKey="passing"
              stroke="#148E79"
              strokeWidth={2.5}
              dot={{ r: 4, fill: "#143C35", stroke: "#ffffff", strokeWidth: 1.5 }}
              activeDot={{ r: 6, fill: "#148E79", stroke: "#ffffff", strokeWidth: 2 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
