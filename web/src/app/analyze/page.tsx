"use client";

import { useEffect, useRef, useState } from "react";
import { useForm, useWatch } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  Loader2,
  Ruler,
  Trash2,
  Upload,
} from "lucide-react";
import { ResultsDashboard } from "@/components/results-dashboard";
import {
  analysisFormSchema,
  type AnalysisFormValues,
  type CameraPreset,
  type ModelInfoResponse,
  type PredictionResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";

const ALLOWED_TYPES = new Set(["image/jpeg", "image/jpg", "image/png", "image/webp"]);
const MAX_FILES = 12;
const MAX_FILE_BYTES = 15 * 1024 * 1024;
const VERCEL_SAFE_TOTAL_BYTES = 3.5 * 1024 * 1024;
const TARGET_MAX_EDGE_PX = 1800;

interface UploadedImage {
  file: File;
  previewUrl: string;
  width: number;
  height: number;
}

const DEFAULT_PRESETS: Record<string, CameraPreset> = {
  iphone_16: { label: "iPhone 16 (210 mm stand-off)", ref_width: 5712, ref_height: 4284, native_ppm: 19.525 },
  iphone_14: { label: "iPhone 14 (210 mm stand-off)", ref_width: 4032, ref_height: 3024, native_ppm: 13.941 },
  motorola_edge: { label: "Motorola Edge (210 mm stand-off)", ref_width: 4000, ref_height: 1800, native_ppm: 11.492 },
  samsung_a52: { label: "Samsung A52 (210 mm stand-off)", ref_width: 4624, ref_height: 3468, native_ppm: 13.151 },
  motorola_edge_60_fusion: { label: "Motorola Edge 60 Fusion (210 mm stand-off)", ref_width: 4096, ref_height: 3072, native_ppm: 17.325 },
};

const PROGRESS_STEPS = [
  "Validating soil photographs and scale calibration",
  "Normalizing physical scale (4.5 px/mm) and extracting DoG/granulometry features",
  "Running frozen DINOv2-S embeddings and soft-routed specialists",
  "Enforcing monotonic CDF constraints and computing engineering parameters",
];

function readImageDimensions(file: File): Promise<{ width: number; height: number; previewUrl: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`Could not read ${file.name}`));
    reader.onload = () => {
      const dataUrl = String(reader.result);
      const img = new Image();
      img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight, previewUrl: dataUrl });
      img.onerror = () => reject(new Error(`Could not decode ${file.name} as a valid image`));
      img.src = dataUrl;
    };
    reader.readAsDataURL(file);
  });
}

async function resizeImageByRatio(
  item: UploadedImage,
  ratio: number,
  quality: number
): Promise<File> {
  const img = new Image();
  await new Promise<void>((resolve, reject) => {
    img.onload = () => resolve();
    img.onerror = () => reject(new Error(`Failed to load ${item.file.name} for optimization`));
    img.src = item.previewUrl;
  });

  const targetW = Math.max(224, Math.round(item.width * ratio));
  const targetH = Math.max(224, Math.round(item.height * ratio));
  const canvas = document.createElement("canvas");
  canvas.width = targetW;
  canvas.height = targetH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return item.file;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(img, 0, 0, targetW, targetH);

  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob((b) => resolve(b), "image/jpeg", quality)
  );
  if (!blob) return item.file;

  const baseName = item.file.name.replace(/\.[^.]+$/, "");
  return new File([blob], `${baseName}.jpg`, { type: "image/jpeg" });
}

export default function AnalyzePage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [images, setImages] = useState<UploadedImage[]>([]);
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [progressIdx, setProgressIdx] = useState(0);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [presets, setPresets] = useState<Record<string, CameraPreset>>(DEFAULT_PRESETS);

  const form = useForm<AnalysisFormValues>({
    resolver: zodResolver(analysisFormSchema),
    defaultValues: {
      sampleId: "",
      calibrationMode: "fov_mm",
      cameraPreset: "iphone_16",
      fovWidthMm: "210",
      effectivePpm: "19.5",
      applyStoneCorrection: true,
    },
  });

  const calibrationMode = useWatch({ control: form.control, name: "calibrationMode" });

  useEffect(() => {
    fetch("/api/model-info")
      .then((r) => (r.ok ? r.json() : null))
      .then((info: ModelInfoResponse | null) => {
        if (info?.camera_presets && Object.keys(info.camera_presets).length > 0) {
          setPresets(info.camera_presets);
        }
      })
      .catch(() => {
        // Fallback to default presets if backend is waking up
      });
  }, []);

  useEffect(() => {
    if (!submitting) return;
    const id = window.setInterval(() => {
      setProgressIdx((prev) => (prev < PROGRESS_STEPS.length - 1 ? prev + 1 : prev));
    }, 2200);
    return () => window.clearInterval(id);
  }, [submitting]);

  const handleFiles = async (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return;
    setUploadError(null);
    setApiError(null);

    const incoming = Array.from(fileList);
    if (images.length + incoming.length > MAX_FILES) {
      setUploadError(`You can upload up to ${MAX_FILES} photographs per soil sample.`);
      return;
    }

    const validated: UploadedImage[] = [];
    for (const file of incoming) {
      const extOk = /\.(jpe?g|png|webp)$/i.test(file.name);
      if (!ALLOWED_TYPES.has(file.type) && !extOk) {
        setUploadError(`Unsupported format: "${file.name}". Allowed formats are JPG, PNG, and WebP.`);
        return;
      }
      if (file.size === 0) {
        setUploadError(`"${file.name}" is empty.`);
        return;
      }
      if (file.size > MAX_FILE_BYTES) {
        setUploadError(`"${file.name}" exceeds the 15 MB per-file limit.`);
        return;
      }
      try {
        const meta = await readImageDimensions(file);
        if (Math.min(meta.width, meta.height) < 224) {
          setUploadError(`"${file.name}" (${meta.width}×${meta.height} px) is below the 224 px minimum dimension.`);
          return;
        }
        validated.push({ file, ...meta });
      } catch (err) {
        setUploadError(err instanceof Error ? err.message : "Failed to decode image.");
        return;
      }
    }

    setImages((prev) => [...prev, ...validated]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const removeImage = (index: number) => {
    setImages((prev) => prev.filter((_, i) => i !== index));
  };

  const onSubmit = async (values: AnalysisFormValues) => {
    setApiError(null);
    if (images.length === 0) {
      setUploadError("Please upload at least one soil photograph before running analysis.");
      return;
    }

    setProgressIdx(0);
    setSubmitting(true);
    try {
      const totalBytes = images.reduce((acc, item) => acc + item.file.size, 0);
      const maxLongEdge = Math.max(...images.map((item) => Math.max(item.width, item.height)));
      let uploadFiles: File[] = images.map((item) => item.file);
      let scaleRatio = 1.0;

      if (totalBytes > VERCEL_SAFE_TOTAL_BYTES) {
        scaleRatio = maxLongEdge > TARGET_MAX_EDGE_PX ? TARGET_MAX_EDGE_PX / maxLongEdge : 1.0;
        uploadFiles = await Promise.all(
          images.map((item) => resizeImageByRatio(item, scaleRatio, 0.88))
        );
        const optTotal = uploadFiles.reduce((acc, f) => acc + f.size, 0);
        if (optTotal > VERCEL_SAFE_TOTAL_BYTES) {
          uploadFiles = await Promise.all(
            images.map((item) => resizeImageByRatio(item, scaleRatio, 0.76))
          );
        }
      }

      const body = new FormData();
      for (const file of uploadFiles) {
        body.append("files", file, file.name);
      }
      if (values.sampleId) body.append("sample_id", values.sampleId);
      body.append("calibration_mode", values.calibrationMode);
      if (values.calibrationMode === "camera_preset" && values.cameraPreset) {
        body.append("camera_preset", values.cameraPreset);
      } else if (values.calibrationMode === "fov_mm" && values.fovWidthMm) {
        body.append("fov_width_mm", values.fovWidthMm);
      } else if (values.calibrationMode === "ppm" && values.effectivePpm) {
        const adjustedPpm = Number(values.effectivePpm) * scaleRatio;
        body.append("effective_ppm", adjustedPpm.toFixed(4));
      }
      body.append("apply_stone_correction", String(values.applyStoneCorrection));

      const res = await fetch("/api/predict", { method: "POST", body });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.detail || `Analysis failed with status ${res.status}.`);
      }
      setResult(data as PredictionResponse);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Unexpected error communicating with inference service.");
    } finally {
      setSubmitting(false);
    }
  };

  if (result) {
    return (
      <div className="mx-auto max-w-6xl px-4 sm:px-6 py-10">
        <ResultsDashboard
          result={result}
          previewUrls={images.map((i) => i.previewUrl)}
          onReset={() => {
            setResult(null);
            setApiError(null);
          }}
        />
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl px-4 sm:px-6 py-10 space-y-8">
      <div className="space-y-2">
        <p className="text-xs font-semibold uppercase tracking-widest text-[var(--accent)]">
          Analysis Workspace
        </p>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight text-[var(--fg)]">
          Soil Grain-Size Analysis
        </h1>
        <p className="text-sm text-[var(--fg-muted)] max-w-2xl">
          Upload 1 to 12 overhead photographs of a single soil sample and specify its physical
          scale calibration. Multi-photo uploads (2–4 views) improve sample-level stability.
        </p>
      </div>

      <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6" noValidate>
        {/* Step 1: Image Upload */}
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-6 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-semibold text-[var(--fg)]">1. Soil Photographs</h2>
              <p className="text-xs text-[var(--fg-muted)]">
                JPG, PNG, or WebP · Min 224×224 px · Up to 15 MB per image · Max 12 images per sample
              </p>
            </div>
            {images.length > 0 && (
              <button
                type="button"
                onClick={() => setImages([])}
                className="text-xs font-medium text-[var(--fg-muted)] hover:text-red-600 transition-colors"
              >
                Clear all ({images.length})
              </button>
            )}
          </div>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragActive(false);
              void handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") fileInputRef.current?.click();
            }}
            className={cn(
              "cursor-pointer rounded-lg border-2 border-dashed p-8 text-center transition-colors",
              dragActive
                ? "border-[var(--accent)] bg-[var(--accent-soft)]/50"
                : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--accent)]"
            )}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"
              multiple
              onChange={(e) => void handleFiles(e.target.files)}
              className="hidden"
              aria-label="Upload soil photographs"
            />
            <Upload className="mx-auto h-7 w-7 text-[var(--accent)]" />
            <p className="mt-3 text-sm font-medium text-[var(--fg)]">
              Drag and drop soil photographs here, or click to browse
            </p>
            <p className="mt-1 text-xs text-[var(--fg-muted)]">
              Overhead, evenly lit photographs covering the full soil surface yield the most consistent results.
            </p>
          </div>

          {uploadError && (
            <div
              role="alert"
              className="rounded-md border border-[var(--danger-border)] bg-[var(--danger-bg)] p-3 text-xs text-[var(--danger-fg)] flex items-center gap-2"
            >
              <AlertCircle className="h-4 w-4 shrink-0" />
              <span>{uploadError}</span>
            </div>
          )}

          {images.length > 0 && (
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3 pt-2">
              {images.map((img, idx) => (
                <div
                  key={`${img.file.name}-${idx}`}
                  className="group relative rounded-md border border-[var(--border)] bg-[var(--surface)] overflow-hidden"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={img.previewUrl}
                    alt={img.file.name}
                    className="h-28 w-full object-cover"
                  />
                  <button
                    type="button"
                    onClick={() => removeImage(idx)}
                    aria-label={`Remove ${img.file.name}`}
                    className="absolute top-1.5 right-1.5 rounded bg-black/70 p-1 text-white opacity-90 hover:bg-red-700 transition-colors"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                  </button>
                  <div className="p-2 text-[11px] tabular-nums">
                    <p className="truncate font-medium text-[var(--fg)]">{img.file.name}</p>
                    <p className="text-[var(--fg-muted)]">
                      {img.width}×{img.height} px · {(img.file.size / (1024 * 1024)).toFixed(1)} MB
                    </p>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Step 2: Sample Metadata & Physical Scale Calibration */}
        <section className="rounded-lg border border-[var(--border)] bg-[var(--surface-elevated)] p-6 space-y-5">
          <div className="flex items-start gap-2.5">
            <Ruler className="h-5 w-5 text-[var(--accent)] mt-0.5 shrink-0" />
            <div>
              <h2 className="text-base font-semibold text-[var(--fg)]">
                2. Physical Scale Calibration &amp; Sample ID
              </h2>
              <p className="text-xs text-[var(--fg-muted)]">
                Grain-size prediction depends directly on physical pixel scale (px/mm). SoilVision AI
                never silently assumes capture geometry.
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div>
              <label htmlFor="sampleId" className="block text-xs font-medium text-[var(--fg)] mb-1.5">
                Sample Name / ID (optional)
              </label>
              <input
                id="sampleId"
                type="text"
                placeholder="e.g., BH-04_Depth_2.5m"
                {...form.register("sampleId")}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
              />
              {form.formState.errors.sampleId && (
                <p className="mt-1 text-xs text-red-600">{form.formState.errors.sampleId.message}</p>
              )}
            </div>

            <div>
              <label htmlFor="calibrationMode" className="block text-xs font-medium text-[var(--fg)] mb-1.5">
                Calibration Method
              </label>
              <select
                id="calibrationMode"
                {...form.register("calibrationMode")}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
              >
                <option value="fov_mm">Known Photo Width (mm) — Recommended</option>
                <option value="ppm">Known Scale (pixels / mm)</option>
                <option value="camera_preset">Reference Camera Preset (~210 mm stand-off)</option>
                <option value="uncalibrated">Uncalibrated (Reduced Accuracy)</option>
              </select>
            </div>
          </div>

          {calibrationMode === "fov_mm" && (
            <div className="max-w-sm">
              <label htmlFor="fovWidthMm" className="block text-xs font-medium text-[var(--fg)] mb-1.5">
                Physical Width Covered by Photograph (mm)
              </label>
              <input
                id="fovWidthMm"
                type="number"
                step="1"
                min="20"
                max="2000"
                {...form.register("fovWidthMm")}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm tabular-nums text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
              />
              <p className="mt-1 text-xs text-[var(--fg-muted)]">
                Measure the horizontal width of the tray/frame in millimeters (e.g., 210–300 mm).
              </p>
              {form.formState.errors.fovWidthMm && (
                <p className="mt-1 text-xs text-red-600">{form.formState.errors.fovWidthMm.message}</p>
              )}
            </div>
          )}

          {calibrationMode === "ppm" && (
            <div className="max-w-sm">
              <label htmlFor="effectivePpm" className="block text-xs font-medium text-[var(--fg)] mb-1.5">
                Effective Scale (pixels per millimeter)
              </label>
              <input
                id="effectivePpm"
                type="number"
                step="0.01"
                min="0.5"
                max="100"
                {...form.register("effectivePpm")}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm tabular-nums text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
              />
              {form.formState.errors.effectivePpm && (
                <p className="mt-1 text-xs text-red-600">{form.formState.errors.effectivePpm.message}</p>
              )}
            </div>
          )}

          {calibrationMode === "camera_preset" && (
            <div className="max-w-md">
              <label htmlFor="cameraPreset" className="block text-xs font-medium text-[var(--fg)] mb-1.5">
                Reference Phone Preset
              </label>
              <select
                id="cameraPreset"
                {...form.register("cameraPreset")}
                className="w-full rounded-md border border-[var(--border)] bg-[var(--bg)] px-3 py-2 text-sm text-[var(--fg)] focus:border-[var(--accent)] focus:outline-none"
              >
                {Object.entries(presets).map(([key, p]) => (
                  <option key={key} value={key}>
                    {p.label} ({p.native_ppm} px/mm native)
                  </option>
                ))}
              </select>
            </div>
          )}

          {calibrationMode === "uncalibrated" && (
            <div className="rounded-md border border-[var(--warning-border)] bg-[var(--warning-bg)] p-3.5 text-xs text-[var(--warning-fg)] flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
              <p className="leading-relaxed">
                <strong>Calibration Warning:</strong> Without physical scale metadata, the pipeline
                falls back to the training median (4.60 px/mm) and disables large-stone sizing. For
                reliable results, place a ruler along the edge of the tray and enter the horizontal
                field-of-view width in millimeters.
              </p>
            </div>
          )}

          <label className="flex items-center gap-2.5 text-xs text-[var(--fg-muted)] select-none pt-1">
            <input
              type="checkbox"
              {...form.register("applyStoneCorrection")}
              disabled={calibrationMode === "uncalibrated"}
              className="h-4 w-4 rounded border-[var(--border)] accent-[#148E79]"
            />
            <span>
              Enable physical large-stone (cobble ≥20 mm) surface-area cap when calibrated
            </span>
          </label>
        </section>

        {/* API Error Banner */}
        {apiError && (
          <div
            role="alert"
            className="rounded-lg border border-[var(--danger-border)] bg-[var(--danger-bg)] p-4 text-sm text-[var(--danger-fg)] flex items-start gap-2.5"
          >
            <AlertCircle className="h-4 w-4 mt-0.5 shrink-0" />
            <div>
              <p className="font-semibold">Inference Request Failed</p>
              <p className="mt-0.5 text-xs">{apiError}</p>
            </div>
          </div>
        )}

        {/* Progress Feedback */}
        {submitting && (
          <div className="rounded-lg border border-[var(--border)] bg-[var(--surface)] p-5 space-y-3">
            <div className="flex items-center gap-2.5 text-sm font-medium text-[var(--fg)]">
              <Loader2 className="h-4 w-4 animate-spin text-[var(--accent)]" />
              <span>Running SoilVision AI Inference Pipeline...</span>
            </div>
            <ul className="space-y-1.5 text-xs">
              {PROGRESS_STEPS.map((step, i) => (
                <li
                  key={step}
                  className={cn(
                    "flex items-center gap-2 transition-colors",
                    i <= progressIdx ? "text-[var(--fg)] font-medium" : "text-[var(--fg-muted)]"
                  )}
                >
                  <CheckCircle2
                    className={cn(
                      "h-3.5 w-3.5 shrink-0",
                      i < progressIdx
                        ? "text-[#148E79]"
                        : i === progressIdx
                        ? "text-[var(--accent)]"
                        : "text-[var(--border)]"
                    )}
                  />
                  {step}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Submit Button */}
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-xs text-[var(--fg-muted)]">
            Uploaded photographs are processed in memory and are not stored after analysis.
          </p>
          <button
            type="submit"
            disabled={submitting || images.length === 0}
            className="inline-flex items-center justify-center gap-2 rounded-md bg-[#143C35] px-6 py-3 text-sm font-medium text-white hover:bg-[#194c43] disabled:opacity-50 disabled:cursor-not-allowed transition-colors min-w-[160px]"
          >
            {submitting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Analyze Soil"
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
