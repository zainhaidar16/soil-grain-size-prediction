import { z } from "zod";

export const SIEVE_DIAMETERS_MM = [
  0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200,
] as const;

export const CALIBRATION_MODES = [
  "camera_preset",
  "fov_mm",
  "ppm",
  "uncalibrated",
] as const;

export type CalibrationMode = (typeof CALIBRATION_MODES)[number];

export interface CameraPreset {
  label: string;
  ref_width: number;
  ref_height: number;
  native_ppm: number;
}

export interface HealthResponse {
  status: "ok" | "degraded" | "offline";
  model_ready: boolean;
  model_version: string | null;
  error: string | null;
}

export interface ModelInfoResponse {
  schema_version: string;
  model_version: string;
  metadata: {
    model_version: string;
    exported_at_utc?: string;
    recovery_commit?: string;
    architecture?: string;
    backbone?: string;
    training_samples?: number;
    training_photos?: number;
    validation_metrics?: {
      kaggle_public_emd?: number;
      option1_kaggle_public_emd?: number;
      location_15_site_oof_emd?: number;
      cross_phone_emd?: number;
    };
  };
  camera_presets: Record<string, CameraPreset>;
  limits: {
    max_files: number;
    max_file_mb: number;
    allowed_formats: string[];
    min_dimension_px: number;
  };
  limitations: string[];
  disclaimer: string;
}

export interface PredictionResponse {
  schema_version: string;
  sample_id: string;
  analyzed_at_utc: string;
  model_version: string;
  num_images: number;
  calibration: {
    mode: CalibrationMode;
    status: "calibrated" | "preset_calibrated" | "uncalibrated";
    effective_ppm: number;
    fov_width_mm?: number;
    camera_preset?: string;
    camera_label?: string;
    source: string;
  };
  sieve_diameters_mm: number[];
  sieve_labels: string[];
  cumulative_passing_pct: number[];
  composition: {
    standard: string;
    fines_pct: number;
    sand_pct: number;
    gravel_and_coarser_pct: number;
    note: string;
  };
  engineering_parameters: {
    d10_mm: number | null;
    d30_mm: number | null;
    d50_mm: number | null;
    d60_mm: number | null;
    cu: number | null;
    cc: number | null;
  };
  classification: {
    standard: string;
    symbol: string;
    category: string;
    rationale: string;
    missing_laboratory_inputs: string[];
    is_definitive: boolean;
  };
  reliability: {
    sand_routing_weight: number;
    bayesian_leverage: number;
    ood_warning: boolean;
    photo_dispersion_pct: number | null;
    coarse_stone_area_20mm: number;
    coarse_stone_area_63mm: number;
    large_stone_correction_applied: boolean;
    calibration_status: string;
    disclaimer: string;
  };
  warnings: string[];
}

export const analysisFormSchema = z
  .object({
    sampleId: z
      .string()
      .trim()
      .max(80, "Sample ID must be 80 characters or fewer")
      .optional(),
    calibrationMode: z.enum(CALIBRATION_MODES),
    cameraPreset: z.string().optional(),
    fovWidthMm: z.string().optional(),
    effectivePpm: z.string().optional(),
    applyStoneCorrection: z.boolean(),
  })
  .superRefine((val, ctx) => {
    if (val.calibrationMode === "camera_preset" && !val.cameraPreset) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        path: ["cameraPreset"],
        message: "Select a camera calibration preset.",
      });
    }
    if (val.calibrationMode === "fov_mm") {
      const n = Number(val.fovWidthMm);
      if (!Number.isFinite(n) || n < 20 || n > 2000) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["fovWidthMm"],
          message: "Enter a field-of-view width between 20 and 2000 mm.",
        });
      }
    }
    if (val.calibrationMode === "ppm") {
      const n = Number(val.effectivePpm);
      if (!Number.isFinite(n) || n < 0.5 || n > 100) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ["effectivePpm"],
          message: "Enter a scale between 0.5 and 100 px/mm.",
        });
      }
    }
  });

export type AnalysisFormValues = z.infer<typeof analysisFormSchema>;
