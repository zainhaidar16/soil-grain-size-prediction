"""EXIF-aware, physical-scale-normalized preprocessing and frozen DINOv2 feature extraction."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"


def crop_physical_square(
    image: Image.Image,
    effective_ppm: float | None,
    crop_mm: float | None = 150.0,
) -> Image.Image:
    """Center-crop a square covering `crop_mm x crop_mm` of soil surface.

    When `crop_mm` or `effective_ppm` is None, falls back to the largest centered
    square (`min(width, height)`). If the requested physical crop exceeds the
    shorter image side, it is clamped to `min(width, height)`.
    """
    width, height = image.size
    min_side = min(width, height)
    if crop_mm is None or effective_ppm is None:
        crop_px = min_side
    else:
        if crop_mm <= 0 or effective_ppm <= 0:
            raise ValueError("crop_mm and effective_ppm must be positive")
        crop_px = max(1, min(min_side, int(round(crop_mm * effective_ppm))))

    left = (width - crop_px) // 2
    top = (height - crop_px) // 2
    return image.crop((left, top, left + crop_px, top + crop_px))


def preprocess_soil_image(
    image_path: Path | str,
    effective_ppm: float | None = None,
    crop_mm: float | None = 150.0,
    image_size: int = 224,
) -> Image.Image:
    """Load, EXIF-orient, physical-scale crop, and resize a soil photograph."""
    with Image.open(image_path) as im:
        oriented = ImageOps.exif_transpose(im).convert("RGB")
    cropped = crop_physical_square(oriented, effective_ppm=effective_ppm, crop_mm=crop_mm)
    return cropped.resize((image_size, image_size), resample=Image.Resampling.BICUBIC)


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, eps)


class DINOv2FeatureExtractor:
    """Frozen pretrained DINOv2 vision backbone for soil texture/grain embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV2_MODEL,
        image_size: int = 224,
        crop_mms: Sequence[float | None] = (90.0, 150.0),
        pool: str = "cls_patch_mean",
        gray_norm: bool = False,
        device: str | None = None,
    ) -> None:
        import timm
        import torch
        from torchvision import transforms

        if pool not in {"cls", "cls_patch_mean"}:
            raise ValueError(f"Unsupported pool={pool!r}; use 'cls' or 'cls_patch_mean'")

        self.model_name = model_name
        self.image_size = int(image_size)
        self.crop_mms = tuple(crop_mms)
        self.pool = pool
        self.gray_norm = bool(gray_norm)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.to(self.device)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _encode_batch(self, pil_images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        batch = torch.stack([self.transform(im) for im in pil_images]).to(self.device)
        with torch.no_grad():
            tokens = self.model.forward_features(batch)
            if tokens.ndim == 3:
                cls_tok = tokens[:, 0, :]
                if self.pool == "cls_patch_mean":
                    n_prefix = getattr(self.model, "num_prefix_tokens", 1)
                    patch_mean = tokens[:, n_prefix:, :].mean(dim=1)
                    feats = torch.cat([cls_tok, patch_mean], dim=-1)
                else:
                    feats = cls_tok
            else:
                feats = tokens
        return l2_normalize(feats.cpu().numpy())

    def extract_manifest_features(
        self,
        manifest: pd.DataFrame,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Extract frozen DINOv2 embeddings for every row in an image manifest."""
        if manifest.empty:
            raise ValueError("Manifest is empty")

        all_scale_features: list[np.ndarray] = []
        paths = manifest["image_path"].tolist()
        ppms = manifest["effective_ppm"].tolist() if "effective_ppm" in manifest else [None] * len(paths)

        for crop_mm in self.crop_mms:
            chunks: list[np.ndarray] = []
            for start in range(0, len(paths), batch_size):
                batch_paths = paths[start : start + batch_size]
                batch_ppms = ppms[start : start + batch_size]
                pil_batch = []
                for p, ppm in zip(batch_paths, batch_ppms):
                    im = preprocess_soil_image(p, effective_ppm=ppm, crop_mm=crop_mm, image_size=self.image_size)
                    if self.gray_norm:
                        g = np.asarray(im.convert("L"), dtype=np.float32)
                        g = np.clip((g - g.mean()) / (g.std() + 1e-6) * 45.0 + 128.0, 0.0, 255.0).astype(np.uint8)
                        im = Image.fromarray(np.stack([g, g, g], axis=-1), mode="RGB")
                    pil_batch.append(im)
                chunks.append(self._encode_batch(pil_batch))
            all_scale_features.append(np.vstack(chunks))

        if len(all_scale_features) == 1:
            return all_scale_features[0]
        return l2_normalize(np.concatenate(all_scale_features, axis=1))


def _disk_footprint(radius_px: int) -> np.ndarray:
    r = max(1, int(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= (r * r)


def _lbp_summary(gray: np.ndarray, radius_px: int) -> list[float]:
    """Rotation-invariant 8-neighbor Local Binary Pattern summary (6 features)."""
    r = max(1, int(radius_px))
    c = gray[r:-r, r:-r]
    offsets = [(-r, -r), (-r, 0), (-r, r), (0, r), (r, r), (r, 0), (r, -r), (0, -r)]
    counts = np.zeros_like(c, dtype=np.int32)
    h, w = c.shape
    for dy, dx in offsets:
        nb = gray[r + dy : r + dy + h, r + dx : r + dx + w]
        counts += (nb >= c).astype(np.int32)
    hist = np.bincount(counts.ravel(), minlength=9).astype(np.float64)
    hist /= hist.sum() + 1e-12
    ent = float(-(hist * np.log(hist + 1e-12)).sum())
    return [
        float(hist[0] + hist[8]),
        float(hist[1] + hist[7]),
        float(hist[2] + hist[6]),
        float(hist[3] + hist[5]),
        float(hist[4]),
        ent,
    ]


PHYSICAL_TEXTURE_FEATURE_NAMES: tuple[str, ...] = (
    "log_e_0.4mm", "log_e_0.8mm", "log_e_1.6mm", "log_e_3.2mm", "log_e_6.4mm", "log_e_12.8mm",
    "log_p90_0.4mm", "log_p90_0.8mm", "log_p90_1.6mm", "log_p90_3.2mm", "log_p90_6.4mm", "log_p90_12.8mm",
    "fft_b0", "fft_b1", "fft_b2", "fft_b3", "fft_b4", "fft_b5",
    "fft_b6", "fft_b7", "fft_b8", "fft_b9", "fft_b10", "fft_b11",
    "morph_open_0.7mm", "morph_open_1.4mm", "morph_open_2.7mm", "morph_open_5.3mm", "morph_open_10.7mm",
    "morph_close_0.7mm", "morph_close_1.4mm", "morph_close_2.7mm", "morph_close_5.3mm", "morph_close_10.7mm",
    "grad_mean_0.5mm", "grad_mean_1.5mm", "grad_mean_4.5mm",
    "grad_p90_0.5mm", "grad_p90_1.5mm", "grad_p90_4.5mm",
    "edge_frac_0.5mm", "edge_frac_1.5mm", "edge_frac_4.5mm",
    "lbp_r1_0", "lbp_r1_1", "lbp_r1_2", "lbp_r1_3", "lbp_r1_4", "lbp_r1_5",
    "lbp_r3_0", "lbp_r3_1", "lbp_r3_2", "lbp_r3_3", "lbp_r3_4", "lbp_r3_5",
    "blob_frac_3mm", "blob_frac_6mm", "blob_count_3mm", "blob_count_6mm",
)


def extract_physical_texture_features(
    manifest: pd.DataFrame,
    grid_px: int = 384,
    crop_mm: float = 128.0,
) -> np.ndarray:
    """Extract 59 camera-invariant, physically calibrated photogranulometry features.

    Takes 3 spatial crops spanning the image at `crop_mm x crop_mm`, resamples to
    a common physical grid (`grid_px / crop_mm` px/mm), removes slow illumination
    gradients (flat-fielding, sigma=22 mm), applies robust median/IQR normalization,
    and aggregates across the 3 crops via element-wise median.
    """
    from scipy.ndimage import grey_closing, grey_opening, label

    if manifest.empty:
        raise ValueError("Manifest is empty")

    ppm = grid_px / crop_mm
    fy = np.fft.fftfreq(grid_px, d=1.0 / ppm)
    fx = np.fft.rfftfreq(grid_px, d=1.0 / ppm)
    r2 = fy[:, None] ** 2 + fx[None, :] ** 2
    rr = np.sqrt(r2)
    win = np.outer(np.hanning(grid_px), np.hanning(grid_px)).astype(np.float32)

    ff_kernel = np.exp(-2.0 * (np.pi ** 2) * (22.0 ** 2) * r2).astype(np.float32)
    log_sigmas_mm = (0.4, 0.8, 1.6, 3.2, 6.4, 12.8)
    log_kernels = [
        ((2.0 * np.pi * s) ** 2 * r2 * np.exp(-2.0 * (np.pi ** 2) * (s ** 2) * r2)).astype(np.float32)
        for s in log_sigmas_mm
    ]
    grad_kernels = [np.exp(-2.0 * (np.pi ** 2) * (s ** 2) * r2).astype(np.float32) for s in (0.5, 1.5, 4.5)]

    freq_edges = np.geomspace(0.03, ppm / 2.0, 13)
    ring_masks = [(rr >= freq_edges[k]) & (rr < freq_edges[k + 1]) for k in range(12)]

    morph_ppm = ppm / 2.0
    morph_sems = [_disk_footprint(max(1, int(round(rad * morph_ppm)))) for rad in (0.7, 1.4, 2.7, 5.3, 10.7)]

    rows: list[np.ndarray] = []
    for _, r in manifest.iterrows():
        with Image.open(r["image_path"]) as im:
            oriented = ImageOps.exif_transpose(im).convert("L")
            w, h = oriented.size
            cpx = max(1, min(w, h, int(round(crop_mm * float(r["effective_ppm"])))))
            if w >= h:
                cy = (h - cpx) // 2
                coords = [(0, cy), ((w - cpx) // 2, cy), (w - cpx, cy)]
            else:
                cx = (w - cpx) // 2
                coords = [(cx, 0), (cx, (h - cpx) // 2), (cx, h - cpx)]

            crop_vecs: list[np.ndarray] = []
            for x0, y0 in coords:
                sub = oriented.crop((x0, y0, x0 + cpx, y0 + cpx)).resize(
                    (grid_px, grid_px), resample=Image.Resampling.BICUBIC
                )
                g = np.asarray(sub, dtype=np.float32)
                bg = np.fft.irfft2(np.fft.rfft2(g) * ff_kernel, s=(grid_px, grid_px))
                g = g - bg
                med = float(np.median(g))
                iqr = float(np.quantile(g[::2, ::2], 0.75) - np.quantile(g[::2, ::2], 0.25))
                g = np.clip((g - med) / (iqr + 1e-6), -4.0, 4.0)

                fft_g = np.fft.rfft2(g)
                log_e, log_p90, log_maps = [], [], []
                for lk in log_kernels:
                    lm = np.fft.irfft2(fft_g * lk, s=(grid_px, grid_px))
                    log_maps.append(lm)
                    log_e.append(float(np.log1p(np.mean(lm ** 2) * 10.0)))
                    log_p90.append(float(np.quantile(np.abs(lm[::2, ::2]), 0.90)))

                spec = np.abs(np.fft.rfft2(g * win)) ** 2
                bp = np.array([spec[m].mean() for m in ring_masks], dtype=np.float64)
                fft_f = np.log1p(bp / (bp.sum() + 1e-12) * 1e3).tolist()

                g_half = g[::2, ::2]
                open_curve = [float(np.mean(g_half - grey_opening(g_half, footprint=s))) for s in morph_sems]
                close_curve = [float(np.mean(grey_closing(g_half, footprint=s) - g_half)) for s in morph_sems]
                o_sum, c_sum = sum(open_curve) + 1e-6, sum(close_curve) + 1e-6
                open_rel = [v / o_sum for v in open_curve]
                close_rel = [v / c_sum for v in close_curve]

                g_mean, g_p90, e_frac = [], [], []
                for gk in grad_kernels:
                    sm = np.fft.irfft2(fft_g * gk, s=(grid_px, grid_px))
                    gy, gx = np.gradient(sm)
                    gm_sub = np.hypot(gx, gy)[::2, ::2]
                    g_mean.append(float(gm_sub.mean()))
                    g_p90.append(float(np.quantile(gm_sub, 0.90)))
                    e_frac.append(float((gm_sub > 1.5 * (gm_sub.mean() + 1e-6)).mean()))

                lbp1 = _lbp_summary(g_half, 1)
                lbp3 = _lbp_summary(g_half, 3)

                bm3, bm6 = np.abs(log_maps[3][::2, ::2]), np.abs(log_maps[4][::2, ::2])
                m3 = bm3 > (1.5 * bm3.std() + 1e-6)
                m6 = bm6 > (1.5 * bm6.std() + 1e-6)
                _, n3 = label(m3)
                _, n6 = label(m6)
                blob_f = [float(m3.mean()), float(m6.mean()), float(np.log1p(n3)), float(np.log1p(n6))]

                crop_vecs.append(np.array(
                    log_e + log_p90 + fft_f + open_rel + close_rel + g_mean + g_p90 + e_frac + lbp1 + lbp3 + blob_f,
                    dtype=np.float32,
                ))
            rows.append(np.median(crop_vecs, axis=0))
    return np.asarray(rows, dtype=np.float32)


def select_camera_robust_features(
    x_train: np.ndarray,
    x_moto: np.ndarray,
    x_sams: np.ndarray,
    x_test: np.ndarray | None = None,
    max_xcam_shift: float = 0.8,
    max_test_shift: float = 1.0,
) -> np.ndarray:
    """Return boolean mask of features stable across cameras and unsupervised train/test shift."""
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0) + 1e-8
    xcam_shift = np.abs(x_moto.mean(axis=0) - x_sams.mean(axis=0)) / sd
    mask = xcam_shift < max_xcam_shift
    if x_test is not None:
        test_shift = np.abs(x_test.mean(axis=0) - mu) / sd
        mask = mask & (test_shift < max_test_shift)
    if not mask.any():
        raise ValueError("No features passed camera-robustness thresholds")
    return mask


def aggregate_sample_features(
    image_features: np.ndarray,
    image_sample_ids: Sequence[str],
    target_sample_ids: Sequence[str],
    agg: str = "mean",
) -> np.ndarray:
    """Aggregate per-image embeddings at the soil-sample level.

    Parameters
    ----------
    image_features:
        Array of shape `(n_images, n_features)`.
    image_sample_ids:
        Sample ID corresponding to each row of `image_features`.
    target_sample_ids:
        Ordered list of unique soil sample IDs to output.
    agg:
        Aggregation mode: `"mean"`, `"median"`, or `"mean_std"`.
    """
    feats = np.asarray(image_features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D image_features, got shape {feats.shape}")
    if len(feats) != len(image_sample_ids):
        raise ValueError("image_features and image_sample_ids must have the same length")
    if agg not in {"mean", "median", "mean_std"}:
        raise ValueError(f"Unsupported agg={agg!r}; use 'mean', 'median', or 'mean_std'")

    sample_arr = np.asarray(list(image_sample_ids))
    rows: list[np.ndarray] = []
    for sid in target_sample_ids:
        mask = sample_arr == sid
        if not mask.any():
            raise ValueError(f"No image features found for sample_id={sid!r}")
        group = feats[mask]
        if agg == "mean":
            rows.append(l2_normalize(group.mean(axis=0, keepdims=True))[0])
        elif agg == "median":
            rows.append(l2_normalize(np.median(group, axis=0, keepdims=True))[0])
        else:
            mean_vec = group.mean(axis=0)
            std_vec = group.std(axis=0)
            rows.append(np.concatenate([mean_vec, std_vec], axis=0))
    return np.vstack(rows)
