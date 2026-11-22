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


def extract_physical_texture_features(
    manifest: pd.DataFrame,
    grid_px: int = 384,
    crop_mm: float = 150.0,
) -> np.ndarray:
    """Extract camera-invariant, physically scale-calibrated multi-scale texture features.

    Resamples a `crop_mm x crop_mm` physical center crop to a fixed `grid_px x grid_px`
    luminance grid, applies zero-mean unit-variance photometric standardization (invariant
    to camera brightness, contrast, and color temperature), and computes 51 features:
    - 16 log-spaced radial 2D FFT power spectrum bands (cycles/mm)
    - 26 multi-scale Difference-of-Gaussians (DoG) band energies & quantiles (0.4-25.6 mm)
    - 9 multi-scale gradient magnitude statistics
    """
    if manifest.empty:
        raise ValueError("Manifest is empty")

    target_ppm = grid_px / crop_mm
    fy = np.fft.fftfreq(grid_px, d=1.0 / target_ppm)
    fx = np.fft.rfftfreq(grid_px, d=1.0 / target_ppm)
    r2 = fy[:, None] ** 2 + fx[None, :] ** 2
    rr = np.sqrt(r2)
    freq_edges = np.geomspace(0.02, target_ppm / 2.0, 17)
    ring_masks = [(rr >= freq_edges[k]) & (rr < freq_edges[k + 1]) for k in range(16)]
    win = np.outer(np.hanning(grid_px), np.hanning(grid_px)).astype(np.float32)

    sigmas_mm = [0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6]
    g_kernels = [np.exp(-2.0 * (np.pi ** 2) * (s ** 2) * r2).astype(np.float32) for s in sigmas_mm]

    rows: list[np.ndarray] = []
    for _, r in manifest.iterrows():
        with Image.open(r["image_path"]) as im:
            oriented = ImageOps.exif_transpose(im).convert("L")
            cropped = crop_physical_square(oriented, effective_ppm=r["effective_ppm"], crop_mm=crop_mm)
            resized = cropped.resize((grid_px, grid_px), resample=Image.Resampling.BICUBIC)
            gray = np.asarray(resized, dtype=np.float32)

        gray = (gray - gray.mean()) / (gray.std() + 1e-6)

        spec = np.abs(np.fft.rfft2(gray * win)) ** 2
        band_power = np.array([spec[m].mean() for m in ring_masks], dtype=np.float64)
        log_spec = np.log1p(band_power / (band_power.sum() + 1e-12) * 1e4)

        fft_g = np.fft.rfft2(gray)
        blurs = [np.fft.irfft2(fft_g * gk, s=(grid_px, grid_px)) for gk in g_kernels]
        dog_feats: list[float] = []
        for k in range(len(blurs) - 1):
            dog = blurs[k] - blurs[k + 1]
            q50, q90, q98 = np.quantile(np.abs(dog[::2, ::2]), [0.50, 0.90, 0.98])
            dog_feats.extend([
                float(np.log1p(np.mean(dog ** 2) * 100.0)),
                float(q50),
                float(q90),
                float(q98),
            ])
        fine = gray - blurs[0]
        dog_feats.extend([
            float(np.log1p(np.mean(fine ** 2) * 100.0)),
            float(np.quantile(np.abs(fine[::2, ::2]), 0.90)),
        ])

        grad_feats: list[float] = []
        for b in (blurs[0], blurs[2], blurs[4]):
            gy, gx = np.gradient(b)
            gmag = np.hypot(gx, gy)
            q75, q95 = np.quantile(gmag[::2, ::2], [0.75, 0.95])
            grad_feats.extend([float(gmag.mean()), float(q75), float(q95)])

        rows.append(np.concatenate([log_spec, np.array(dog_feats), np.array(grad_feats)]).astype(np.float32))
    return np.asarray(rows, dtype=np.float32)


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
