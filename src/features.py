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
                pil_batch = [
                    preprocess_soil_image(p, effective_ppm=ppm, crop_mm=crop_mm, image_size=self.image_size)
                    for p, ppm in zip(batch_paths, batch_ppms)
                ]
                chunks.append(self._encode_batch(pil_batch))
            all_scale_features.append(np.vstack(chunks))

        if len(all_scale_features) == 1:
            return all_scale_features[0]
        return l2_normalize(np.concatenate(all_scale_features, axis=1))


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
        Aggregation mode: `"mean"` (L2-normalized mean embedding) or `"mean_std"`.
    """
    feats = np.asarray(image_features, dtype=np.float32)
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D image_features, got shape {feats.shape}")
    if len(feats) != len(image_sample_ids):
        raise ValueError("image_features and image_sample_ids must have the same length")
    if agg not in {"mean", "mean_std"}:
        raise ValueError(f"Unsupported agg={agg!r}; use 'mean' or 'mean_std'")

    sample_arr = np.asarray(list(image_sample_ids))
    rows: list[np.ndarray] = []
    for sid in target_sample_ids:
        mask = sample_arr == sid
        if not mask.any():
            raise ValueError(f"No image features found for sample_id={sid!r}")
        group = feats[mask]
        mean_vec = group.mean(axis=0)
        if agg == "mean":
            rows.append(l2_normalize(mean_vec[None, :])[0])
        else:
            std_vec = group.std(axis=0)
            rows.append(np.concatenate([mean_vec, std_vec], axis=0))
    return np.vstack(rows)
