"""Dataset discovery, image-to-sample mapping, and camera scale (PPM) metadata."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Ordered from most specific to least specific so e.g. Motorola_Edge_60_fusion_
# matches before Motorola_Edge_.
CAMERA_PREFIX_TO_PHONE: tuple[tuple[str, str], ...] = (
    ("Motorola_Edge_60_fusion_", "Motorola Edge 60 Fusion"),
    ("Motorola_Edge_", "Motorola Edge"),
    ("Samsung_A52_", "Samsung A52"),
    ("iPhone_16_", "iPhone 16"),
    ("iPhone16_", "iPhone 16"),
    ("iPhone_14_", "iPhone 14"),
    ("iPhone14_", "iPhone 14"),
)

_UMLAUT_TRANSLATION = str.maketrans({
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "Ä": "Ae",
    "Ö": "Oe",
    "Ü": "Ue",
    "ß": "ss",
})


def resolve_data_dir(data_dir: Path | str | None = None, repo_root: Path | None = None) -> Path:
    """Resolve the competition dataset directory from argument, env var, local data/, or kagglehub."""
    if data_dir is not None:
        resolved = Path(data_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Dataset directory not found: {resolved}")
        return resolved

    env_dir = os.environ.get("SOIL_DATA_DIR")
    if env_dir:
        resolved = Path(env_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"SOIL_DATA_DIR not found: {resolved}")
        return resolved

    root = repo_root or Path.cwd()
    local_data = root / "data"
    if local_data.exists() and any(local_data.rglob("*.csv")):
        return local_data

    cache_dir = Path.home() / ".cache" / "kagglehub" / "competitions" / "soil-grain-size-from-photos"
    if cache_dir.exists() and any(cache_dir.rglob("*.csv")):
        return cache_dir

    import kagglehub

    return Path(kagglehub.competition_download("soil-grain-size-from-photos"))


def normalize_sample_token(text: str) -> str:
    """Normalize a sample identifier token for deterministic matching."""
    cleaned = text.strip().translate(_UMLAUT_TRANSLATION)
    cleaned = cleaned.replace(",", "_")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_camera_from_filename(filename: str) -> str:
    """Map an image filename prefix to the canonical phone name in ppm_updated.csv."""
    stem = Path(filename).stem
    for prefix, phone in CAMERA_PREFIX_TO_PHONE:
        if stem.lower().startswith(prefix.lower()):
            return phone
    raise ValueError(f"Unrecognized camera prefix in filename: {filename}")


def strip_camera_prefix(filename: str) -> str:
    """Remove the leading camera prefix from an image filename stem."""
    stem = Path(filename).stem
    for prefix, _ in CAMERA_PREFIX_TO_PHONE:
        if stem.lower().startswith(prefix.lower()):
            return stem[len(prefix):]
    raise ValueError(f"Unrecognized camera prefix in filename: {filename}")


def parse_sample_id_from_filename(filename: str, known_sample_ids: Iterable[str]) -> str:
    """Extract and validate the canonical sample_id from an image filename.

    Handles both training naming (`<camera>_<sample_id>_<nn>.jpg`) and test naming
    (`<camera>_<sample_id> (<n>).JPG`, optional index, German umlauts, and comma
    decimal separators such as `Münster_BS6_9,0-10m` -> `Muenster_BS6_9_0-10m`).
    """
    remainder = strip_camera_prefix(filename)
    # Strip trailing photo counter: either ` (1)` or `_01`.
    remainder = re.sub(r"\s*\(\d+\)\s*$", "", remainder)
    remainder = re.sub(r"_\d{2}$", "", remainder)
    normalized = normalize_sample_token(remainder)

    lookup = {normalize_sample_token(sid): sid for sid in known_sample_ids}
    if normalized in lookup:
        return lookup[normalized]

    # Case-insensitive fallback if exact normalized key did not match.
    lower_lookup = {k.lower(): v for k, v in lookup.items()}
    if normalized.lower() in lower_lookup:
        return lower_lookup[normalized.lower()]

    raise ValueError(
        f"Could not map filename '{filename}' (parsed '{normalized}') "
        f"to any known sample_id"
    )


def compute_effective_ppm(
    width: int,
    height: int,
    ref_width: int,
    ref_height: int,
    native_ppm: float,
) -> float:
    """Compute effective pixels-per-millimeter accounting for resized images."""
    if width <= 0 or height <= 0 or ref_width <= 0 or ref_height <= 0 or native_ppm <= 0:
        raise ValueError("Image dimensions and native_ppm must be positive")
    scale = max(width, height) / max(ref_width, ref_height)
    return float(native_ppm * scale)


def _find_split_images(data_dir: Path, split: str) -> list[Path]:
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    keyword = "training" if split == "train" else "test"
    images = [
        p
        for p in sorted(data_dir.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and keyword in p.relative_to(data_dir).parts[0].lower()
    ]
    if not images:
        raise FileNotFoundError(f"No {split} images found under {data_dir}")
    return images


def build_image_manifest(data_dir: Path, split: str) -> pd.DataFrame:
    """Build a verified image-to-sample manifest with camera and physical scale metadata."""
    data_dir = Path(data_dir)
    ppm_files = sorted(data_dir.rglob("ppm_updated.csv"))
    if not ppm_files:
        raise FileNotFoundError(f"ppm_updated.csv not found under {data_dir}")
    ppm_df = pd.read_csv(ppm_files[0])
    ppm_by_phone = ppm_df.set_index("phone").to_dict(orient="index")

    csv_name = "Training_labels_updated.csv" if split == "train" else "sample_submission.csv"
    csv_files = sorted(data_dir.rglob(csv_name))
    if not csv_files:
        raise FileNotFoundError(f"{csv_name} not found under {data_dir}")
    known_sample_ids = pd.read_csv(csv_files[0])["sample_id"].tolist()

    image_paths = _find_split_images(data_dir, split)
    rows: list[dict[str, object]] = []
    for path in image_paths:
        phone = parse_camera_from_filename(path.name)
        if phone not in ppm_by_phone:
            raise ValueError(f"Phone '{phone}' missing from ppm_updated.csv")
        sample_id = parse_sample_id_from_filename(path.name, known_sample_ids)

        with Image.open(path) as im:
            width, height = im.size
            orientation = im.getexif().get(274, 1)
            if orientation in {5, 6, 7, 8}:
                width, height = height, width

        cam = ppm_by_phone[phone]
        ref_width = int(cam["width"])
        ref_height = int(cam["height"])
        native_ppm = float(cam["ppm"])
        eff_ppm = compute_effective_ppm(width, height, ref_width, ref_height, native_ppm)

        rows.append(
            {
                "image_path": str(path),
                "filename": path.name,
                "split": split,
                "sample_id": sample_id,
                "phone": phone,
                "width": width,
                "height": height,
                "ref_width": ref_width,
                "ref_height": ref_height,
                "native_ppm": native_ppm,
                "effective_ppm": eff_ppm,
                "fov_short_mm": min(width, height) / eff_ppm,
                "fov_long_mm": max(width, height) / eff_ppm,
            }
        )

    manifest = pd.DataFrame(rows)

    missing_samples = sorted(set(known_sample_ids) - set(manifest["sample_id"]))
    if missing_samples:
        raise ValueError(f"Split '{split}' has samples with zero mapped images: {missing_samples}")

    return manifest


# 15 unique sampling sites for the 24 training soils (Soranzo et al., 2025, Table 1),
# used to prevent site leakage during grouped cross-validation (e.g. 6 Steinakirchen gravels).
SAMPLE_LOCATIONS: dict[str, str] = {
    "F827": "Zistersdorf",
    "G190": "Aichdorf",
    "H030": "Wien",
    "H037": "StCorona",
    "H038": "StCorona",
    "H126": "Wimpassing",
    "H181": "Ybbs",
    "H183": "Ybbs",
    "H366": "Steinakirchen",
    "H367": "Steinakirchen",
    "H368": "Steinakirchen",
    "H371": "Steinakirchen",
    "H372": "Steinakirchen",
    "H374": "Steinakirchen",
    "H405": "Hagsdorf",
    "H493": "Gajuri",
    "H516": "Omding",
    "H549": "Gottsdorf",
    "H615": "Poechlarn",
    "H616": "Simbach",
    "H617": "Simbach",
    "H637": "Krems",
    "H666": "Heiligenkreuz",
    "H668": "Heiligenkreuz",
}


def add_location_column(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the manifest with a `location` column mapped from `SAMPLE_LOCATIONS`."""
    out = manifest.copy()
    out["location"] = out["sample_id"].map(SAMPLE_LOCATIONS)
    return out

