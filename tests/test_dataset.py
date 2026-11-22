import numpy as np
import pytest
from PIL import Image

from src.dataset import (
    compute_effective_ppm,
    parse_camera_from_filename,
    parse_sample_id_from_filename,
)
from src.features import aggregate_sample_features, crop_physical_square


@pytest.mark.parametrize(
    ("filename", "expected_phone"),
    [
        ("Motorola_Edge_60_fusion_H374_01.jpg", "Motorola Edge 60 Fusion"),
        ("Motorola_Edge_F827_01.jpg", "Motorola Edge"),
        ("Samsung_A52_H668_04.jpg", "Samsung A52"),
        ("iPhone14_HPC_Audorfring (1).JPG", "iPhone 14"),
        ("iPhone16_HPC_Airbus BS6-3 (2).JPG", "iPhone 16"),
        ("iPhone_16_HPC_Airbus BS10-4bis7 (4).JPG", "iPhone 16"),
    ],
)
def test_parse_camera_from_filename(filename: str, expected_phone: str):
    assert parse_camera_from_filename(filename) == expected_phone


def test_parse_camera_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="Unrecognized camera prefix"):
        parse_camera_from_filename("Pixel8_H030_01.jpg")


@pytest.mark.parametrize(
    ("filename", "expected_sample_id"),
    [
        ("Motorola_Edge_60_fusion_H374_01.jpg", "H374"),
        ("Samsung_A52_F827_03.jpg", "F827"),
        ("iPhone14_HPC_Münster_BS6_9,0-10m (5).JPG", "HPC_Muenster_BS6_9_0-10m"),
        ("iPhone16_HPC_Kleinkummerfeld 9-4.JPG", "HPC_Kleinkummerfeld 9-4"),
        ("iPhone_16_HPC_Airbus BS10-4bis7 (4).JPG", "HPC_Airbus BS10-4bis7"),
    ],
)
def test_parse_sample_id_from_filename(filename: str, expected_sample_id: str):
    known = [
        "H374",
        "F827",
        "HPC_Muenster_BS6_9_0-10m",
        "HPC_Kleinkummerfeld 9-4",
        "HPC_Airbus BS10-4bis7",
    ]
    assert parse_sample_id_from_filename(filename, known) == expected_sample_id


def test_parse_sample_id_rejects_unmatched():
    with pytest.raises(ValueError, match="Could not map filename"):
        parse_sample_id_from_filename("Samsung_A52_UNKNOWN_01.jpg", ["F827", "G190"])


def test_compute_effective_ppm():
    # Resized Motorola Edge (1600x720 vs 4000x1800 native)
    assert compute_effective_ppm(1600, 720, 4000, 1800, 11.492) == pytest.approx(11.492 * 0.4)
    # Portrait orientation uses max(w, h)
    assert compute_effective_ppm(720, 1600, 4000, 1800, 11.492) == pytest.approx(11.492 * 0.4)
    # Native iPhone 16
    assert compute_effective_ppm(5712, 4284, 5712, 4284, 19.525) == pytest.approx(19.525)


def test_crop_physical_square_and_clamping():
    im = Image.new("RGB", (1000, 600), color=(128, 128, 128))
    # 100mm * 4 px/mm = 400px square
    cropped = crop_physical_square(im, effective_ppm=4.0, crop_mm=100.0)
    assert cropped.size == (400, 400)
    # 200mm * 4 px/mm = 800px -> clamped to min(1000, 600) = 600px
    clamped = crop_physical_square(im, effective_ppm=4.0, crop_mm=200.0)
    assert clamped.size == (600, 600)


def test_aggregate_sample_features_order_and_missing():
    feats = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    agg = aggregate_sample_features(feats, ["A", "A", "B"], ["B", "A"], agg="mean")
    assert agg.shape == (2, 2)
    assert np.allclose(agg[0], [0.0, 1.0])
    assert np.allclose(agg[1], [1.0, 0.0])

    with pytest.raises(ValueError, match="No image features found"):
        aggregate_sample_features(feats, ["A", "A", "B"], ["A", "C"])
