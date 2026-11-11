"""Training-data-mean baseline, evaluated by holding out entire soil samples."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.metric import TARGET_COLUMNS, mean_emd, project_curves, validate_curves


def read_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in ["sample_id", *TARGET_COLUMNS] if c not in df]
    if missing:
        raise ValueError(f"Missing columns: {missing}; verify Kaggle's actual CSV schema")
    if df["sample_id"].isna().any() or df["sample_id"].duplicated().any():
        raise ValueError("Expected unique sample_id values")
    validate_curves(df[TARGET_COLUMNS].to_numpy(dtype=float))
    return df


def loocv_mean_baseline(df: pd.DataFrame) -> float:
    """Leave out a complete soil sample (not one of its photographs)."""
    y = df[TARGET_COLUMNS].to_numpy(dtype=float)
    if len(y) < 2:
        raise ValueError("At least two labeled samples required")
    predictions = np.vstack([
        project_curves(y[np.arange(len(y)) != i].mean(axis=0))[0]
        for i in range(len(y))
    ])
    return mean_emd(y, predictions)


def write_submission(labels: pd.DataFrame, template: Path, output: Path) -> None:
    sample = pd.read_csv(template)
    if "sample_id" not in sample or sample["sample_id"].isna().any():
        raise ValueError("Invalid sample_submission.csv")
    if sample["sample_id"].duplicated().any():
        raise ValueError("Duplicate test sample IDs")
    curve = project_curves(labels[TARGET_COLUMNS].to_numpy(dtype=float).mean(axis=0))[0]
    predictions = pd.DataFrame(np.tile(curve, (len(sample), 1)), columns=TARGET_COLUMNS)
    predictions.insert(0, "sample_id", sample["sample_id"].to_numpy())
    if set(sample.columns) != set(predictions.columns):
        raise ValueError("Unexpected submission columns; inspect official template")
    predictions = predictions[sample.columns]
    validate_curves(predictions[TARGET_COLUMNS].to_numpy())
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False)
    print(f"Saved {len(predictions)} predictions: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--submission-template", type=Path)
    parser.add_argument("--output", type=Path, default=Path("submissions/mean_baseline.csv"))
    args = parser.parse_args()
    labels = read_labels(args.labels)
    print(f"Labeled samples: {len(labels)}")
    print(f"Leave-one-sample-out EMD: {loocv_mean_baseline(labels):.6f}")
    if args.submission_template is not None:
        write_submission(labels, args.submission_template, args.output)


if __name__ == "__main__":
    main()
