# Soil Grain Size Prediction

Competition-focused machine learning project for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview).

**Goal:** Predict cumulative grain-size mass percentages at 11 diameters from soil sample photographs. The final competition ranking is determined by private-leaderboard log-weighted Earth Mover's Distance (lower is better).

## Status

**Best-known public submission:** Milestone 3 DINOv2-S RGB + Ridge, saved as `submissions/submission.csv` (public EMD `68.59422`, rank `602` at submission time). The Milestone 4 hybrid improved local CV but worsened the Kaggle public EMD to **above 76** (rank worse than 700). Its separate CSV was removed; its model remains available only as a research comparison. Better local validation is not proof of better unseen-camera generalization.

### Validation results (24 training samples, strictly leakage-free, lower is better)

| Model | Nested LOOCV EMD | Cross-Camera EMD (21 dual-cam) | Train-vs-Test AdvAUC | Public Kaggle EMD | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Mean-curve baseline (`src/baseline.py`) | `89.888867` | — | — | — | Image-independent baseline |
| Milestone 3: DINOv2-S RGB + Ridge (`submissions/submission.csv`) | `45.160960` | `48.552` (`M→S 53.86`, `S→M 43.24`) | `0.975` | `68.59422` | High color/contrast domain shift |
| PhysTex-L2 150mm + Ridge (`B1-L2`) | `43.252` | `47.872` (`M→S 53.61`, `S→M 42.14`) | **`0.640`** | — | Camera-invariant 51-dim FFT/DoG/grad |
| GrayNorm DINOv2-S + PLS (`A2-PLS`, `k=2`) | `41.498` | `46.270` (`M→S 47.50`, `S→M 45.04`) | `0.935` | — | Discards camera-noise directions |
| **Milestone 4: Hybrid 50/50 (`B1-L2` + `A2-PLS`)** (research experiment) | **`41.062`** | **`45.1` (`M→S ~47.3`, `S→M ~42.6`)** | `0.640` / `0.935` | `>76` (worse) | Local CV improved but test generalization worsened |
| *Rejected*: PhysTex 3-crop + SVR-RBF (`C=100`) | `32.133` | `56.627` (`M→S 56.75`, `S→M 56.50`) | `0.560` | — | **Overfit** on Cross-Camera CV (`+24.5` gap) |

## Repository layout

```text
notebooks/
  01_dataset_exploration.ipynb     # CSV/image audits and visualizations
  02_baseline_and_validation.ipynb # Sample-level EMD mean-curve baseline
  03_dinov2_ridge_model.ipynb      # Milestone 3: DINOv2 + LOOCV Ridge experiment
  README.md
configs/
  experiment.json                  # Experiment defaults and validation policy
src/
  __init__.py
  metric.py                        # Log-diameter weighted EMD ([0, 500]) + curve validation
  baseline.py                      # Leave-one-soil-sample-out mean baseline and submission
  dataset.py                       # Image-to-sample mapping and effective PPM metadata
  features.py                      # EXIF/PPM physical crops, GrayNorm DINOv2, FFT/DoG texture
  model.py                         # Leakage-free LOOCV Ridge, PLS, hybrid ensemble & CLI
submissions/
  mean_baseline.csv                # Mean-curve baseline predictions (10 test samples)
  submission.csv                   # Milestone 3 DINOv2-S RGB + Ridge (Public LB: 68.59422)
tests/
  test_metric.py                   # EMD normalization, Wasserstein-1 equivalence, schema tests
  test_dataset.py                  # Filename/sample mapping, PPM, cropping, physical texture tests
  test_model.py                    # LOOCV leakage, PLS, hybrid ensemble, submission tests
requirements.txt
.gitignore                         # Keeps raw competition data, model binaries, and secrets out of Git
```

## Dataset

The [Kaggle competition dataset](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview) contains **162 soil photographs** and **3 CSV files** (`Training_labels_updated.csv` with 24 soil samples, `sample_submission.csv` with 10 test samples, and `ppm_updated.csv` with camera calibration metadata for 5 phones).

> **Why raw competition data is not committed:** This repository is public, and Kaggle competition rules restrict public redistribution of competition data (no permissive redistribution license is included with the dataset). Pretrained DINOv2 weights (`vit_small_patch14_dinov2.lvd142m`) are downloaded automatically at runtime by `timm`.

## Quick start and training process

Python 3.10+ recommended.

```bash
git clone https://github.com/zainhaidar16/soil-grain-size-prediction.git
cd soil-grain-size-prediction
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Join the [Kaggle competition](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview) and accept its rules first. Both the CLI and notebooks resolve data via `SOIL_DATA_DIR`, local `data/`, the local `kagglehub` cache, or `kagglehub.competition_download("soil-grain-size-from-photos")`. Outside Kaggle, set `KAGGLE_API_TOKEN` in your local environment (never commit tokens or `.env` files).

Run any of the three reproducible pipelines:

```bash
# Mean-curve baseline
python -m src.baseline --labels data/Training_labels_updated.csv --submission-template data/sample_submission.csv --output submissions/mean_baseline.csv

# Milestone 3 (preserved): Frozen DINOv2-S RGB + nested LOOCV Ridge
python -m src.model --mode ridge --output submissions/submission.csv

# Milestone 4 (recommended): Hybrid PhysTex-L2 Ridge + GrayNorm DINOv2-S PLS
```

The CSV schema expects `sample_id` and the 11 target columns `0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, 200` (verified against `Training_labels_updated.csv` and `sample_submission.csv`).

## Submitting to Kaggle


## Notebook workflow

1. `notebooks/01_dataset_exploration.ipynb` — inspect all local CSVs, preview photographs, and inspect target grading curves.
2. `notebooks/02_baseline_and_validation.ipynb` — validate the weighted-EMD calculation and evaluate the leave-one-soil-sample-out mean curve (`89.888867`).
3. `notebooks/03_dinov2_ridge_model.ipynb` — audit the 162 image-to-sample mappings and camera PPMs, extract frozen DINOv2 multi-scale physical-crop embeddings, evaluate leakage-free nested LOOCV Ridge (`45.160960`), plot per-sample diagnostics, and generate `submissions/submission.csv`.

**Run notebooks with the repository as working directory or from `notebooks/`** so their project-root discovery works.

## Validation, metric, and limitations

- **Never split photographs randomly.** Each soil sample has 3–8 photographs; all evaluations hold out entire soil samples (leave-one-sample-out CV) and fit scalers/hyperparameters strictly inside each training fold.
- **Verified competition metric normalization:** `src/metric.py` computes the unnormalized log10-diameter weighted sum of absolute cumulative percentage differences across the 10 adjacent intervals, taking values in `[0, 500]` (`LOG_WEIGHTS.sum() == 5.0`), which is mathematically identical to `100 * scipy.stats.wasserstein_distance` on `log10(DIAMETERS)`.
- **Physical-scale (PPM) normalization:** Most training photographs (`Motorola Edge` at `1600x720`, `Samsung A52` at `1599x1200`) are downscaled relative to `ppm_updated.csv` (`effective_ppm ~ 4.55–4.60 px/mm`), whereas test photographs (`iPhone 14`, `iPhone 16`) are at native resolution (`13.94–19.53 px/mm`). Cropping fixed physical squares (`90 mm` and `150 mm`) normalizes apparent grain scale across cameras.
- **Known limitations:**
  - Very small labeled dataset (`n = 24` soil samples), so high variance is expected on unseen geological sites.
  - Camera domain shift remains between training phones (Android) and test phones (iPhones) in lighting, sharpness, and color rendition despite physical-scale cropping.
  - Only frozen DINOv2-S embeddings + linear Ridge regression are used so far; fine-grained particle-size texture features, multi-backbone ensembles, and EMD-optimal monotonic projection are left for future milestones.

## Research and integrity

Never check in raw contest images, test labels, Kaggle tokens, downloaded model binaries, or unlicensed data. Follow [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules) for external data, pretrained weights, and sharing.


## Submission file policy

Use **only** `submissions/submission.csv` as the final competition submission artifact. Do not create numbered, milestone-specific, or alternate submission CSV files. Experiment outputs should stay in memory or in ignored temporary scratch storage. When an experiment has been validated and explicitly selected for submission, overwrite `submissions/submission.csv` with the exact verified official-template row/column order; run physical-curve and missing-value checks before writing. Preserve old reproducibility through Git commits instead of duplicate CSV files. The best-known public submission currently remains Milestone 3. Never commit Kaggle tokens or credentials.
