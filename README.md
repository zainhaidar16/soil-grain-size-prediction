# Soil Grain Size Prediction

Competition-focused machine learning project for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview).

**Goal:** Predict cumulative grain-size mass percentages at 11 diameters from soil sample photographs. The final competition ranking is determined by private-leaderboard log-weighted Earth Mover's Distance (lower is better).

## Status

- **Previous best-known Kaggle Public LB**: **`68.59422`** (Frozen DINOv2-S RGB + Ridge, Nested LOOCV `45.161`; preserved in git history prior to replacement).
- **Rejected intermediate candidate**: Physical Texture + GrayNorm DINOv2 PLS Hybrid scored `41.062` LOOCV but **`> 76` Public LB** (overpredicted fines on unseen iPhones due to un-filtered shifted features and linear PLS extrapolation).
- **Active submission (`submissions/submission.csv`, Experiment `E-A`, `--mode logq_ensemble`)**: Equal 50/50 ensemble in **log-grain-size quantile space (`LogQ11`: D5–D95)** combining **33 camera-robust flat-fielded photogranulometry features + winsorized RBF Kernel Ridge (`C3`)** and **DINOv2-S RGB + 1-component PLS Latent-Kernel-Median (`D3`)**. Leakage audit passed (strict in-fold LOOCV **`37.24`**, zero-test-access in-fold LOOCV **`38.13`**, label-perturbation diff **`0.0`**, Cross-Camera **`37.82`**, 6-Group Series **`42.12`**, paired bootstrap win rate **`100.0%`** over 2,000 resamples vs. Baseline `A`, 95% CI `[-13.39, -2.97]`).

### Research findings informing this design

1. **GRAI3 & GRAI4 (Soranzo, BOKU University, 2025–2026)** — [GRAI4 paper (PDF)](https://deepsoil.at/grai4.pdf), [Data in Brief (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12152572/), [deepsoil.at](https://www.deepsoil.at/):
   - Competition images were captured in a standardized photo-box at ~210 mm stand-off; camera PPM normalization to a common physical grid (`~4.6 px/mm`) is essential for device-agnostic inference.
   - **GRAI3** regressed a 2-parameter Weibull curve (`EfficientNetV2-S`), whereas **GRAI4** showed that unimodal Weibull introduces structural bias on well-graded/bimodal soils (confirmed locally: Weibull approximation floor = `7.63` EMD vs. `12.33` for Log-Normal) and moved to direct Wasserstein-1 (EMD) optimization over the 11 classes.
2. **Log-quantile EMD duality & camera-robust photogranulometry** — [vitalclick/Predicting-Soil-Grain-Size](https://github.com/vitalclick/Predicting-Soil-Grain-Size) *(all rights reserved; methodology studied, no code copied)*:
   - By 1D Wasserstein-1 duality, $\text{EMD} = \int_0^{100} |\log_{10} Q(p) - \log_{10} \hat{Q}(p)| \, dp$. Regressing log-diameters at fixed percentiles aligns directly with the competition metric, naturally handles coarse soils, and makes the **weighted median** in quantile space the L1-optimal point estimate.
   - Unsupervised train-vs-test per-feature shift auditing catches camera-fragile descriptors (e.g., morphological closings shifted `4.5–9.8 σ` on iPhone test photos) before they corrupt test predictions.

### Controlled Experiments A–F (24 training samples, strictly leakage-free, lower is better)

| Experiment | Features → Target (Model) | Nested LOOCV [95% CI] | Cross-Camera (21) | 6-Group Series | Coarse (8) | Hard (4) | AdvAUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | Mean training curve | `89.89` | — | — | — | — | — |
| **A (Current LB `68.59`)** | DINOv2-S RGB → 11 CDF % (Ridge) | `45.22 [33.0, 58.9]` | `48.48 (53.5/43.5)` | `50.61` | `19.3` | `100.1` | `0.97` |
| B1 | PhysV2-Rob (33d) → 11 CDF % (Ridge) | `44.31 [33.8, 55.7]` | `48.15 (51.0/45.3)` | `51.21` | `25.5` | `90.7` | `0.96` |
| B2 | PhysV2-Rob (33d) → 11 CDF % (PLS) | `45.64 [34.6, 57.8]` | `51.13 (59.4/42.9)` | `51.49` | `30.7` | `90.5` | `0.96` |
| C1 | PhysV2-Rob (33d) → LogQ11 (Ridge) | `41.10 [31.1, 52.4]` | `48.58 (50.7/46.4)` | `48.72` | `27.0` | `88.4` | `0.96` |
| **C3** | PhysV2-Rob (33d) → LogQ11 (KRR-RBF) | `39.82 [28.9, 52.6]` | `41.63 (42.4/40.8)` | `47.36` | `19.9` | `95.8` | `0.96` |
| C4 | PhysV2-Rob (33d) → LogQ11 (LKM) | `41.30 [29.5, 56.1]` | `43.41 (44.6/42.3)` | `42.66` | `17.1` | `94.2` | `0.96` |
| C5 | PhysV2-Rob (33d) → LogQ5 (PLS) | `49.80 [38.7, 61.7]` | `51.02 (56.2/45.9)` | `51.49` | `35.3` | `99.0` | `0.96` |
| D1 | DINOv2-S RGB → LogQ11 (Ridge) | `44.55 [33.7, 57.4]` | `45.72 (48.9/42.6)` | `49.00` | `22.4` | `95.1` | `0.97` |
| D2 | DINOv2-S RGB → LogQ11 (PLS) | `40.11 [29.3, 53.2]` | `41.09 (42.0/40.1)` | `43.04` | `19.8` | `91.2` | `0.97` |
| **D3** | DINOv2-S RGB → LogQ11 (LKM) | `37.79 [26.5, 52.3]` | `38.59 (38.7/38.5)` | `42.20` | `15.9` | `94.7` | `0.97` |
| **E-A (Recommended)** | **50/50 LogQ11: C3 (PhysV2-KRR) + D3 (DINO-LKM)** | **`37.21 [26.5, 50.9]`** | **`37.82 (38.0/37.7)`** | **`42.12`** | **`17.1`** | **`94.8`** | — |
| F1 | PhysV2-Rob (33d) → Weibull (PLS) | `43.02 [32.6, 55.2]` | `49.81 (53.1/46.5)` | `45.14` | `27.7` | `99.7` | `0.96` |
| F2 | DINOv2-S RGB → Weibull (Ridge) | `43.45 [32.4, 56.6]` | `45.51 (47.5/43.5)` | `48.69` | `20.2` | `94.0` | `0.97` |

## Repository layout

```text
notebooks/
  01_dataset_exploration.ipynb     # CSV/image audits and visualizations
  02_baseline_and_validation.ipynb # Sample-level EMD mean-curve baseline
  03_dinov2_ridge_model.ipynb      # Main DINOv2 experiment notebook
  README.md
configs/
  experiment.json                  # Experiment defaults and validation policy
src/
  __init__.py
  metric.py                        # Exact EMD ([0, 500]), log-quantile & Weibull transforms
  baseline.py                      # Leave-one-soil-sample-out mean baseline
  dataset.py                       # Image-to-sample mapping and effective PPM metadata
  features.py                      # EXIF/PPM crops, frozen DINOv2, 59d photogranulometry, shift filter
  model.py                         # LOOCV Ridge, PLS, winsorized KRR-RBF, LKM, logq_ensemble & CLI
submissions/
  submission.csv                   # Single active submission (currently best-known LB 68.59422)
tests/
  test_metric.py                   # EMD, Wasserstein-1, log-quantile round-trip, Weibull tests
  test_dataset.py                  # Mapping, PPM, cropping, 59d physical texture, shift filter tests
  test_model.py                    # LOOCV leakage, PLS, KRR, LKM, log-quantile ensemble tests
requirements.txt
.gitignore                         # Keeps raw competition data, model binaries, and secrets out of Git
```

## Dataset

The [Kaggle competition dataset](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview) contains **162 soil photographs** and **3 CSV files** (`Training_labels_updated.csv` with 24 soil samples, `sample_submission.csv` with 10 test samples, and `ppm_updated.csv` with camera calibration metadata for 5 phones).

> **Why raw competition data is not committed:** This repository is public, and Kaggle competition rules restrict public redistribution of competition data (no permissive redistribution license is included with the dataset). Pretrained DINOv2 weights (`vit_small_patch14_dinov2.lvd142m`) are downloaded automatically at runtime by `timm`.

## Quick start and reproduction

```bash
git clone https://github.com/zainhaidar16/soil-grain-size-prediction.git
cd soil-grain-size-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest -q
```

Set `KAGGLE_API_TOKEN` in your local environment (never commit tokens or `.env` files) or point `SOIL_DATA_DIR` to the extracted dataset.

```bash
# Dry-run the current best-known LB model (DINOv2-S RGB + Ridge)
python -m src.model --mode ridge

# Dry-run the recommended Log-Quantile Ensemble (E-A: PhysV2-Rob KRR + DINOv2-S LKM)
python -m src.model --mode logq_ensemble

# Only overwrite the single active submission file once approved:
python -m src.model --mode logq_ensemble --write-submission --output submissions/submission.csv
```

## Submitting to Kaggle

Only **one** submission file is maintained in the repository: `submissions/submission.csv` (currently preserving our best-known `68.59422` public LB submission until replacement is approved). Git history tracks prior versions.

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

