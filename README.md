# SoilVision AI — Intelligent Soil Analysis from Images

Production-ready computer-vision web application and Kaggle research repository for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview).

**Goal:** Predict cumulative grain-size mass percentages at 11 diameters (`0.002–200 mm`) from calibrated soil sample photographs, with interactive logarithmic grading curves, ISO 14688-1 composition estimates, log-interpolated engineering parameters (`D10–D60`, `Cu`, `Cc`), indicative classification, and downloadable CSV/PDF reports.

## SoilVision AI Web Application

### Architecture

- **Frontend (`web/`):** Next.js App Router (TypeScript strict, Tailwind CSS, Recharts, Lucide icons, React Hook Form + Zod, jsPDF) designed for **Vercel** deployment. Communicates exclusively with Next.js server-side proxy routes (`/api/health`, `/api/model-info`, `/api/predict`) so backend URLs and API keys are never exposed to the browser.
- **ML Inference Backend (`backend/`):** FastAPI service wrapping the exact fitted **`37.32208` Public LB** pipeline (`backend/artifacts/soilvision_v1.joblib`, `267 KB`, verified `0.000000` max EMD diff vs. `submissions/submission.csv` at commit `38100253e14ed13ab61d8acabee43138f30973f7`, and `0.0297` EMD diff end-to-end from raw iPhone 16 JPEGs).
- **Why the Python ML backend runs separately from Vercel Functions:** The real inference pipeline requires PyTorch, `timm` (`vit_small_patch14_dinov2.lvd142m`), OpenCV, SciPy, and scikit-learn (`> 1.2 GB` uncompressed), which exceeds Vercel Python Serverless Functions' `250 MB` bundle limit. Deploy `backend/` via `backend/Dockerfile` (e.g., Render, Railway, Fly.io, or Cloud Run) and connect `web/` on Vercel via `SOILVISION_BACKEND_URL`.

### Local Development

```bash
# 1. Start the Python FastAPI inference backend (from repo root)
source .venv/bin/activate
pip install -r backend/requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000

# 2. Start the Next.js frontend (in a second terminal)
cd web
cp .env.example .env.local
npm install
npm run dev
```

Re-export the fitted model artifact from authorized local competition data:
```bash
python -m backend.export_model
```

### API Endpoints (`backend/app.py`)

| Endpoint | Method | Description |
| --- | --- | --- |
| `/health` | `GET` | Service health, `model_ready` flag, and active `model_version` |
| `/model-info` | `GET` | Architecture metadata, camera presets, upload limits, limitations, and disclaimer |
| `/predict` | `POST` | Multipart upload (`files`: 1–12 JPG/PNG/WebP, `sample_id`, `calibration_mode`, `fov_width_mm` / `effective_ppm` / `camera_preset`, `apply_stone_correction`). Returns versioned `schema_version: "1.0"` JSON. Optional `X-API-Key` header when `SOILVISION_API_KEY` is set. |

### Vercel Deployment (`web/`)

1. Import `zainhaidar16/soil-grain-size-prediction` into Vercel.
2. Set **Root Directory** to `web`, **Framework Preset** to `Next.js`, **Build Command** to `npm run build`.
3. Deploy `backend/` using `backend/Dockerfile` (set `SOILVISION_CORS_ORIGINS` to your Vercel domain and optional `SOILVISION_API_KEY`).
4. Set server-only environment variables in Vercel (never `NEXT_PUBLIC_`):
   - `SOILVISION_BACKEND_URL=https://<your-backend-host>`
   - `SOILVISION_BACKEND_API_KEY=<matching-api-key>`

---

## Kaggle Competition Research Status

- **Previous best-known Kaggle Public LB**: **`68.59422`** (Frozen DINOv2-S RGB + Ridge, Nested LOOCV `45.161`; preserved in git history prior to replacement).
- **Rejected intermediate candidate**: Physical Texture + GrayNorm DINOv2 PLS Hybrid scored `41.062` LOOCV but **`> 76` Public LB** (overpredicted fines on unseen iPhones due to un-filtered shifted features and linear PLS extrapolation).
- **Previous verified Kaggle Public LB**: **`43.66208`** (Option 1 — Improved Blend + Large-Stone Correction; recovery commit `a1b366439fde283b43ad006f33d30abb6e45ba27`).
- **Best verified Kaggle Public LB**: **`37.32208`** (active `submissions/submission.csv`: `SoftRoute(Option 1 → Scale14 + DINO-GN Weibull, s=5)` + unchanged Large-Stone Correction; recovery commit **`38100253e14ed13ab61d8acabee43138f30973f7`**; user-reported Kaggle Public LB). Local `15-Site OOF = 40.10` and `Cross-Phone = 45.85` were worse than Option 1's local results despite improved public LB, indicating test-domain shift and public-leaderboard uncertainty. The seven test sands shift coarser at `0.063–0.63 mm`; Muenster and Lidl stay unchanged. No hard `2.78%` fines override.
- **Latest research candidate (code added, submission NOT overwritten)**: `Unc-Aware SR: Scale14 + Native7 (soil-wt α=20)` adds 7 optically resolved `0.63–4.0 mm` native DoG features and Bayesian leverage + photo-dispersion uncertainty routing, achieving our best local **15-Site OOF = `37.69`** (median `27.43`, `Inter 0.2–0.63 = 11.25`, `Cross-Phone = 42.53`). Held back from overwriting `submissions/submission.csv` pending user review of its silty-shift trade-off on the 7 test sands.

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

### 15-Site Location CV & Cross-Camera Benchmark (Reference Reproduction, Ablations & Improvements)

Using the true **15-site location grouping** (Soranzo et al., 2025 — preventing leakage across the 6 Steinakirchen gravels and other paired sites) and **site-grouped Cross-Phone CV** (`Motorola Edge` ↔ `Samsung A52`), we faithfully reproduced the reference notebook *"Photos to Sieve Curves: The Fines Question"* (test curve diff `0.01–0.03` mean EMD vs. `pick1`/`pick2`) and ran controlled ablations and architectural improvements:

| Candidate / Variant | 15-Site OOF EMD | Median | Sand (11) | Gravel (13) | Cross-Phone EMD | Notes / Reported Ref. LB |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Ref `GRAVEL_MODEL`: Logit-PCA (`cs`, 29d) | `40.33` | `31.43` | `40.55` | `40.14` | `46.81` | Common-scale JPEG q75 |
| Ref `GRAVEL_MODEL`: Logit-PCA (`tl`, 29d) | `39.62` | `33.27` | `40.19` | `39.14` | `44.52` | Train-like JPEG improves both CV & cross-phone |
| Ref `SAND_MODEL_2`: Weibull (`tl`, 29d) | `39.51` | `33.06` | `44.58` | `35.23` | `45.39` | `cs` ablation worsens Cross-Phone to `53.35` (`+7.96`) |
| Ref `SAND_MODEL_1`: Weibull (`b5` + ResNet-50) | `47.19` | `43.09` | `52.80` | `42.44` | `57.66` | Lower predicted fines on test sands (`11–24%`) |
| Ref Hard Routed `Pick 2` (`cs` → `S2`) | `41.86` | `33.52` | `44.58` | `39.55` | `47.32` | Reported Ref. Public LB: `26.89` (with SAM cobble step) |
| Ref Hard Routed `Pick 1` (`cs` → `S1`) | `44.22` | `33.34` | `52.80` | `36.96` | `58.46` | Reported Ref. Public LB: `23.17` (with SAM cobble step) |
| Ref `Pick 3` (`Pick 1` + `2.78%` Clean-Sand Fines) | `51.36` | `37.77` | `65.78` | `39.15` | `64.94` | Reported Ref. Public LB: `19.42515` (`+7.14` worse on local OOF) |
| Our `DINOv2-S` LKM (`LogQ11`) | `40.80` | `29.91` | `39.49` | `41.91` | `41.55` | Strongest single-model cross-camera stability |
| Our `E-A` (`DINO-LKM` + `PhysV2-KRR`) | `42.05` | `31.74` | `40.40` | `43.45` | `45.76` | Active `submissions/submission.csv` (no cobble step) |
| **New Sand**: Weibull (`b5` + `DINOv2-GN`) | `47.17` | `42.96` | `52.80` | `42.42` | `54.20` | Beats ResNet-50 by `-3.46` Cross-Phone EMD |
| **Improved Blend**: `LogitPCA(tl)` + `Weibull(tl)` + `DINO-LKM` | **`37.83`** | **`28.16`** | **`39.53`** | **`36.39`** | **`41.81`** | **Best overall 15-site OOF & Cross-Phone** (+ cobble step on test) |
| **Soft-Routed Hybrid**: `Blend` → `b5+DINO-GN` (`s=5`) | `40.06` | `29.70` | `48.38` | `33.02` | `46.46` | Beats Ref `Pick 1` by `-4.16` OOF and `-12.00` Cross-Phone |

**Key ablation takeaways:**
1. **Unseen cobbles (`Muenster`, `Lidl WHV`)** are the primary cause of the `~70 → ~27` LB gap: no training soil contains cobbles (`F(20 mm) = 76.5–100%` in train), whereas physical coarse-stone area correction shifts `Muenster` by `73.4` EMD and `Lidl WHV` by `45.5` EMD.
2. **Train-like JPEG/scale normalization (`tl`)** cuts Cross-Phone EMD by `-7.96` on `SAND_MODEL_2` (`53.35 → 45.39`) and `-2.29` on `GRAVEL_MODEL` (`46.81 → 44.52`).
3. **Calibrated soft sand/gravel routing** (`sigmoid` around `F(2 mm) = 90%`) outperforms hard routing on both 15-site OOF (`-1.48` to `-2.22` EMD) and Cross-Phone CV (`-1.39` to `-4.55` EMD).
4. **Clean-sand fines correction (`2.78%` at `0.063 mm`)** is an assumption-dependent prior: at `4.5 px/mm`, a `0.063 mm` silt grain is `0.28 px` wide (below optical resolution), and 10 of 11 sand-like training soils are silty (`35–90%` passing `0.063 mm`). Applying it blindly hurts local 15-site OOF by `+7.14` EMD (helping only `H616` by `-13.89` while hurting all 10 silty sands by `+7` to `+23` EMD). Data-driven `F(0.063 mm)` regression also predicts `~40%` for `H616` due to this label imbalance.

### Controlled Fines-Reduction Experiments (Option 1 Base, Active Public LB `43.66208`)

Applying the identical `F(2 mm) >= 90%` sand-routing rule inside 15-site Location CV, Cross-Phone CV, and on the 7 sand-like test soils (preserving Option 1's large-stone correction unchanged):

| Candidate | 15-Site OOF | Median | Sand (11) | Gravel (13) | Cross-Phone | Test Sand `max(0.063→0.2)` | Shift vs. Sub | OOF Improved / Worsened |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Option 1: **0% (No correction)** *(Public LB `43.66208`)* | **`37.83`** | **`28.16`** | **`39.53`** | `36.39` | `41.81` | `33.2%` *(realistic)* | `0.00` | Baseline |
| Option 1 + **25% fines reduction** | `38.22` | `28.25` | `41.87` | `35.14` | **`41.71`** | `42.3%` *(within train max `45.5%`)* | `5.21` | `5 / 8` |
| Option 1 + **50% fines reduction** | `40.38` | `28.25` | `47.00` | **`34.78`** | `43.83` | `51.4%` *(exceeds train max)* | `10.41` | `5 / 8` |
| Option 1 + **Clean-sand `2.78%`** | `46.77` | `34.25` | `58.35` | `36.97` | `52.20` | `66.9%` *(unrealistic cliff)* | `19.22` | `2 / 11` |
| Soft-Routed (`Opt1 → b5+DINO-GN`, `s=5`) + **0%** | `40.54` | `29.70` | `49.62` | `32.87` | `47.25` | `30.9%` *(realistic)* | `18.31` | — |
| Soft-Routed (`Opt1 → b5+DINO-GN`, `s=5`) + **50%** | `43.74` | `30.11` | `55.52` | `33.78` | `50.90` | `44.2%` *(within train max `45.5%`)* | `24.77` | — |

### Large-Stone Physics Validation, Calibration Sensitivity & Feature-Shift Audit

- **Validated on 24 training soils (`>= 6.3 mm` & `>= 20 mm`)**: `9/24` training soils have `>= 10%` mass retained above `20 mm` (max `23.5%` on `H374`), and `12/24` have `>= 20%` above `6.3 mm` (max `49.3%`). Boundary-contrast-gated CPU stone detection ([`estimate_coarse_stone_area`](file:///home/antigravity/zainhaidar16/soil-grain-size-prediction/src/features.py)) correlates with 3D lab mass at `r = +0.644` (`>= 6.3 mm`) and `r = +0.598` (`>= 20 mm`, MAE `4.72%`, implied stereology `k = 1.38`), while eliminating false positives on uniform sands (`Kleinkummerfeld = 0.000`, `Airbus <= 0.015`).
- **Unverifiable test assumptions**: True cobbles (`>= 63 mm`) are nearly absent in train (max `7.4%`), 2D surface area depends strongly on segmentation method (`Muenster` `A20 = 0.215` watershed vs `0.655` SAM), and stereology factor `k` (`±0.2`) swings `Muenster` by `±19.7` EMD and `Lidl WHV` by `±10.5` EMD. Camera calibration error (`effective_ppm × 0.8–1.2`) shifts `Muenster` by only `±3%` (`0.210–0.224`) but `Lidl WHV` by `±12%` (`0.118–0.149`).
- **Sand/gravel feature-shift audit (`tl` 29d)**: `Lab` color (`tl_b_mean` `1.52 σ`, `tl_L_mean` `2.05 σ` train→test) and fine DoG (`0.25–0.5 mm`, `1.17–1.36 σ`) shift most across cameras. Dropping `Lab` improves Cross-Phone CV (`41.81 → 40.46` with 4-way blend) but worsens 15-site OOF (`37.83 → 39.72`), so Option 1 (`37.83` OOF / `41.81` Cross-Phone) remains the preserved active submission.

### Next research gate (do not change active submission)

- **Hypothesis:** The main remaining disagreement with the reference `19.42515` Public LB notebook lies in sand-range `0.063–0.63 mm` predictions, especially the seven Airbus/Kleinkummerfeld samples; the observed disagreement is **not** ground-truth error because competition test labels are hidden.
- **Test one constrained sand-expert experiment:** Compare the active Scale14 soft-routed prediction with per-sample, physically calibrated native-resolution DoG grain-size statistics; estimate a continuous, uncertainty-aware adjustment to the sand expert at `0.2` and `0.63 mm`, while fitting all calibration coefficients and model-selection decisions *inside* 15-site held-out training folds. Do not use the reference notebook's test predictions as labels.
- **Validation:** Report per-soil full-curve EMD, sand-specific contributions near `0.2/0.63 mm`, cross-phone EMD, bootstrap uncertainty by site, and the entire test-vs-current prediction delta. Confirm cumulative `[0,100]` curves and retain the unchanged large-stone correction for Muenster/Lidl.
- **Decision:** Prefer a new public-LB test only if a physically defensible candidate emerges; document that local CV may not rank models correctly under train/test domain shift. Preserve `submissions/submission.csv` until an experimental submission is explicitly approved. Never claim to achieve `0.00000` without hidden-label evidence.

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

Only **one** submission file is maintained in the repository: `submissions/submission.csv` (currently the best verified **`37.32208` Public LB** version, recovery commit `38100253e14ed13ab61d8acabee43138f30973f7`). Restore this file from that exact commit if an experimental submission performs worse. Git history preserves the earlier `43.66208` version at `a1b366439fde283b43ad006f33d30abb6e45ba27`. Do not overwrite the active CSV without explicit approval.

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
  - The production submission now combines train-like calibrated texture and DINOv2-based methods, a Scale14 sand specialist, soft routing, and a large-stone correction. Training soils are disproportionately silty relative to the observed test-sand texture; the 37.32208 public-LB gain despite worse grouped CV highlights distribution shift, not confirmed private-set generalization.

## Research and integrity

Never check in raw contest images, test labels, Kaggle tokens, downloaded model binaries, or unlicensed data. Follow [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules) for external data, pretrained weights, and sharing.

