# Soil Grain Size Prediction

Competition-focused machine learning project for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview).

**Goal:** Predict cumulative grain-size mass percentages at 11 diameters from soil sample photographs. The final competition ranking is determined by private-leaderboard log-weighted Earth Mover's Distance (lower is better).

## Status

**Stage 1: Hybrid notebook + Python baseline scaffold.** The code currently implements an image-independent mean-curve baseline, rigorous per-sample EMD evaluation, physical output checks, and a submission writer. It does **not** yet use soil photographs or represent a competitive trained model.

## Repository layout

```text
notebooks/
  01_dataset_exploration.ipynb  # CSV/image audits and visualizations
  02_baseline_and_validation.ipynb # Sample-level EMD baseline
  README.md
configs/
  experiment.json     # Experiment defaults and validation policy
src/
  __init__.py
  metric.py            # Log-diameter weighted EMD + valid curves
  baseline.py          # Leave-one-soil-sample-out baseline and submission
tests/
  test_metric.py       # Scoring and structural tests
requirements.txt
.gitignore             # Keeps raw data and trained models out of Git
```

## Quick start

Python 3.10+ recommended.

```bash
git clone https://github.com/zainhaidar16/soil-grain-size-prediction.git
cd soil-grain-size-prediction
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
jupyter lab  # install jupyterlab separately if needed, or open .ipynb in VS Code
```

Get the competition dataset through Kaggle according to its rules. Place CSV files in your local `data/` folder; do **not** commit the dataset or credentials.

Once you have verified the actual CSV filenames and schema:

```bash
python -m src.baseline --labels data/Training_labels_updated.csv
python -m src.baseline --labels data/Training_labels_updated.csv --submission-template data/sample_submission.csv --output submissions/mean_baseline.csv
```

The baseline expects a `sample_id` column and the 11 target columns `0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20, 63, 200`. If the real training CSV uses different identifiers or formatting, **inspect and adapt the loader** before running; the schema has not yet been validated against downloaded data.

## Notebook workflow

Open the notebooks in order:

1. `notebooks/01_dataset_exploration.ipynb` — inspect all local CSVs, preview photographs, check labels and metadata; deliberately does not assume mapping between images and soil samples.
2. `notebooks/02_baseline_and_validation.ipynb` — validate the weighted-EMD calculation, evaluate a leave-one-soil-sample-out mean curve, and generate a baseline submission only when the official CSV files are available.

Both notebooks import reusable functions from `src/`; notebook cells are unexecuted and no competition scores are claimed. **Run notebooks with the repository as working directory or from `notebooks/`** so their project-root discovery works.

You can install `jupyterlab` separately (`pip install jupyterlab`) if you do not use VS Code or Kaggle's notebook environment.

## Validation and scoring

- **Never split photographs randomly.** Each soil sample can have several photos; splitting photos permits sample leakage. Hold out entire soil samples.
- The starter baseline uses leave-one-sample-out validation, predicting each held-out sample from the mean training curve of the other samples.
- The scoring implementation computes the sum of absolute cumulative percentage differences weighted by `log10(diameter[i+1] / diameter[i])` across the ten adjacent intervals. Verify implementation against the official metric if source code becomes available.
- A valid submission contains exactly 11 nondecreasing percentages in `[0, 100]` for each test ID; the last value must be 100.
- The visible public leaderboard is **not** the final score; prioritize robust cross-validation.

## Planned roadmap

1. Audit real dataset: sample count, image-to-sample IDs, resolutions, scales (PPM), photographic views and missing values.
2. Implement group-safe preprocessing and pretrained frozen image embeddings; aggregate multiple photos per sample.
3. Evaluate regularized regressors (Ridge, PLS, nearest neighbors) using the same holdout protocol.
4. Add calibrated physical texture/particle-scale features and optional robust model ensembles.
5. Validate constraints, reproducibility, and test submission before uploading to Kaggle.

## Research and integrity

Never check in raw contest images, test labels, Kaggle tokens, downloaded model binaries, or unlicensed data. Follow [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules) for external data, pretrained weights, and sharing. No leaderboard results have been achieved or claimed by this repository yet.
