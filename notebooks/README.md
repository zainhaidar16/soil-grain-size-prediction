# Competition notebooks

Run notebooks in numbered order from JupyterLab, VS Code, or Kaggle after uploading the official data.

1. **01_dataset_exploration.ipynb** — inventory images and CSV schemas, preview sample photographs, and explore target curves.
2. **02_baseline_and_validation.ipynb** — verify metric behavior, run sample-level holdout validation, plot the mean distribution, and optionally create a baseline submission.

Notebooks import the reusable implementation from `src/`. Source-of-truth evaluation, model classes, and feature extraction belong in Python modules, not copied between notebooks.

**Do not commit** raw Kaggle images, labels, large model weights, executed cell outputs containing private data, or Kaggle API tokens.
