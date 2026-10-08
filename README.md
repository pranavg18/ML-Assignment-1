# Polynomial Regression Assignment

**Name:** Pranav Goyal

**Roll Number:** BT2024086

## Overview

This repository contains the complete implementation used for the polynomial regression assignment (two personalized datasets, `var1` and `var2`).

The program:

- uses only polynomial regression: all monomials up to a chosen total degree (including interactions) with linear coefficients
- searches polynomial degree, input subset, estimator and penalties with nested cross-validation
- evaluates OLS, Ridge, Lasso, Elastic Net, Adaptive Lasso, Relaxed Lasso and degree-weighted Ridge
- standardizes polynomial terms using the training fold only
- selects models by mean inner-fold MSE; outer folds are never used to choose the final model
- refits the selected model on all 1,000 labelled rows
- saves a portable JSON model and writes the two prediction CSV files

The implementation is CPU-only and does not require a GPU.

## Repository Structure

```text
ML-Assignment-1/
├── results/
│   ├── var1/model.json  # final fitted var1 model
│   └── var2/model.json  # final fitted var2 model
├── tests/
│   └── test_workflow.py
├── README.md
├── config.json          # all search ranges, penalties, seeds and solver settings
├── polynomial.py        # monomial expansion, data loading, JSON inference
├── predict.py           # regenerates predictions from a saved model.json
├── requirements.txt
├── run_all.py           # trains var1 and var2 concurrently
├── train.py             # nested-CV search, final fit, predictions, summaries
└── data/                # NOT included: place the four BT2024086 CSVs here
```

The two `model.json` files are the final fitted models, so predictions can be regenerated without training. Running `train.py` or `run_all.py` recreates `results/` (overwriting these files), `submission/` (prediction CSVs) and `logs/`.

## Requirements

- Python 3.11 or newer (the recorded run used Python 3.14)
- NumPy, SciPy, scikit-learn, pandas, threadpoolctl

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Input Files

The course datasets are not redistributed. Place the four files for BT2024086 in `data/`:

```text
data/BT2024086_train_var1.csv
data/BT2024086_test_var1.csv
data/BT2024086_train_var2.csv
data/BT2024086_test_var2.csv
```

Training files must have columns `x1..xp, y` (var1: six inputs, var2: three inputs). Test files must have the same input columns. Input files are never modified.

## Running the Program

```bash
python run_all.py
```

`run_all.py` trains the two problems as separate CPU jobs and writes progress to `logs/train_var1.log` and `logs/train_var2.log`. Training performs thousands of fits and can take a long time on a laptop. It always starts fresh and does not resume checkpoints.

Alternatives:

```bash
python train.py                        # both problems, sequentially
python train.py --problems 1           # only var1 (or 2)
python train.py --output experiments/new_run/results   # isolated run; CSVs go to experiments/new_run/submission
```

Do not run two jobs for the same problem into the same output folder.

To regenerate predictions from the included models without any retraining (after placing the test CSVs in `data/`):

```bash
python predict.py --model results/var1/model.json --test data/BT2024086_test_var1.csv --output submission/BT2024086_pred_var1.csv
python predict.py --model results/var2/model.json --test data/BT2024086_test_var2.csv --output submission/BT2024086_pred_var2.csv
```

## Output Files

```text
submission/BT2024086_pred_var1.csv     # one column: y, 1000 rows
submission/BT2024086_pred_var2.csv     # one column: y, 1000 rows
results/var1/                          # var2 has the same layout
  manifest.json                        # data/code hashes, configuration, versions
  outer_1/ ... outer_5/                # search.json / search.csv for each outer fold
  outer_folds.json                     # selected model and held-out metrics per fold
  oof_predictions.csv                  # out-of-fold predictions
  final/search.json, search.csv        # full-data model selection
  model.json                           # monomial exponents, scalers, coefficients, intercept
  summary.json                         # final selection and measured scores
```

Predictions keep the test-row order and are neither clipped nor rounded. No index column is written.

## Methodology

### 1. Polynomial expansion

For `p` inputs and degree `d`, the expansion contains every nonconstant monomial of total degree at most `d` (so `x1^2 * x2` has degree 3). A separate unpenalized intercept is fitted. Inputs already lie in [-1, 1] and are not rescaled before expansion. Each monomial column is then standardized using the current training fold only. The fitted model is still a polynomial in the original inputs.

### 2. Nested cross-validation

- **Outer evaluation:** 5 shuffled folds (seed 87).
- **Inner selection:** 3 shuffled folds (seed 86), run inside each outer training set. The configuration with the lowest mean inner MSE wins; an exact tie favors the lower degree, then fewer inputs.
- **Final model:** the same inner search on all labelled rows, refitted on all 1,000 rows.

Reported out-of-fold (OOF) scores come from outer folds only. Inner scores are tuning scores, not generalization estimates.

### 3. Estimators

- **OLS / Ridge:** standard least squares and L2 penalty.
- **Lasso / Elastic Net:** L1 and mixed L1/L2 penalties (Elastic Net ratio 0.2 or 0.8).
- **Adaptive Lasso:** an initial Lasso gives weights `w = max(|beta0|, 0.01)^gamma`; a second Lasso is fitted on the reweighted terms.
- **Relaxed Lasso:** Lasso selects terms, Ridge (alpha 0.1) refits them, and a fraction of the refit is blended in.
- **Degree-weighted Ridge (var1):** a term of degree `k` is penalized by `(k/d)^(2q)`, so high-degree terms are shrunk more (q = 1 or 2).

### 4. Search space (see `config.json`)

| Problem | Inputs | OLS/Ridge degrees | Sparse-model degrees |
|---|---|---|---|
| var1 | x1-x3 | 1-10 | 1-5 |
| var1 | all six | 1-7 | 1-5 |
| var2 | x1 | 1-20 | 1-6 |
| var2 | all three | 1-20 | 1-12 |

| Parameter | Values |
|---|---|
| Ridge alpha (var1) | 0 (OLS), 1e-4, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100 |
| Sparse alpha (var1) | 0.001, 0.003, 0.007, 0.01, 0.02, 0.03, 0.1 |
| Ridge alpha (var2) | 0 (OLS), 1e-6, 1e-4, 0.01, 0.1, 1, 10, 100 |
| Sparse alpha (var2) | 0.0001, 0.001, 0.01, 0.1 |
| Elastic Net L1 ratio | 0.2, 0.8 (Lasso uses 1) |
| Adaptive Lasso | initial alpha 0.003, 0.01; gamma 0.5, 1; second-stage alpha 0.0003, 0.001, 0.003, 0.01 |
| Relaxed Lasso | initial alpha 0.01, 0.02, 0.03; refit blend 0.5, 1 |
| Degree-weighted Ridge (var1, 3 inputs, degree >= 6) | q = 1, 2 |

Adaptive and Relaxed Lasso are tried for the six-input sparse search. Var2 compares OLS, Ridge, Lasso and Elastic Net using the settings in `problem_settings["2"]`, which override the shared configuration for that problem. All degrees respect the assignment limits (var1 <= 10, var2 <= 20). The search is finite; it is not proven to find the global optimum.

### 5. Convergence and speed

Sparse solvers use tolerance 1e-5. Var1 uses a 100,000-iteration limit and a warm-start retry of up to 300,000 iterations; var2 uses 30,000 iterations without a retry. Any candidate with a remaining convergence warning is excluded, and a nonconvergent final fit raises an error. Ridge SVDs are reused across penalties, and sparse paths use Gram precomputation and warm starts, only within the same fold and degree, so no fitted statistics are shared between training and validation rows. Each job uses one BLAS thread.

## Final Results

Selected models (final all-data search) and measured outer-fold scores:

| Problem | Degree | Final model | Parameters | Nonzero / total terms | OOF MSE | OOF R2 |
|---|---:|---|---|---:|---:|---:|
| var1 | 5 | Adaptive Lasso | alpha = 0.001, alpha0 = 0.003, gamma = 1 | 56 / 461 | 0.313727 | 0.971849 |
| var2 | 12 | Elastic Net | alpha = 0.001, L1 ratio = 0.8 | 178 / 454 | 0.292238 | 0.992450 |

## Notes

- Fold standard deviations are descriptive, not confidence intervals. Random folds assume independently sampled rows.
