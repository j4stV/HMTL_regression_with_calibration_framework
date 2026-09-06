# HMTL with calibration

A research framework for tabular prediction with hierarchical multi-task learning, ensemble uncertainty estimation, and conformal calibration. Developed as part of my bachelor's thesis work, it studies whether auxiliary supervision at an intermediate network layer improves prediction and uncertainty ranking. The main focus is regression; the code also includes classification experiments.

## Research question

A useful regression model should estimate both a target and the uncertainty of its prediction. This project compares a hierarchical neural model with a single MLP, flat multi-task learning, and CatBoost, examining prediction error, error-retention curves, and prediction-interval coverage.

The saved experiments show mixed results. They support investigating when auxiliary tasks help; they do not establish general superiority over gradient boosting.

## Method

1. **Preprocessing.** Training-fitted imputation and scaling, optional feature binning, PCA, and target encoding. Targets can be standardized for regression.
2. **Hierarchical model.** A lower encoder feeds an auxiliary head; an upper encoder feeds the main prediction head. The encoder uses SELU, LeCun-style initialization, AlphaDropout, and optional residual connections.
3. **Auxiliary supervision.** Target-bin classification or supervised contrastive learning. The implementation also includes reconstruction, ranking, combinations of auxiliary losses, and a pilot-based selector.
4. **Regression objective.** A mean and positive scale are learned with Gaussian negative log-likelihood, together with the weighted auxiliary loss. An optional quantile head supports conformalized quantile regression (CQR).
5. **Ensembling.** Models are trained with different seeds and configurable resampling. The runners evaluate prediction error and how well uncertainty ranks errors.

```text
Tabular features → lower encoder → upper encoder → mean, scale
                         │                    └──→ optional quantile head
                         └──→ auxiliary head

Ensemble predictions → uncertainty decomposition → evaluation and calibration
```

The architecture is implemented in [src/models/hmtl.py](src/models/hmtl.py). Configuration files are the source of truth: the current research defaults use a 128-wide, 18-layer model, a 20-model ensemble, and up to 1,000 epochs. Use the small configuration below for an installation check.

## Uncertainty and calibration

For regression, the ensemble reports the mean prediction and combines the variance of model means with the mean predicted variance:

```text
mean_prediction = mean(mu_m)
variance_total  = variance(mu_m) + mean(sigma_m ** 2)
```

The implementation uses this decomposition as an estimate of model and observation uncertainty. It evaluates RMSE, MAE, R-AUC MSE (area under the error-retention curve), interval coverage, and interval width. Lower R-AUC MSE is better, but it reflects both prediction error and uncertainty ordering.

Two interval methods are implemented:

- **Symmetric residual calibration:** constant-width intervals around the ensemble mean, fitted on absolute calibration residuals. These intervals do not use the predicted scale to vary width across observations.
- **CQR:** adjusts intervals produced by the quantile head using calibration nonconformity scores.

For classification, the framework averages class probabilities, decomposes predictive entropy, and constructs prediction sets using `1 - p(y_true | x)` scores.

**Calibration caveats:** the default `cal_csv: null` reuses validation data for calibration, even though validation also guides training. For a separate assessment, provide a calibration split unused by model selection and evaluate on an untouched test split. The symmetric regression implementation uses an empirical residual quantile without the finite-sample correction; the CQR and classification implementations clip their quantile level for small samples. The current code should therefore not be presented as providing an unconditional finite-sample coverage guarantee. See [conformal.py](src/eval/conformal.py), [cqr.py](src/eval/cqr.py), and the [conformal prediction reference](https://arxiv.org/abs/2107.07511).

## Saved experimental evidence

### Regression benchmark

The committed [aggregated results](experiments/automl_5sizes_v4_4workers_v2/aggregated_results.json) contain 23 datasets from OpenML study 269, five training-size fractions, and one requested seed (`42`) per dataset. At the full training fraction, all 23 datasets have paired HMTL and CatBoost results:

| Metric, lower is better | HMTL lower | CatBoost lower | Ties |
| --- | ---: | ---: | ---: |
| RMSE | 9 | 14 | 0 |
| R-AUC MSE | 7 | 16 | 0 |

These are descriptive counts from the stored run, not a significance test or a new benchmark of the current code. Smaller fractions have incomplete runs. The adjacent `summary_by_size.csv` describes fewer datasets than the JSON, so the table above is calculated directly from the JSON:

```bash
python scripts/summarize_saved_benchmark.py
```

The saved run predates later model changes, and its metadata names configuration paths rather than preserving an immutable environment and configuration snapshot. Exact reproduction of its numbers is not established by the current defaults.

### Prediction intervals

The [saved Superconductor report](experiments/full_report_superconductor/report.md), generated on 10 December 2025, records the following test-set coverage after symmetric calibration:

| Nominal coverage | Recorded test coverage |
| --- | ---: |
| 80% | 81.10% |
| 90% | 89.99% |
| 95% | 95.16% |

Source: [results.json](experiments/full_report_superconductor/results.json), `main_experiment.metrics`. These are recorded measurements from one experiment, subject to the split and calibration caveats above.

## Installation

Use Python 3.11 or newer in a virtual environment. Run commands from the repository root.

```bash
git clone https://github.com/j4stV/HMTL_regression_with_calibration_framework.git
cd HMTL_regression_with_calibration_framework
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. CatBoost and MLflow are optional:

```bash
python -m pip install -r requirements-baselines.txt  # CatBoost comparisons
python -m pip install "mlflow>=2.16"                 # Experiment tracking
```

For servers or other environments without a display, set `MPLBACKEND=Agg` before running commands that generate plots. The examples below use POSIX shell syntax.

## Quick start

The repository includes Wine Quality CSV splits in `data/`. This small run checks training, ensemble prediction, evaluation, and plotting without downloading a dataset:

```bash
MPLBACKEND=Agg python scripts/main.py \
  --data configs/data.yaml \
  --model configs/model_smoke.yaml \
  --train configs/train_smoke.yaml \
  --ensemble configs/ensemble_smoke.yaml
```

It trains two small models for four epochs. Its metrics are an installation check, not a research result. Metrics are printed to the console; plots are written under `experiments/plots/`, and repeated runs can replace plots there.

For a full run using the current research settings:

```bash
MPLBACKEND=Agg python scripts/main.py
```

Superconductor CSV splits are also committed; no preparation script is required:

```bash
MPLBACKEND=Agg python scripts/main.py --data configs/data_superconductor.yaml
```

To use your own data, copy [configs/data.yaml](configs/data.yaml), set the CSV paths and target column, and supply separate training, validation, calibration, and test files. Fit preprocessing only on training data. The standalone runner and benchmark runner do not use identical adaptation rules; treat their configurations and results separately.

## Experiments and tests

The benchmark runner supports paired datasets, several training fractions, explicit seeds, and optional baseline models. Start with one dataset before launching a full study:

```bash
MPLBACKEND=Agg python scripts/run_automlbenchmark_experiment.py \
  --dataset-id 287 --sizes 1.0 --seeds 42 \
  --baselines catboost single_mlp flat_mtl \
  --output experiments/new_comparison
```

This command downloads OpenML data and trains with the current configuration; it does not recreate the historical table above. The repository also includes runners for multiple seeds, ablations, classification, and report generation. Use each script's `--help` for its arguments.

Run the existing unit and integration tests in headless mode:

```bash
MPLBACKEND=Agg python -m pytest -q tests
```

Tests cover preprocessing, target encoding, model behavior, ensemble aggregation, calibration, early stopping, numerical stability, mixed precision, and runner configuration. Device-dependent tests may be skipped when the required hardware is unavailable.

## Repository layout

| Path | Purpose |
| --- | --- |
| [configs/](configs/) | Data, model, training, ensemble, and small-run configurations |
| [src/data/](src/data/) | Tabular preprocessing and OpenML loading |
| [src/models/](src/models/) | Encoders and task/auxiliary heads |
| [src/train/](src/train/) | Training loop, optimizers, ensembles, and auxiliary-task selection |
| [src/eval/](src/eval/) | Metrics, uncertainty, calibration, and plots |
| [src/baselines/](src/baselines/) | Single MLP, flat MTL, and CatBoost |
| [src/tasks/](src/tasks/) | Regression and classification interfaces |
| [scripts/](scripts/) | Experiment runners and result analysis |
| [tests/](tests/) | Unit and integration tests |
| [experiments/](experiments/) | Historical reports and saved results |
| [data/](data/) | Included dataset splits and dataset metadata |

## Research context and remaining work

This repository grew out of bachelor's thesis research and subsequent coursework at Novosibirsk State University. The Russian practice reports are historical context, not a peer-reviewed publication or the final thesis text.

The next research steps are to separate model selection from calibration throughout the runners, validate finite-sample behavior, freeze benchmark environments and configurations, and repeat comparisons across seeds with consistent compute budgets. The code is a research prototype; it is not packaged as a production prediction service. No repository license file is currently included.
