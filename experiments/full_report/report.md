# HMTL with Calibration - Experimental Report

**Generated:** 2025-11-22T18:13:52.633331

## Executive Summary

This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.

---

## 1. Main HMTL Experiment

Main experiment completed successfully.

## 2. Multi-Seed Experiments

**Number of seeds:** 3
**Seeds used:** 42, 43, 44

### Aggregated Metrics (Mean ± Std)

| Metric | Mean | Std |
|--------|------|-----|
| rmse | 0.828386 | 0.005414 |
| mse | 0.686253 | 0.008989 |
| mae | 0.613477 | 0.005107 |
| r_auc_mse | 0.546289 | 0.014069 |
| mean_uncertainty | 0.716020 | 0.022378 |
| mean_epistemic | 0.229149 | 0.011171 |
| mean_aleatoric | 0.674950 | 0.020071 |
| rejection_ratio | 14.268709 | 0.995941 |
| rejection_auc | 0.308238 | 0.004756 |
| f_beta_auc | 0.506610 | 0.008121 |
| f_beta_95 | 0.657088 | 0.007168 |

### Individual Seed Results

**Seed 42:**
- RMSE: 0.823942
- R-AUC MSE: 0.545152

**Seed 43:**
- RMSE: 0.836008
- R-AUC MSE: 0.564060

**Seed 44:**
- RMSE: 0.825210
- R-AUC MSE: 0.529654


## 3. Baseline Comparison

### Comparison Table

| Model      |     RMSE |      MSE |      MAE |   R-AUC MSE |   Mean Uncertainty |   Coverage@80 |   Coverage@90 |   Coverage@95 |   Rejection Ratio (%) |   Rejection AUC |   F-Beta AUC |   F-Beta@95 |
|:-----------|---------:|---------:|---------:|------------:|-------------------:|--------------:|--------------:|--------------:|----------------------:|----------------:|-------------:|------------:|
| single_mlp | 0.842414 | 0.709661 | 0.621531 |    0.579677 |           0.672492 |      0        |      0        |      0        |               10.3868 |        0.3286   |     0.476216 |    0.643678 |
| flat_mtl   | 0.847901 | 0.718936 | 0.638539 |    0.621305 |           0.80741  |      0        |      0        |      0        |               10.6641 |        0.332676 |     0.480304 |    0.672414 |
| catboost   | 0.780725 | 0.609532 | 0.575671 |    0.329735 |           0.136098 |      0.803347 |      0.903766 |      0.953975 |               39.5243 |        0.218431 |     0.581439 |    0.683908 |
| hmtl       | 0.829804 | 0.688575 | 0.619571 |    0.571711 |           0.852208 |      0.803347 |      0.903766 |      0.953975 |               14.0702 |        0.310079 |     0.496684 |    0.66092  |

### Метрики неопределенности

- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). 
  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.
- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).
- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.

### Best Models

- **RMSE:** catboost (0.780725)
- **R-AUC MSE:** catboost (0.329735)
- **MAE:** catboost (0.575671)


## 4. Summary and Conclusions

### Key Findings

- **Best R-AUC MSE:** catboost (0.329735)
- **Best RMSE:** catboost (0.780725)
- **HMTL R-AUC MSE (multi-seed):** 0.546289 ± 0.014069
- **HMTL RMSE (multi-seed):** 0.828386 ± 0.005414

### Model Comparison

Based on the baseline comparison:

- HMTL model performance compared to baselines:

Top 3 models by R-AUC MSE:

| Model      |   R-AUC MSE |     RMSE |   Rejection Ratio (%) |
|:-----------|------------:|---------:|----------------------:|
| catboost   |    0.329735 | 0.780725 |               39.5243 |
| hmtl       |    0.571711 | 0.829804 |               14.0702 |
| single_mlp |    0.579677 | 0.842414 |               10.3868 |

### Recommendations

- Review plots in `experiments/plots/` for detailed visualizations
- Check `experiments/baselines/comparison_table.csv` for full baseline comparison
- Examine `experiments/multi_seed/results.json` for detailed multi-seed statistics
- Analyze error-retention curves to understand uncertainty calibration quality
- Compare coverage metrics before and after conformal calibration

### Output Files

All experiment outputs are saved in the following locations:

- `experiments/plots/` - Visualization plots (error-retention, calibration, reliability)
- `experiments/baselines/` - Baseline comparison results
- `experiments/multi_seed/` - Multi-seed experiment results
- `experiments/runs/` - Individual training runs (if MLflow disabled)
