# HMTL with Calibration - Experimental Report

**Generated:** 2025-12-10T15:16:43.869897

## Executive Summary

This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.

---

## 1. Main HMTL Experiment

### Метрики (валидация / тест)

| Metric | Validation | Test |
|--------|------------|------|
| RMSE | 0.853416 | 0.738786 |
| MAE | 0.652957 | 0.556811 |
| R-AUC MSE | 0.645360 | 0.400096 |
| Mean Uncertainty | 0.980063 | 0.981340 |
| Mean Epistemic | 0.233678 | — |
| Mean Aleatoric | 0.950512 | — |

### Покрытие после конформной калибровки

| Level | Val Coverage | Val Width | Test Coverage | Test Width |
|-------|--------------|-----------|---------------|------------|
| 80% | 79.92% | 2.083622 | 86.31% | 2.083622 |
| 90% | 89.96% | 2.787494 | 94.61% | 2.787493 |
| 95% | 94.98% | 3.494281 | 96.27% | 3.494281 |

### Разложение неопределенности

- Вал. неопределенность: 0.980063 (эпистемическая: 0.233678, алеаторная: 0.950512)
- Тестовая неопределенность: 0.981340
- Средний R-AUC MSE по ансамблю: 0.530452
- Размер ансамбля: 20

### Ключевые графики

![Error-Retention (валидация)](experiments/full_report/main/plots/val_error_retention.png)

![Rejection Curve (валидация)](experiments/full_report/main/plots/val_rejection_curve.png)

![Retention vs Rejection (валидация)](experiments/full_report/main/plots/val_retention_vs_rejection.png)

![Calibration Curve (валидация)](experiments/full_report/main/plots/val_calibration.png)

![Calibration before/after conformal (валидация)](experiments/full_report/main/plots/val_calibration_before_after.png)

![Residual histogram (валидация)](experiments/full_report/main/plots/val_residual_hist.png)

![Residual QQ (валидация)](experiments/full_report/main/plots/val_residual_qq.png)

![Residual vs pred (валидация)](experiments/full_report/main/plots/val_residual_vs_pred.png)

![Residual vs uncertainty (валидация)](experiments/full_report/main/plots/val_residual_vs_uncertainty.png)

![|error| vs uncertainty (валидация)](experiments/full_report/main/plots/val_uncertainty_vs_error.png)

![Uncertainty by error quantile (валидация)](experiments/full_report/main/plots/val_uncertainty_by_error_quantile.png)

![PI width dist @80%](experiments/full_report/main/plots/val_pi_width_80.png)

![PI width dist @90%](experiments/full_report/main/plots/val_pi_width_90.png)

![PI width dist @95%](experiments/full_report/main/plots/val_pi_width_95.png)

![Error-Retention (тест)](experiments/full_report/main/plots/test_error_retention.png)

![Rejection Curve (тест)](experiments/full_report/main/plots/test_rejection_curve.png)

![Retention vs Rejection (тест)](experiments/full_report/main/plots/test_retention_vs_rejection.png)

![Calibration Curve (тест)](experiments/full_report/main/plots/test_calibration.png)

![Calibration before/after conformal (тест)](experiments/full_report/main/plots/test_calibration_before_after.png)

![Residual histogram (тест)](experiments/full_report/main/plots/test_residual_hist.png)

![Residual QQ (тест)](experiments/full_report/main/plots/test_residual_qq.png)

![Residual vs pred (тест)](experiments/full_report/main/plots/test_residual_vs_pred.png)

![Residual vs uncertainty (тест)](experiments/full_report/main/plots/test_residual_vs_uncertainty.png)

![|error| vs uncertainty (тест)](experiments/full_report/main/plots/test_uncertainty_vs_error.png)

![Uncertainty by error quantile (тест)](experiments/full_report/main/plots/test_uncertainty_by_error_quantile.png)

![PI width dist @80% (тест)](experiments/full_report/main/plots/test_pi_width_80.png)

![PI width dist @90% (тест)](experiments/full_report/main/plots/test_pi_width_90.png)

![PI width dist @95% (тест)](experiments/full_report/main/plots/test_pi_width_95.png)

### Кривые обучения ансамбля HMTL

![Training model_10_training_curve.png](experiments/full_report/main/plots/training/model_10_training_curve.png)

![Training model_11_training_curve.png](experiments/full_report/main/plots/training/model_11_training_curve.png)

![Training model_12_training_curve.png](experiments/full_report/main/plots/training/model_12_training_curve.png)

![Training model_13_training_curve.png](experiments/full_report/main/plots/training/model_13_training_curve.png)

![Training model_14_training_curve.png](experiments/full_report/main/plots/training/model_14_training_curve.png)

![Training model_15_training_curve.png](experiments/full_report/main/plots/training/model_15_training_curve.png)

![Training model_16_training_curve.png](experiments/full_report/main/plots/training/model_16_training_curve.png)

![Training model_17_training_curve.png](experiments/full_report/main/plots/training/model_17_training_curve.png)

![Training model_18_training_curve.png](experiments/full_report/main/plots/training/model_18_training_curve.png)

![Training model_19_training_curve.png](experiments/full_report/main/plots/training/model_19_training_curve.png)

![Training model_1_training_curve.png](experiments/full_report/main/plots/training/model_1_training_curve.png)

![Training model_20_training_curve.png](experiments/full_report/main/plots/training/model_20_training_curve.png)

![Training model_2_training_curve.png](experiments/full_report/main/plots/training/model_2_training_curve.png)

![Training model_3_training_curve.png](experiments/full_report/main/plots/training/model_3_training_curve.png)

![Training model_4_training_curve.png](experiments/full_report/main/plots/training/model_4_training_curve.png)

![Training model_5_training_curve.png](experiments/full_report/main/plots/training/model_5_training_curve.png)

![Training model_6_training_curve.png](experiments/full_report/main/plots/training/model_6_training_curve.png)

![Training model_7_training_curve.png](experiments/full_report/main/plots/training/model_7_training_curve.png)

![Training model_8_training_curve.png](experiments/full_report/main/plots/training/model_8_training_curve.png)

![Training model_9_training_curve.png](experiments/full_report/main/plots/training/model_9_training_curve.png)


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

### Покрытие (Mean ± Std)

| Level | Mean | Std |
|-------|------|-----|
| 80% | 80.33% | 0.34% |
| 90% | 90.10% | 0.20% |
| 95% | 95.40% | 0.00% |

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

### Визуализация по сиду

![Multi-seed](experiments/full_report/multi_seed/coverage_distributions.png)

![Multi-seed](experiments/full_report/multi_seed/metric_distributions.png)


## 3. Baseline Comparison

### Comparison Table

| Model      |     RMSE |      MSE |      MAE |   R-AUC MSE |   Mean Uncertainty |   Coverage@80 |   Coverage@90 |   Coverage@95 |   Rejection Ratio (%) |   Rejection AUC |   F-Beta AUC |   F-Beta@95 |   ΔRMSE_vs_HMTL |   ΔR-AUC_vs_HMTL |
|:-----------|---------:|---------:|---------:|------------:|-------------------:|--------------:|--------------:|--------------:|----------------------:|----------------:|-------------:|------------:|----------------:|-----------------:|
| single_mlp | 0.908126 | 0.824692 | 0.692498 |    0.884325 |           0.900968 |      0        |      0        |      0        |              0.511493 |        0.4109   |     0.432062 |    0.666667 |       0.0547093 |        0.238965  |
| flat_mtl   | 0.866033 | 0.750013 | 0.654886 |    0.598752 |           0.960885 |      0        |      0        |      0        |             15.0752   |        0.335945 |     0.481645 |    0.643678 |       0.0126163 |       -0.0466078 |
| hmtl       | 0.853416 | 0.72832  | 0.652957 |    0.64536  |           0.980063 |      0.799163 |      0.899582 |      0.949791 |            nan        |      nan        |   nan        |  nan        |       0         |        0         |

### Метрики неопределенности

- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). 
  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.
- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).
- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.

![Baseline metrics](experiments/full_report/baselines/baseline_comparison.png)

### Best Models

- **RMSE:** hmtl (0.853416)
- **R-AUC MSE:** flat_mtl (0.598752)
- **MAE:** hmtl (0.652957)

_HMTL включен в сравнение базлайнов по умолчанию._


## 4. Summary and Conclusions

### Key Findings

- **Best R-AUC MSE:** flat_mtl (0.598752)
- **Best RMSE:** hmtl (0.853416)
- **HMTL (val)** RMSE 0.853416, R-AUC MSE 0.645360
- **HMTL (test)** RMSE 0.738786, R-AUC MSE 0.400096
- **Conformal coverage@90 (val):** 89.96%
- **HMTL R-AUC MSE (multi-seed):** 0.546289 ± 0.014069
- **HMTL RMSE (multi-seed):** 0.828386 ± 0.005414

### Model Comparison

Based on the baseline comparison:

- HMTL model performance compared to baselines:

Top 3 models by R-AUC MSE:

| Model      |   R-AUC MSE |     RMSE |   Rejection Ratio (%) |
|:-----------|------------:|---------:|----------------------:|
| flat_mtl   |    0.598752 | 0.866033 |             15.0752   |
| hmtl       |    0.64536  | 0.853416 |            nan        |
| single_mlp |    0.884325 | 0.908126 |              0.511493 |

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
