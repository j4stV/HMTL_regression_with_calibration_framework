# HMTL with Calibration - Experimental Report

**Generated:** 2025-12-10T16:09:34.655610

## Executive Summary

This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.

---

## 1. Main HMTL Experiment

### Метрики (валидация / тест)

| Metric | Validation | Test |
|--------|------------|------|
| RMSE | 0.296722 | 0.294180 |
| MAE | 0.190745 | 0.185180 |
| R-AUC MSE | 0.024995 | 0.024434 |
| Mean Uncertainty | 0.392680 | 0.389794 |
| Mean Epistemic | 0.154717 | — |
| Mean Aleatoric | 0.356158 | — |

### Покрытие после конформной калибровки

| Level | Val Coverage | Val Width | Test Coverage | Test Width |
|-------|--------------|-----------|---------------|------------|
| 80% | 79.96% | 0.587196 | 81.10% | 0.587196 |
| 90% | 89.98% | 0.905961 | 89.99% | 0.905961 |
| 95% | 95.01% | 1.319216 | 95.16% | 1.319216 |

### Разложение неопределенности

- Вал. неопределенность: 0.392680 (эпистемическая: 0.154717, алеаторная: 0.356158)
- Тестовая неопределенность: 0.389794
- Средний R-AUC MSE по ансамблю: 0.055765
- Размер ансамбля: 20

### Ключевые графики

![Error-Retention (валидация)](main/plots/val_error_retention.png)

![Rejection Curve (валидация)](main/plots/val_rejection_curve.png)

![Retention vs Rejection (валидация)](main/plots/val_retention_vs_rejection.png)

![Calibration Curve (валидация)](main/plots/val_calibration.png)

![Calibration before/after conformal (валидация)](main/plots/val_calibration_before_after.png)

![Residual histogram (валидация)](main/plots/val_residual_hist.png)

![Residual QQ (валидация)](main/plots/val_residual_qq.png)

![Residual vs pred (валидация)](main/plots/val_residual_vs_pred.png)

![Residual vs uncertainty (валидация)](main/plots/val_residual_vs_uncertainty.png)

![|error| vs uncertainty (валидация)](main/plots/val_uncertainty_vs_error.png)

![Uncertainty by error quantile (валидация)](main/plots/val_uncertainty_by_error_quantile.png)

![PI width dist @80%](main/plots/val_pi_width_80.png)

![PI width dist @90%](main/plots/val_pi_width_90.png)

![PI width dist @95%](main/plots/val_pi_width_95.png)

![Error-Retention (тест)](main/plots/test_error_retention.png)

![Rejection Curve (тест)](main/plots/test_rejection_curve.png)

![Retention vs Rejection (тест)](main/plots/test_retention_vs_rejection.png)

![Calibration Curve (тест)](main/plots/test_calibration.png)

![Calibration before/after conformal (тест)](main/plots/test_calibration_before_after.png)

![Residual histogram (тест)](main/plots/test_residual_hist.png)

![Residual QQ (тест)](main/plots/test_residual_qq.png)

![Residual vs pred (тест)](main/plots/test_residual_vs_pred.png)

![Residual vs uncertainty (тест)](main/plots/test_residual_vs_uncertainty.png)

![|error| vs uncertainty (тест)](main/plots/test_uncertainty_vs_error.png)

![Uncertainty by error quantile (тест)](main/plots/test_uncertainty_by_error_quantile.png)

![PI width dist @80% (тест)](main/plots/test_pi_width_80.png)

![PI width dist @90% (тест)](main/plots/test_pi_width_90.png)

![PI width dist @95% (тест)](main/plots/test_pi_width_95.png)

### Кривые обучения ансамбля HMTL

![Training model_10_training_curve.png](main/plots/training/model_10_training_curve.png)

![Training model_11_training_curve.png](main/plots/training/model_11_training_curve.png)

![Training model_12_training_curve.png](main/plots/training/model_12_training_curve.png)

![Training model_13_training_curve.png](main/plots/training/model_13_training_curve.png)

![Training model_14_training_curve.png](main/plots/training/model_14_training_curve.png)

![Training model_15_training_curve.png](main/plots/training/model_15_training_curve.png)

![Training model_16_training_curve.png](main/plots/training/model_16_training_curve.png)

![Training model_17_training_curve.png](main/plots/training/model_17_training_curve.png)

![Training model_18_training_curve.png](main/plots/training/model_18_training_curve.png)

![Training model_19_training_curve.png](main/plots/training/model_19_training_curve.png)

![Training model_1_training_curve.png](main/plots/training/model_1_training_curve.png)

![Training model_20_training_curve.png](main/plots/training/model_20_training_curve.png)

![Training model_2_training_curve.png](main/plots/training/model_2_training_curve.png)

![Training model_3_training_curve.png](main/plots/training/model_3_training_curve.png)

![Training model_4_training_curve.png](main/plots/training/model_4_training_curve.png)

![Training model_5_training_curve.png](main/plots/training/model_5_training_curve.png)

![Training model_6_training_curve.png](main/plots/training/model_6_training_curve.png)

![Training model_7_training_curve.png](main/plots/training/model_7_training_curve.png)

![Training model_8_training_curve.png](main/plots/training/model_8_training_curve.png)

![Training model_9_training_curve.png](main/plots/training/model_9_training_curve.png)


## 2. Multi-Seed Experiments

**Number of seeds:** 3
**Seeds used:** 42, 43, 44

### Aggregated Metrics (Mean ± Std)

| Metric | Mean | Std |
|--------|------|-----|
| rmse | 46.524107 | 0.000000 |
| mse | 2164.492538 | 0.000000 |
| mae | 34.794618 | 0.000000 |
| r_auc_mse | 617.815537 | 0.000000 |
| mean_uncertainty | 12.860828 | 0.000000 |
| mean_epistemic | 5.109567 | 0.000000 |
| mean_aleatoric | 11.614145 | 0.000000 |
| rejection_ratio | 80.336850 | 0.000000 |
| rejection_auc | 496.694539 | 0.000000 |
| f_beta_auc | 0.696427 | 0.000000 |
| f_beta_95 | 0.689588 | 0.000000 |

### Покрытие (Mean ± Std)

| Level | Mean | Std |
|-------|------|-----|
| 80% | 0.00% | 0.00% |
| 90% | 0.00% | 0.00% |
| 95% | 0.00% | 0.00% |

### Individual Seed Results

**Seed 43:**
- RMSE: 46.524107
- R-AUC MSE: 617.815537

### Визуализация по сиду

![Multi-seed](multi_seed/coverage_distributions.png)

![Multi-seed](multi_seed/error_retention_seeds.png)

![Multi-seed](multi_seed/metric_distributions.png)


## 3. Baseline Comparison

### Comparison Table

| Model      |     RMSE |       MSE |      MAE |   R-AUC MSE |   Mean Uncertainty |   Coverage@80 |   Coverage@90 |   Coverage@95 |   Rejection Ratio (%) |   Rejection AUC |   F-Beta AUC |   F-Beta@95 |   ΔRMSE_vs_HMTL |   ΔR-AUC_vs_HMTL |
|:-----------|---------:|----------:|---------:|------------:|-------------------:|--------------:|--------------:|--------------:|----------------------:|----------------:|-------------:|------------:|----------------:|-----------------:|
| single_mlp | 0.354199 | 0.125457  | 0.228164 |   0.0346055 |           0.383213 |      0        |      0        |      0        |               73.2344 |       0.0258702 |     0.637956 |    0.685047 |       0.0574768 |       0.00961055 |
| flat_mtl   | 0.34636  | 0.119965  | 0.219221 |   0.029481  |           0.306764 |      0        |      0        |      0        |               75.8167 |       0.0230137 |     0.643888 |    0.682452 |       0.049638  |       0.0044861  |
| catboost   | 0.391341 | 0.153148  | 0.268451 |   0.0420738 |           0.160143 |      0.800094 |      0.900282 |      0.950141 |               77.4471 |       0.0308446 |     0.635318 |    0.684398 |       0.094619  |       0.0170788  |
| hmtl       | 0.296722 | 0.0880438 | 0.190745 |   0.0249949 |           0.39268  |      0.799624 |      0.899812 |      0.950141 |              nan      |     nan         |   nan        |  nan        |       0         |       0          |

### Метрики неопределенности

- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). 
  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.
- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).
- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.

![Baseline metrics](baselines/baseline_comparison.png)

### Best Models

- **RMSE:** hmtl (0.296722)
- **R-AUC MSE:** hmtl (0.024995)
- **MAE:** hmtl (0.190745)

_HMTL включен в сравнение базлайнов по умолчанию._


## 4. Summary and Conclusions

### Key Findings

- **Best R-AUC MSE:** hmtl (0.024995)
- **Best RMSE:** hmtl (0.296722)
- **HMTL (val)** RMSE 0.296722, R-AUC MSE 0.024995
- **HMTL (test)** RMSE 0.294180, R-AUC MSE 0.024434
- **Conformal coverage@90 (val):** 89.98%
- **HMTL R-AUC MSE (multi-seed):** 617.815537 ± 0.000000
- **HMTL RMSE (multi-seed):** 46.524107 ± 0.000000

### Model Comparison

Based on the baseline comparison:

- HMTL model performance compared to baselines:

Top 3 models by R-AUC MSE:

| Model      |   R-AUC MSE |     RMSE |   Rejection Ratio (%) |
|:-----------|------------:|---------:|----------------------:|
| hmtl       |   0.0249949 | 0.296722 |              nan      |
| flat_mtl   |   0.029481  | 0.34636  |               75.8167 |
| single_mlp |   0.0346055 | 0.354199 |               73.2344 |

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
