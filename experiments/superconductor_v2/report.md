# HMTL with Calibration - Experimental Report

**Generated:** 2026-02-21T13:05:34.541573

## Executive Summary

This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.

---

## 1. Main HMTL Experiment

### Метрики (валидация / тест)

| Metric | Validation | Test |
|--------|------------|------|
| RMSE | 0.306821 | 0.301728 |
| MAE | 0.205135 | 0.198332 |
| R-AUC MSE | 0.028482 | 0.027995 |
| Mean Uncertainty | 0.446407 | 0.443310 |
| Mean Epistemic | 0.183148 | — |
| Mean Aleatoric | 0.399124 | — |

### Покрытие после конформной калибровки

| Level | Val Coverage | Val Width | Test Coverage | Test Width |
|-------|--------------|-----------|---------------|------------|
| 80% | 79.96% | 0.634603 | 80.87% | 0.634603 |
| 90% | 89.98% | 0.948448 | 90.08% | 0.948448 |
| 95% | 95.01% | 1.348350 | 95.20% | 1.348350 |

### Разложение неопределенности

- Вал. неопределенность: 0.446407 (эпистемическая: 0.183148, алеаторная: 0.399124)
- Тестовая неопределенность: 0.443310
- Средний R-AUC MSE по ансамблю: 0.081764
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

Multi-seed experiment results not found.

## 3. Baseline Comparison

### Comparison Table

| Model | RMSE | MSE | MAE | R-AUC MSE | Mean Uncertainty | Coverage@80 | Coverage@90 | Coverage@95 | Rejection Ratio (%) | Rejection AUC | F-Beta AUC | F-Beta@95 | ΔRMSE_vs_HMTL | ΔR-AUC_vs_HMTL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_mlp | 1.421784 | 2.021469 | 1.198342 | 1.815385 | 1.289584 | 0.00 | 0.00 | 0.00 | 25.94 | 0.865434 | 0.492212 | 0.689588 | 1.114962 | 1.786903 |
| flat_mtl | 0.357087 | 0.127511 | 0.221953 | 0.030484 | 0.291089 | 0.00 | 0.00 | 0.00 | 75.93 | 0.023841 | 0.635315 | 0.683750 | 0.050266 | 0.002002 |
| catboost | 0.391341 | 0.153148 | 0.268451 | 0.042070 | 0.160143 | 0.80 | 0.90 | 0.95 | 77.45 | 0.030845 | 0.635318 | 0.684398 | 0.084520 | 0.013588 |
| hmtl | 0.306821 | 0.094139 | 0.205135 | 0.028482 | 0.446407 | 0.80 | 0.90 | 0.95 | 70.91 | 0.020696 | 0.616888 | 0.681155 | 0.000000 | 0.000000 |

### Метрики неопределенности

- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). 
  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.
- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).
- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.

![Baseline metrics](baselines/baseline_comparison.png)

![Δ vs HMTL](baselines/baseline_delta_vs_hmtl.png)

### Best Models

- **RMSE:** hmtl (0.306821)
- **R-AUC MSE:** hmtl (0.028482)
- **MAE:** hmtl (0.205135)

_HMTL включен в сравнение базлайнов по умолчанию._


## 4. Summary and Conclusions

### Key Findings

- **Best R-AUC MSE:** hmtl (0.028482)
- **Best RMSE:** hmtl (0.306821)
- **HMTL (val)** RMSE 0.306821, R-AUC MSE 0.028482
- **HMTL (test)** RMSE 0.301728, R-AUC MSE 0.027995
- **Conformal coverage@90 (val):** 89.98%

### Model Comparison

Based on the baseline comparison:

- HMTL model performance compared to baselines:

Top 3 models by R-AUC MSE:

| Model | R-AUC MSE | RMSE | Rejection Ratio (%) |
|---|---|---|---|
| hmtl | 0.028482 | 0.306821 | 70.91 |
| flat_mtl | 0.030484 | 0.357087 | 75.93 |
| catboost | 0.042070 | 0.391341 | 77.45 |

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
