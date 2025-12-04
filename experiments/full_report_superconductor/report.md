# HMTL with Calibration - Experimental Report

**Generated:** 2025-12-04T01:25:22.646532

## Executive Summary

This report summarizes the results of comprehensive experiments on Hierarchical Multi-Task Learning (HMTL) with uncertainty estimation and conformal calibration for tabular regression.

---

## 1. Main HMTL Experiment

Main experiment completed successfully.

## 2. Multi-Seed Experiments

Multi-seed experiment results not found.

## 3. Baseline Comparison

### Comparison Table

| Model      |     RMSE |      MSE |      MAE |   R-AUC MSE |   Mean Uncertainty |   Coverage@80 |   Coverage@90 |   Coverage@95 |   Rejection Ratio (%) |   Rejection AUC |   F-Beta AUC |   F-Beta@95 |
|:-----------|---------:|---------:|---------:|------------:|-------------------:|--------------:|--------------:|--------------:|----------------------:|----------------:|-------------:|------------:|
| single_mlp | 0.624757 | 0.390321 | 0.454548 |   0.139274  |           0.431527 |      0        |      0        |      0        |               70.8277 |       0.0954365 |     0.600179 |    0.682452 |
| flat_mtl   | 0.679207 | 0.461322 | 0.49301  |   0.173443  |           0.219089 |      0        |      0        |      0        |               70.2491 |       0.11354   |     0.593155 |    0.680506 |
| catboost   | 0.343151 | 0.117752 | 0.22361  |   0.0300009 |           0.106093 |      0.800094 |      0.900282 |      0.950141 |               77.5558 |       0.0223266 |     0.629712 |    0.684398 |
| hmtl       | 0.507136 | 0.257187 | 0.349343 |   0.0821325 |          24.5982   |      0.799624 |      0.900282 |      0.949671 |               68.788  |       0.0610957 |     0.623187 |    0.677911 |

### Метрики неопределенности

- **Rejection Ratio**: Нормализованная метрика качества неопределенности (0-100%). 
  Чем выше значение, тем лучше модель ранжирует ошибки по неопределенности.
- **Rejection AUC**: Площадь под кривой отбрасывания (rejection curve).
- **F-Beta AUC**: Площадь под кривой F-beta для оценки качества неопределенности.

### Best Models

- **RMSE:** catboost (0.343151)
- **R-AUC MSE:** catboost (0.030001)
- **MAE:** catboost (0.223610)


## 4. Summary and Conclusions

### Key Findings

- **Best R-AUC MSE:** catboost (0.030001)
- **Best RMSE:** catboost (0.343151)

### Model Comparison

Based on the baseline comparison:

- HMTL model performance compared to baselines:

Top 3 models by R-AUC MSE:

| Model      |   R-AUC MSE |     RMSE |   Rejection Ratio (%) |
|:-----------|------------:|---------:|----------------------:|
| catboost   |   0.0300009 | 0.343151 |               77.5558 |
| hmtl       |   0.0821325 | 0.507136 |               68.788  |
| single_mlp |   0.139274  | 0.624757 |               70.8277 |

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
