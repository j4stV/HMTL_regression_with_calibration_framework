# HMTL with Calibration

Прототип фреймворка для иерархического многозадачного обучения (HMTL) с оценкой неопределённости и conformal-калибровкой для регрессии на табличных данных.

## Основные возможности

- **HMTL-архитектура**: Иерархическое многозадачное обучение с вспомогательными задачами (классификация бинов или supervised contrastive learning)
- **SNN-энкодер**: Self-Normalizing Network с SELU, LeCun инициализацией и AlphaDropout
- **Ансамбль моделей**: Стратифицированный баггинг для эпистемической неопределённости
- **Оценка неопределённости**: Комбинация эпистемической (Var_epi) и алеаторной (E[σ²]) неопределённости
- **Conformal калибровка**: Split-conformal калибровка предиктивных интервалов для уровней покрытия 80%, 90%, 95%
- **Метрики**: RMSE, R-AUC MSE, coverage, ширина интервалов
- **Визуализация**: Error-retention кривые, calibration plots, reliability диаграммы
- **MLflow интеграция**: Трекинг экспериментов с конфигами, метриками и артефактами

## Структура проекта

```
configs/          # Конфигурационные файлы (data, model, train, ensemble)
src/
  data/           # Предобработка данных
  models/         # Модели (SNN, HMTL, heads, contrastive)
  losses/         # Функции потерь (NLL)
  train/          # Тренировка (loop, ensemble, optimizers)
  eval/           # Оценка (metrics, conformal, ensemble, evaluator, visualization)
  utils/          # Утилиты (logger, mlflow_tracker)
scripts/          # Скрипты запуска
tools/            # Вспомогательные инструменты
docs/             # Документация
experiments/      # Результаты экспериментов (plots, runs)
```

## Установка

1. Python 3.11+ требуется
2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Быстрый старт

### Подготовка датасетов

#### Wine Quality (по умолчанию)

Датасет уже должен быть разделен на `data/train.csv`, `data/valid.csv`, `data/test.csv`.

#### Superconductor Dataset

Для использования более сложного superconductor датасета (21,263 образца, 81 признак), выполните:

```bash
python scripts/prepare_superconductor.py
```

Этот скрипт:
1. Загружает датасет с UCI ML Repository
2. Разделяет на train/valid/test (80%/10%/10%)
3. Сохраняет в `data/superconductor_train.csv`, `data/superconductor_valid.csv`, `data/superconductor_test.csv`

Затем используйте конфигурацию для superconductor:

```bash
python scripts/main.py --data configs/data_superconductor.yaml
```

**Опции скрипта подготовки:**
- `--train-ratio`, `--val-ratio`, `--test-ratio` - пропорции разделения (по умолчанию 0.8/0.1/0.1)
- `--seed` - random seed для воспроизводимости (по умолчанию 42)
- `--csv-path` - путь к уже загруженному CSV файлу (пропускает загрузку)
- `--force-download` - перезагрузить датасет даже если он уже существует

### Базовый запуск

Обучить HMTL ансамбль с conformal калибровкой:

```bash
python scripts/main.py --data configs/data.yaml --model configs/model_snn.yaml --train configs/train.yaml
```

Все конфигурационные файлы имеют значения по умолчанию, поэтому можно запустить просто:

```bash
python scripts/main.py
```

Для superconductor датасета:

```bash
python scripts/main.py --data configs/data_superconductor.yaml
```

### Многократные запуски с разными сидами

Запустить эксперимент с несколькими сидами и агрегировать результаты:

```bash
python scripts/run_multi_seed.py --n_seeds 5 --output experiments/multi_seed
```

Результаты сохраняются в `experiments/multi_seed/`:
- `results.json` - агрегированные метрики и результаты по каждому сиду
- `metric_distributions.png` - распределения метрик
- `coverage_distributions.png` - распределения покрытия

### Сравнение с baseline моделями

Сравнить HMTL с baseline моделями:

```bash
python scripts/run_baselines.py --baselines single_mlp flat_mtl catboost
```

Доступные baseline:
- `single_mlp` - одиночная MLP без HMTL
- `flat_mtl` - плоское многозадачное обучение
- `catboost` - ансамбль CatBoost с оценкой неопределённости

Результаты сохраняются в `experiments/baselines/`:
- `comparison_table.csv` - сводная таблица метрик
- `baseline_comparison.png` - визуализация сравнения

### Абляционные исследования

Провести систематические абляционные исследования:

```bash
python scripts/run_ablation.py
```

Исследуются варианты:
- С/без HMTL
- Тип вспомогательной задачи (бины vs contrastive)
- Глубина сети
- Dropout
- Вес вспомогательной задачи

### Полный эксперимент и генерация отчёта

Запустить все эксперименты и автоматически сформировать итоговый отчёт:

```bash
python scripts/generate_report.py
```

Этот скрипт выполняет:
1. Основной эксперимент HMTL
2. Многократные запуски с разными сидами
3. Сравнение с baseline моделями
4. Абляционные исследования
5. Сбор всех результатов
6. Генерацию итогового отчёта в формате Markdown

Опции:
- `--skip-main` - пропустить основной эксперимент
- `--skip-multi-seed` - пропустить multi-seed эксперименты
- `--skip-baselines` - пропустить сравнение с baseline
- `--skip-ablation` - пропустить абляционные исследования
- `--n-seeds N` - количество сидов для multi-seed (по умолчанию 3)
- `--output DIR` - директория для сохранения отчёта (по умолчанию `experiments/full_report`)

Результаты сохраняются в указанной директории:
- `report.md` - итоговый отчёт в формате Markdown
- `results.json` - все результаты в формате JSON
- `main/` - результаты основного эксперимента
- `multi_seed/` - результаты multi-seed экспериментов
- `baselines/` - результаты сравнения с baseline
- `ablation/` - результаты абляционных исследований

## Конфигурация

### Data config (`configs/data.yaml`)

```yaml
paths:
  train_csv: data/train.csv      # Путь к обучающему набору
  valid_csv: data/valid.csv      # Путь к валидационному набору
  cal_csv: null                   # Опционально: отдельный калибровочный набор
                                  # Если null, используется validation set
  test_csv: data/test.csv         # Путь к тестовому набору (опционально)
  target: quality                 # Название целевой колонки

preprocess:
  impute_const: -1                # Константа для импьютации пропусков
  quantile_binning:
    enabled: false                 # Включить квантильный биннинг признаков
    bins: 5                        # Количество бинов для квантильного биннинга
  standardize: true                # Стандартизация признаков (zero-mean, unit-var)
  pca:
    enabled: true                  # Включить PCA декорреляцию
    n_components: 0.95             # Количество компонент (float = доля дисперсии)
  target_standardize: true         # Стандартизация таргета
```

### Model config (`configs/model_snn.yaml`)

```yaml
encoder:
  hidden_width: 512               # Ширина скрытых слоёв
  depth_base: 12                  # Глубина базового энкодера
  depth_hmtl: 18                  # Глубина с HMTL (верхний уровень)
  alpha_dropout: 0.0003           # Вероятность AlphaDropout
  activation: SELU                # Функция активации
  init: lecun                     # Тип инициализации весов

hmtl:
  enabled: true                   # Включить HMTL
  aux_task: bins                  # Тип вспомогательной задачи: "bins" или "contrastive"
  low_layer: 12                   # Слой для нижней задачи (вспомогательной)
  high_layer: 18                  # Слой для верхней задачи (регрессии)
  lambda_aux: 0.3                 # Вес вспомогательной задачи в loss
  n_bins: 5                       # Количество бинов для классификации
  proj_dim: 128                   # Размерность проекции для contrastive learning
```

### Train config (`configs/train.yaml`)

```yaml
optimizer:
  name: radam_lookahead           # "radam_lookahead" или "adamw"
  lr: 0.0003                      # Learning rate
  lookahead_sync_period: 6        # Период синхронизации для Lookahead
  lookahead_slow_step: 0.5        # Шаг для медленных весов Lookahead

training:
  seed: 42                         # Сид для воспроизводимости
  epochs: 200                      # Максимальное количество эпох
  batch_size: 256                  # Размер батча
  early_stop:
    metric: r_auc_mse              # Метрика для early stopping
    patience: 20                   # Терпение (количество эпох без улучшения)
    mode: min                      # "min" или "max"

logging:
  mlflow:
    enabled: false                 # Включить MLflow трекинг
    tracking_uri: null             # URI MLflow сервера (null = локальный)
  save.dir: experiments/runs      # Директория для сохранения результатов
```

### Ensemble config (`configs/ensemble.yaml`)

```yaml
ensemble:
  n_models: 10                    # Количество моделей в ансамбле (рекомендуется 10-20)
  bagging: stratified_bins        # Стратегия баггинга: "stratified_bins" или "bootstrap"
  val_metric: r_auc_mse           # Метрика для выбора лучшей модели
```

**Примечания:**
- `stratified_bins`: Стратифицированный баггинг по бинам таргета - обеспечивает лучшее покрытие редких регионов
- `bootstrap`: Стандартный бутстрэп без стратификации

## Метрики

### Метрики точности
- **RMSE** (Root Mean Squared Error): Корень из среднего квадрата ошибок
- **MSE** (Mean Squared Error): Средний квадрат ошибок
- **MAE** (Mean Absolute Error): Средняя абсолютная ошибка

### Метрики неопределённости
- **R-AUC MSE**: Площадь под кривой error-retention (главная метрика для оценки неопределённости)
  - Чем ниже, тем лучше
  - Показывает, насколько хорошо модель ранжирует примеры по уверенности
- **Coverage**: Покрытие предиктивных интервалов (доля примеров, попадающих в интервал)
  - Измеряется для уровней 80%, 90%, 95%
  - До и после conformal калибровки
- **Mean Width**: Средняя ширина предиктивных интервалов
  - Показывает "цену" покрытия - более узкие интервалы при том же покрытии лучше

### Компоненты неопределённости
- **Epistemic Uncertainty**: Эпистемическая неопределённость (из-за неопределённости модели)
  - Вычисляется как дисперсия предсказаний по ансамблю
- **Aleatoric Uncertainty**: Алеаторная неопределённость (из-за шума в данных)
  - Вычисляется как среднее предсказанных σ² по ансамблю
- **Total Uncertainty**: Общая неопределённость = √(Var_epi + E[σ²])

## Визуализация

Графики сохраняются в `experiments/plots/`:
- `error_retention.png`: Кривая error-retention
- `calibration.png`: Калибровочная кривая (actual vs nominal coverage)
- `reliability.png`: Reliability диаграмма


## Архитектура

### Предобработка
1. Импьютация пропусков константой -1
2. Опциональный квантильный биннинг
3. Стандартизация признаков
4. PCA декорреляция
5. Стандартизация таргета

### Модель
- **Энкодер**: SNN (SELU + LeCun init + AlphaDropout)
- **Нижний уровень** (после ~12 слоя): Вспомогательная задача (бины или contrastive)
- **Верхний уровень** (после ~18 слоя): Регрессия μ, σ с NLL loss

### Ансамбль
- 5-20 моделей с разными сидами
- Стратифицированный баггинг по бинам таргета
- Агрегация: μ̄, Var_epi(μ_i) + E[σ_i²]

### Conformal калибровка
- Split-conformal на калибровочном наборе
- Множественные уровни покрытия (80%, 90%, 95%)
- Метрики до/после калибровки

## Baseline модели

Для сравнения реализованы следующие baseline модели:

- **Single MLP**: Одиночная MLP без HMTL и вспомогательных задач
- **Flat MTL**: Многозадачное обучение без иерархии (оба таска используют один энкодер)
- **CatBoost**: Ансамбль CatBoost с встроенной оценкой неопределённости (RMSEWithUncertainty)

## Абляционные исследования

Фреймворк поддерживает систематические абляционные исследования:

- Включение/выключение HMTL
- Тип вспомогательной задачи (бины vs contrastive)
- Глубина сети (shallow/deep)
- Dropout (низкий/высокий)
- Вес вспомогательной задачи (низкий/высокий)

Скрипт `scripts/run_ablation.py` автоматически запускает все варианты и сравнивает результаты.
