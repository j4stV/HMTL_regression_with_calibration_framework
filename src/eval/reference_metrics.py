"""Метрики из референсного репозитория

Этот модуль содержит реализации метрик из референсного репозитория для сравнения.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score
from sklearn.utils.multiclass import type_of_target
from sklearn.utils import check_consistent_length, column_or_1d, assert_all_finite
from sklearn.utils.extmath import stable_cumsum

from src.utils.logger import get_logger

logger = get_logger("eval.reference_metrics")


def calc_uncertainty_rejection_curve(errors, uncertainty, group_by_uncertainty=True):
    """
    Вычисляет кривую отбрасывания (rejection curve) по неопределенности.
        
    Args:
        errors: Массив ошибок для каждого объекта [n_samples]
        uncertainty: Массив неопределенностей [n_samples]
        group_by_uncertainty: Группировать ли ошибки по одинаковым значениям неопределенности
    
    Returns:
        error_rates: Массив из n_objects + 1 элементов с уровнями ошибок при отбрасывании
    """
    n_objects = errors.shape[0]
    if group_by_uncertainty:
        data = pd.DataFrame(dict(
            errors=errors,
            uncertainty=uncertainty
        ))
        mean_errors = data.groupby("uncertainty").mean()
        mean_errors.rename(columns={"errors": "mean_errors"}, inplace=True)
        data = data.join(mean_errors, "uncertainty")
        data.drop("errors", axis=1, inplace=True)

        uncertainty_order = data["uncertainty"].argsort()
        errors = data["mean_errors"][uncertainty_order]
    else:
        uncertainty_order = uncertainty.argsort()
        errors = errors[uncertainty_order]

    error_rates = np.zeros(n_objects + 1)
    error_rates[:-1] = np.cumsum(errors)[::-1] / n_objects
    return error_rates


def calc_aucs(errors, uncertainty):
    """
    Вычисляет нормализованную метрику rejection ratio.
    
    Args:
        errors: Массив ошибок [n_samples]
        uncertainty: Массив неопределенностей [n_samples]
    
    Returns:
        rejection_ratio: Нормализованная метрика в процентах (0-100)
        uncertainty_rejection_auc: Сырое значение AUC кривой отбрасывания
    """
    uncertainty_rejection_curve = calc_uncertainty_rejection_curve(errors, uncertainty)
    uncertainty_rejection_auc = uncertainty_rejection_curve.mean()
    random_rejection_auc = uncertainty_rejection_curve[0] / 2
    ideal_rejection_auc = calc_uncertainty_rejection_curve(errors, errors).mean()

    rejection_ratio = (uncertainty_rejection_auc - random_rejection_auc) / (
            ideal_rejection_auc - random_rejection_auc) * 100.0
    return rejection_ratio, uncertainty_rejection_auc


def prr_regression(targets, preds, measure):
    """
    Обертка для регрессии - вычисляет rejection ratio для регрессионной задачи.
    
    Args:
        targets: Истинные значения [n_samples]
        preds: Предсказания [n_samples] или [n_samples, 1]
        measure: Мера неопределенности [n_samples]
    
    Returns:
        rejection_ratio: Нормализованная метрика в процентах
        uncertainty_rejection_auc: Сырое значение AUC
    """
    preds = np.squeeze(preds)
    # Compute MSE errors
    errors = (preds - targets) ** 2
    return calc_aucs(errors, measure)


def _check_pos_label_consistency(pos_label, y_true):
    """Проверка согласованности pos_label для бинарной классификации."""
    classes = np.unique(y_true)
    if (pos_label is None and (
            classes.dtype.kind in 'OUS' or
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1])))):
        classes_repr = ", ".join(repr(c) for c in classes)
        raise ValueError(
            f"y_true takes value in {{{classes_repr}}} and pos_label is not "
            f"specified: either make y_true take value in {{0, 1}} or "
            f"{{-1, 1}} or pass pos_label explicitly."
        )
    elif pos_label is None:
        pos_label = 1.0

    return pos_label


def _binary_clf_curve_ret(y_true, y_score, pos_label=None, sample_weight=None):
    """Вспомогательная функция для построения кривой классификации."""
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    pos_label = _check_pos_label_consistency(pos_label, y_true)

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)
    if sample_weight is not None:
        fps = stable_cumsum((1 - y_true) * weight)
    else:
        fps = stable_cumsum((1 - y_true))
    return fps, tps, y_score


def _precision_recall_curve_retention(y_true, probas_pred, *, pos_label=None,
                                      sample_weight=None):
    """Вспомогательная функция для построения precision-recall кривой."""
    fps, tps, thresholds = _binary_clf_curve_ret(y_true, probas_pred,
                                                 pos_label=pos_label,
                                                 sample_weight=sample_weight)

    precision = tps / (tps + fps)
    precision[np.isnan(precision)] = 0
    recall = tps / tps[-1]

    # stop when full recall attained
    # and reverse the outputs so recall is decreasing
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(-1, None, -1)
    return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _acceptable_error(errors, threshold):
    """Проверяет, является ли ошибка приемлемой (ниже порога)."""
    return np.asarray(errors <= threshold, dtype=np.float32)


def _calc_fbeta_rejection_curve(errors, uncertainty, threshold, beta=1.0, group_by_uncertainty=True, eps=1e-10):
    """
    Вычисляет F-beta кривую отбрасывания.
    
    """
    ae = _acceptable_error(errors, threshold)
    pr, rec, _ = _precision_recall_curve_retention(ae, -uncertainty)
    pr = np.asarray(pr)
    rec = np.asarray(rec)
    f_scores = (1 + beta ** 2) * pr * rec / (pr * beta ** 2 + rec + eps)

    return f_scores, pr, rec


def f_beta_metrics(errors, uncertainty, threshold, beta=1.0):
    """
    Вычисляет F-beta метрики для оценки качества неопределенности.
    
    
    Args:
        errors: Ошибки для каждого образца [n_samples]
        uncertainty: Неопределенности для каждого предсказания [n_samples]
        threshold: Порог ошибки, ниже которого предсказание считается приемлемым
        beta: Значение beta для F-beta метрики (по умолчанию 1.0 = F1)
    
    Returns:
        fbeta_auc: AUC под кривой F-beta
        fbeta_95: Значение F-beta при 95% retention
        retention: Массив значений retention (доля удержанных объектов)
        f_scores: Полный массив F-scores по кривой
    """
    f_scores, pr, rec = _calc_fbeta_rejection_curve(errors, uncertainty, threshold, beta)
    ret = np.arange(pr.shape[0]) / pr.shape[0]

    f_auc = auc(ret[::-1], f_scores)
    f95 = f_scores[::-1][int(0.95 * pr.shape[0])]

    return f_auc, f95, f_scores[::-1]


def ood_detect(domain_labels, in_measure, out_measure, mode='ROC', pos_label=1):
    """
    Детекция out-of-distribution данных.
    
    
    Args:
        domain_labels: Метки домена (1 для in-distribution, 0 для out-of-distribution)
        in_measure: Мера неопределенности для in-distribution данных
        out_measure: Мера неопределенности для out-of-distribution данных
        mode: 'ROC' или 'PR' для выбора метрики
        pos_label: Метка положительного класса
    
    Returns:
        aupr или roc_auc в зависимости от mode
    """
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores)
    if pos_label != 1:
        scores *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == 'ROC':
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc


def nll_regression(target, mu, var, epsilon=1e-8, raw=False):
    """
    Negative Log-Likelihood для регрессии.
    
    
    Args:
        target: Истинные значения
        mu: Предсказанные средние значения
        var: Предсказанные дисперсии
        epsilon: Малое значение для численной стабильности
        raw: Возвращать ли сырые значения (без усреднения)
    
    Returns:
        NLL значение (среднее или массив)
    """
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    if raw:
        return nll
    return np.mean(nll)


def ens_nll_regression(target, preds, epsilon=1e-8, raw=False):
    """
    NLL для ансамбля моделей.
    
    
    Args:
        target: Истинные значения [n_samples]
        preds: Предсказания ансамбля [n_models, n_samples, 2] где последняя размерность [mean, var]
        epsilon: Малое значение для численной стабильности
        raw: Возвращать ли сырые значения
    
    Returns:
        NLL значение
    """
    mu = preds[:, :, 0]
    var = preds[:, :, 1]
    nll = (target - mu) ** 2 / (2.0 * var + epsilon) + np.log(var + epsilon) / 2.0 + np.log(2 * np.pi) / 2.0
    proba = np.exp(-1 * nll)
    if raw:
        return -1 * np.log(np.mean(proba, axis=0))
    return np.mean(-1 * np.log(np.mean(proba, axis=0)))



