"""Shared accuracy metrics for backtest.py, evaluate_fixed.py, and tuning.py
-- kept in one place so every script scores predictions the same way and the
numbers in README.md are actually comparable to each other.
"""
import numpy as np


def mae(actual, predicted):
    return float(np.mean(np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))))


def rmse(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def nse(actual, predicted):
    """Nash-Sutcliffe Efficiency: 1.0 is perfect, 0.0 matches predicting the
    mean, negative means worse than predicting the mean. Standard hydrology
    metric (as opposed to plain R^2) because "would a persistence/mean
    forecast have done better" is exactly the question that matters for a
    flood model -- see the negative NSE at 12h+ that drove the <=6h scope
    decision in README.md.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.sum((actual - actual.mean()) ** 2)
    if denom == 0:
        return float("nan")  # actual is constant over the window -- NSE is undefined, not zero
    return 1 - np.sum((actual - predicted) ** 2) / denom
