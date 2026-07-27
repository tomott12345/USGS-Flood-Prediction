"""Split conformal prediction for calibrated confidence intervals.

The earlier prototype (evaluation/xgboost_prototype.py) trained two
independent XGBoost quantile-regression models (alpha=0.1 and 0.9) and found
they badly under-covered -- as low as 11% empirical coverage against an 80%
nominal target. Unsurprising: quantile-regression trees need much more data
than a few dozen to a few hundred walk-forward training rows to estimate tail
behavior reliably, and nothing constrains the two independently-trained
quantiles to actually bracket 80% of outcomes.

Split conformal prediction sidesteps that: instead of modeling the *shape* of
the error distribution, it directly measures the empirical error on a
held-out calibration slice and uses that to build an interval with
guaranteed marginal coverage (under the standard exchangeability assumption),
regardless of what the error distribution looks like.
"""
import math

import numpy as np


def conformal_margin(residuals, alpha=0.2):
    """Return the margin m such that pred +/- m is expected to cover
    ~(1 - alpha) of future outcomes, given absolute residuals observed on a
    calibration set. Uses the standard finite-sample-corrected empirical
    quantile (Lei et al. 2018, "Distribution-Free Predictive Inference").
    """
    residuals = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    n = len(residuals)
    if n == 0:
        raise ValueError("Cannot compute a conformal margin with zero calibration residuals.")

    # The "+1" and ceil() are the finite-sample correction: a plain quantile
    # (e.g. the 80th percentile of n residuals) undercovers slightly on
    # finite data, because it doesn't account for the future test point
    # itself being one more exchangeable draw. Rounding up to the ceil(...)-th
    # order statistic of n+1 points (n observed + 1 unseen) is what gives the
    # marginal-coverage guarantee, not just an approximation of it.
    rank = math.ceil((n + 1) * (1 - alpha))
    rank = min(rank, n)  # cap at the largest observed residual when alpha is tiny relative to n
    return float(residuals[rank - 1])  # -1: rank is 1-indexed, numpy arrays aren't
