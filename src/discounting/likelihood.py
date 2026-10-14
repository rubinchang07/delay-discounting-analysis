"""Likelihood and prediction functions for trial-level choice data.

Trial data are pandas DataFrames with one row per choice and these columns:

- ``ss_value`` and ``ll_value``: smaller-sooner and larger-later amounts
- two delay columns, named by ``delay_cols`` (default ``ss_delay``, ``ll_delay``)
- ``choice``: 1 for larger-later, 0 for smaller-sooner
"""

import numpy as np

from .model import choice_probability

DEFAULT_DELAY_COLS = ("ss_delay", "ll_delay")
DEFAULT_EPS = 1e-10


def trial_arrays(df, delay_cols=DEFAULT_DELAY_COLS):
    """Extract (ss_value, ss_delay, ll_value, ll_delay) arrays from a DataFrame."""
    ss_delay_col, ll_delay_col = delay_cols
    return (
        df["ss_value"].to_numpy(dtype=float),
        df[ss_delay_col].to_numpy(dtype=float),
        df["ll_value"].to_numpy(dtype=float),
        df[ll_delay_col].to_numpy(dtype=float),
    )


def predict_ll_probability(df, k, beta, delay_cols=DEFAULT_DELAY_COLS):
    """Predicted probability of a larger-later choice for each trial."""
    return choice_probability(*trial_arrays(df, delay_cols), k, beta)


def log_likelihood(k, beta, df, delay_cols=DEFAULT_DELAY_COLS, eps=DEFAULT_EPS):
    """Log-likelihood of the observed choices under parameters ``k`` and ``beta``.

    Probabilities are clipped to ``[eps, 1 - eps]`` so a single extreme
    prediction cannot produce ``log(0)``.
    """
    p_ll = np.clip(predict_ll_probability(df, k, beta, delay_cols), eps, 1 - eps)
    choices = df["choice"].to_numpy()
    return np.log(np.where(choices == 1, p_ll, 1 - p_ll)).sum()


def binary_log_loss(actual, probabilities, eps=DEFAULT_EPS):
    """Mean binary cross-entropy between 0/1 outcomes and predicted probabilities."""
    actual = np.asarray(actual, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), eps, 1 - eps)
    return -np.mean(
        actual * np.log(probabilities) + (1 - actual) * np.log(1 - probabilities)
    )
