"""Participant-level parameter estimation by maximum likelihood."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .likelihood import DEFAULT_DELAY_COLS, DEFAULT_EPS, log_likelihood, trial_arrays


def fit_grid(
    df,
    candidate_ks,
    candidate_betas,
    delay_cols=DEFAULT_DELAY_COLS,
    eps=DEFAULT_EPS,
):
    """Maximum-likelihood fit by exhaustive grid search.

    Evaluates every (k, beta) pair at once with NumPy broadcasting instead of
    a Python double loop. Ties go to the first pair in (beta, k) order, the
    same as a loop over betas then ks that keeps strictly better values.

    Returns
    -------
    (best_k, best_beta, best_log_likelihood)
    """
    ks = np.asarray(candidate_ks, dtype=float)
    betas = np.asarray(candidate_betas, dtype=float)
    ss_value, ss_delay, ll_value, ll_delay = trial_arrays(df, delay_cols)
    choices = df["choice"].to_numpy()

    # Value difference for every k and trial: shape (n_k, n_trials).
    value_difference = (
        ll_value / (1 + ks[:, None] * ll_delay)
        - ss_value / (1 + ks[:, None] * ss_delay)
    )

    # Log-likelihood for every (beta, k): shape (n_beta, n_k).
    grid = np.empty((len(betas), len(ks)))
    for i, beta in enumerate(betas):
        p_ll = np.clip(expit(beta * value_difference), eps, 1 - eps)
        grid[i] = np.log(np.where(choices == 1, p_ll, 1 - p_ll)).sum(axis=1)

    beta_index, k_index = np.unravel_index(np.argmax(grid), grid.shape)
    return ks[k_index], betas[beta_index], grid[beta_index, k_index]


DEFAULT_STARTS = [
    (k, beta) for k in (1e-3, 1e-2, 0.1, 1.0) for beta in (0.01, 0.1, 1.0, 10.0)
]


def fit_mle(
    df,
    delay_cols=DEFAULT_DELAY_COLS,
    k_bounds=(1e-6, 100),
    beta_bounds=(1e-4, 100),
    starts=None,
    eps=DEFAULT_EPS,
):
    """Maximum-likelihood fit with SciPy's bounded L-BFGS-B optimizer.

    Optimizes log(k) and log(beta) so both parameters stay positive. Much
    faster than a fine grid search.

    The likelihood surface can have flat regions where the optimizer stops
    early and still reports success (for example, beta collapsing to its
    lower bound). To guard against that, the fit runs from several starting
    points and keeps the best result.

    Parameters
    ----------
    starts : list of (k, beta) pairs, optional
        Starting points. Defaults to a 4 x 4 grid spanning several orders
        of magnitude.

    Returns
    -------
    dict with ``estimated_k``, ``estimated_beta``, ``negative_log_likelihood``,
    ``converged``, and ``at_boundary`` (True if either estimate sits on a
    bound, meaning it is not well identified).
    """

    def negative_log_likelihood(log_params):
        k, beta = np.exp(log_params)
        return -log_likelihood(k, beta, df, delay_cols, eps)

    log_bounds = [np.log(k_bounds), np.log(beta_bounds)]
    result = None
    for start in DEFAULT_STARTS if starts is None else starts:
        x0 = np.clip(np.log(start), [b[0] for b in log_bounds], [b[1] for b in log_bounds])
        candidate = minimize(
            negative_log_likelihood, x0=x0, method="L-BFGS-B", bounds=log_bounds
        )
        if result is None or candidate.fun < result.fun:
            result = candidate
    estimated_k, estimated_beta = np.exp(result.x)

    return {
        "estimated_k": estimated_k,
        "estimated_beta": estimated_beta,
        "negative_log_likelihood": result.fun,
        "converged": bool(result.success),
        "at_boundary": bool(
            np.isclose(estimated_k, k_bounds).any()
            or np.isclose(estimated_beta, beta_bounds).any()
        ),
    }
