"""Hyperbolic delay-discounting model and probabilistic choice rule."""

import numpy as np
from scipy.special import expit


def subjective_value(amount, delay, k):
    """Hyperbolic subjective value: V = A / (1 + kD).

    Works on scalars or NumPy arrays.
    """
    return amount / (1 + k * delay)


def choice_probability(ss_value, ss_delay, ll_value, ll_delay, k, beta):
    """Probability of choosing the larger-later (LL) option.

    Uses a logistic choice rule on the difference in subjective value:
    P(LL) = 1 / (1 + exp(-beta * (V_LL - V_SS))).
    """
    value_difference = (
        subjective_value(ll_value, ll_delay, k)
        - subjective_value(ss_value, ss_delay, k)
    )
    # expit is the logistic function; it avoids overflow warnings from np.exp.
    return expit(beta * value_difference)


def simulate_choices(ss_value, ss_delay, ll_value, ll_delay, k, beta, rng=None):
    """Simulate binary choices (1 = larger-later, 0 = smaller-sooner).

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Random number generator, so simulations are reproducible.
    """
    rng = np.random.default_rng() if rng is None else rng
    p_ll = choice_probability(ss_value, ss_delay, ll_value, ll_delay, k, beta)
    return (rng.random(np.shape(p_ll)) < p_ll).astype(int)
