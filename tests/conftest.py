import numpy as np
import pandas as pd
import pytest

from discounting import simulate_choices


def make_trials(n_trials, k, beta, seed=0):
    """Simulate a participant's choices on random smaller-sooner vs larger-later trials."""
    rng = np.random.default_rng(seed)
    ss_value = rng.uniform(10, 50, n_trials)
    ll_value = ss_value * rng.uniform(1.1, 3.0, n_trials)
    ss_delay = rng.choice([0, 7, 14], n_trials).astype(float)
    ll_delay = ss_delay + rng.uniform(7, 180, n_trials)
    choice = simulate_choices(ss_value, ss_delay, ll_value, ll_delay, k, beta, rng)
    return pd.DataFrame({
        "ss_value": ss_value,
        "ss_delay": ss_delay,
        "ll_value": ll_value,
        "ll_delay": ll_delay,
        "choice": choice,
    })


@pytest.fixture
def trials_factory():
    return make_trials
