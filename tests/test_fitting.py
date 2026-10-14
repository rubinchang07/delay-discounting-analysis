import numpy as np
import pandas as pd
import pytest

from discounting import fit_grid, fit_mle, log_likelihood

KS = np.logspace(-4, np.log10(0.5), 60)
BETAS = np.logspace(-2, 1, 50)


def loop_grid_search(df, candidate_ks, candidate_betas):
    """Reference implementation: the original nested loop from the notebooks."""
    best = (None, None, -np.inf)
    for beta in candidate_betas:
        for k in candidate_ks:
            ll = log_likelihood(k, beta, df)
            if ll > best[2]:
                best = (k, beta, ll)
    return best


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_grid_search_matches_original_loop(trials_factory, seed):
    df = trials_factory(60, k=0.03, beta=0.4, seed=seed)
    fast = fit_grid(df, KS, BETAS)
    slow = loop_grid_search(df, KS, BETAS)
    assert fast[0] == slow[0] and fast[1] == slow[1]
    assert np.isclose(fast[2], slow[2])


@pytest.mark.parametrize("true_k, true_beta", [(0.01, 0.5), (0.05, 0.2), (0.2, 1.0)])
def test_mle_recovers_simulated_parameters(trials_factory, true_k, true_beta):
    # Parameter recovery: with enough trials, the fit should find the
    # parameters that generated the data.
    df = trials_factory(3000, k=true_k, beta=true_beta, seed=42)
    fit = fit_mle(df)
    assert fit["converged"]
    assert not fit["at_boundary"]
    assert abs(np.log10(fit["estimated_k"]) - np.log10(true_k)) < 0.15
    assert abs(np.log10(fit["estimated_beta"]) - np.log10(true_beta)) < 0.15


def test_grid_and_mle_agree(trials_factory):
    df = trials_factory(2000, k=0.03, beta=0.4, seed=7)
    k_grid, beta_grid, ll_grid = fit_grid(df, np.logspace(-4, 0, 200), np.logspace(-2, 1, 160))
    fit = fit_mle(df)
    assert abs(np.log10(k_grid) - np.log10(fit["estimated_k"])) < 0.05
    # The continuous optimizer should do at least as well as the grid.
    assert -fit["negative_log_likelihood"] >= ll_grid - 1e-6


def test_multiple_starts_avoid_a_false_boundary_solution(trials_factory):
    # Regression test: from the single start (k=0.1, beta=1) on this dataset,
    # L-BFGS-B "converges" with beta stuck at its lower bound and a far worse
    # likelihood. Multiple starting points find the real optimum.
    df = trials_factory(2000, k=0.03, beta=0.4, seed=7)
    single = fit_mle(df, starts=[(0.1, 1.0)])
    multi = fit_mle(df)
    assert single["at_boundary"]
    assert not multi["at_boundary"]
    assert multi["negative_log_likelihood"] < single["negative_log_likelihood"] - 100


def test_always_waiting_hits_the_lower_k_bound():
    # Someone who always picks larger-later shows no measurable discounting,
    # so k is not identified and should be flagged.
    df = pd.DataFrame({
        "ss_value": np.full(80, 20.0), "ss_delay": np.zeros(80),
        "ll_value": np.full(80, 30.0), "ll_delay": np.full(80, 60.0),
        "choice": np.ones(80, dtype=int),
    })
    assert fit_mle(df)["at_boundary"]
