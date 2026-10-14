import numpy as np

from discounting import choice_probability, simulate_choices, subjective_value


def test_no_delay_keeps_full_value():
    assert subjective_value(100, 0, k=0.5) == 100


def test_no_discounting_keeps_full_value():
    assert subjective_value(100, 365, k=0) == 100


def test_value_matches_hyperbolic_formula():
    assert np.isclose(subjective_value(100, 10, k=0.1), 50)


def test_value_decreases_with_delay():
    values = subjective_value(100, np.array([0, 10, 50, 100]), k=0.05)
    assert np.all(np.diff(values) < 0)


def test_equal_values_give_even_odds():
    # $50 now vs $100 in 10 days at k = 0.1: both worth $50.
    assert np.isclose(choice_probability(50, 0, 100, 10, k=0.1, beta=1.0), 0.5)


def test_higher_beta_makes_choices_more_deterministic():
    low = choice_probability(10, 0, 100, 30, k=0.01, beta=0.01)
    high = choice_probability(10, 0, 100, 30, k=0.01, beta=1.0)
    assert 0.5 < low < high <= 1.0


def test_extreme_values_do_not_overflow():
    p = choice_probability(0, 0, 1e6, 0, k=0.1, beta=100.0)
    assert np.isfinite(p) and np.isclose(p, 1.0)


def test_simulation_is_reproducible_with_a_seed():
    args = (np.full(100, 20.0), np.zeros(100), np.full(100, 40.0), np.full(100, 30.0))
    first = simulate_choices(*args, k=0.05, beta=0.2, rng=np.random.default_rng(1))
    second = simulate_choices(*args, k=0.05, beta=0.2, rng=np.random.default_rng(1))
    assert np.array_equal(first, second)


def test_simulated_choice_rate_matches_model_probability():
    n = 20_000
    args = (np.full(n, 20.0), np.zeros(n), np.full(n, 40.0), np.full(n, 30.0))
    expected = choice_probability(*args, k=0.05, beta=0.2)[0]
    choices = simulate_choices(*args, k=0.05, beta=0.2, rng=np.random.default_rng(2))
    assert abs(choices.mean() - expected) < 0.02
