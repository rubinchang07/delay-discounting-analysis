import numpy as np
import pandas as pd

from discounting import binary_log_loss, choice_probability, log_likelihood


def test_log_likelihood_matches_manual_calculation(trials_factory):
    df = trials_factory(50, k=0.02, beta=0.3)
    p = choice_probability(df.ss_value, df.ss_delay, df.ll_value, df.ll_delay, 0.02, 0.3)
    expected = np.sum(np.where(df.choice == 1, np.log(p), np.log(1 - p)))
    assert np.isclose(log_likelihood(0.02, 0.3, df), expected)


def test_true_parameters_beat_wrong_parameters(trials_factory):
    df = trials_factory(500, k=0.02, beta=0.3)
    assert log_likelihood(0.02, 0.3, df) > log_likelihood(0.5, 0.3, df)


def test_custom_delay_column_names(trials_factory):
    df = trials_factory(50, k=0.02, beta=0.3)
    renamed = df.rename(columns={"ss_delay": "ss_time", "ll_delay": "ll_time"})
    assert np.isclose(
        log_likelihood(0.02, 0.3, renamed, delay_cols=("ss_time", "ll_time")),
        log_likelihood(0.02, 0.3, df),
    )


def test_clipping_prevents_infinite_log_likelihood():
    # A near-certain LL prediction paired with an SS choice would be log(0).
    df = pd.DataFrame({
        "ss_value": [1.0], "ss_delay": [0.0],
        "ll_value": [1000.0], "ll_delay": [0.0], "choice": [0],
    })
    assert np.isfinite(log_likelihood(0.01, 100.0, df))


def test_log_loss_of_coin_flip_is_log_2():
    assert np.isclose(binary_log_loss([0, 1, 1, 0], [0.5] * 4), np.log(2))


def test_log_loss_is_near_zero_for_confident_correct_predictions():
    assert binary_log_loss([1, 0], [0.999, 0.001]) < 0.01
