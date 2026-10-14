import numpy as np
import pandas as pd

from discounting import cross_validate_participant, summarize_folds

KS = np.logspace(-4, 0, 60)
BETAS = np.logspace(-2, 1, 50)


def test_every_trial_is_held_out_exactly_once(trials_factory):
    df = trials_factory(63, k=0.03, beta=0.4)
    folds = cross_validate_participant(df, KS, BETAS, n_folds=5)
    assert len(folds) == 5
    assert folds["n_test"].sum() == len(df)


def test_results_are_reproducible_with_a_seed(trials_factory):
    df = trials_factory(60, k=0.03, beta=0.4)
    first = cross_validate_participant(df, KS, BETAS, seed=3)
    second = cross_validate_participant(df, KS, BETAS, seed=3)
    pd.testing.assert_frame_equal(first, second)


def test_model_beats_baseline_on_data_it_generated(trials_factory):
    df = trials_factory(400, k=0.03, beta=0.4, seed=11)
    summary = summarize_folds(cross_validate_participant(df, KS, BETAS))
    assert summary["accuracy_gain"] > 0
    assert summary["log_loss_gain"] > 0


def test_summary_weights_folds_by_size():
    folds = pd.DataFrame({
        "n_test": [3, 1],
        "model_accuracy": [1.0, 0.0],
        "baseline_accuracy": [0.0, 0.0],
        "model_log_loss": [0.2, 0.6],
        "baseline_log_loss": [0.7, 0.7],
    })
    summary = summarize_folds(folds)
    assert np.isclose(summary["model_accuracy"], 0.75)
    assert np.isclose(summary["accuracy_gain"], 0.75)
    assert np.isclose(summary["log_loss_gain"], 0.7 - 0.3)
