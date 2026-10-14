"""Out-of-sample evaluation with within-participant k-fold cross-validation."""

import numpy as np
import pandas as pd

from .fitting import fit_grid
from .likelihood import DEFAULT_DELAY_COLS, binary_log_loss, predict_ll_probability


def cross_validate_participant(
    participant_df,
    candidate_ks,
    candidate_betas,
    n_folds=5,
    seed=42,
    delay_cols=DEFAULT_DELAY_COLS,
):
    """Cross-validate the model for one participant.

    For each fold: fit k and beta on the other folds, predict the held-out
    trials, and compare with a majority-choice baseline from the training
    trials. Each trial is held out exactly once.

    Returns a DataFrame with one row per fold.
    """
    participant_df = participant_df.reset_index(drop=True)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(participant_df))
    rng.shuffle(indices)
    folds = np.array_split(indices, n_folds)

    fold_results = []
    for fold_index in range(n_folds):
        test_indices = folds[fold_index]
        train_indices = np.concatenate(
            [folds[i] for i in range(n_folds) if i != fold_index]
        )
        train_df = participant_df.iloc[train_indices]
        test_df = participant_df.iloc[test_indices]

        estimated_k, estimated_beta, _ = fit_grid(
            train_df, candidate_ks, candidate_betas, delay_cols
        )

        actual = test_df["choice"].to_numpy()
        p_ll = predict_ll_probability(test_df, estimated_k, estimated_beta, delay_cols)
        predicted = (p_ll >= 0.5).astype(int)

        training_ll_rate = train_df["choice"].mean()
        baseline_choice = 1 if training_ll_rate >= 0.5 else 0
        baseline_probs = np.full(len(test_df), training_ll_rate)

        fold_results.append({
            "n_test": len(test_df),
            "estimated_k": estimated_k,
            "estimated_beta": estimated_beta,
            "model_accuracy": (predicted == actual).mean(),
            "baseline_accuracy": (actual == baseline_choice).mean(),
            "model_log_loss": binary_log_loss(actual, p_ll),
            "baseline_log_loss": binary_log_loss(actual, baseline_probs),
        })

    return pd.DataFrame(fold_results)


def summarize_folds(fold_df):
    """Combine fold-level scores, weighting each fold by its number of test trials."""
    weights = fold_df["n_test"]
    summary = {
        column: np.average(fold_df[column], weights=weights)
        for column in [
            "model_accuracy",
            "baseline_accuracy",
            "model_log_loss",
            "baseline_log_loss",
        ]
    }
    summary["accuracy_gain"] = summary["model_accuracy"] - summary["baseline_accuracy"]
    # Positive means the model beats the baseline.
    summary["log_loss_gain"] = summary["baseline_log_loss"] - summary["model_log_loss"]
    return summary
