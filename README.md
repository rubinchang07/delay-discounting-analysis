# Delay Discounting Analysis

A computational modeling project examining how people choose between smaller
rewards available sooner and larger rewards available later.

The project builds a hyperbolic delay-discounting model from first principles,
simulates intertemporal choices, estimates participant-level parameters,
validates those estimates through parameter recovery and cross-validation,
and examines cognitive individual differences using two behavioral datasets.

## Overview

Intertemporal choice involves decisions that trade off reward magnitude against
time. A person might prefer $10 today over $15 next month, while another person
may be more willing to wait.

This project models these decisions using hyperbolic delay discounting:

\[
V = \frac{A}{1 + kD}
\]

where:

- `V` is subjective value
- `A` is reward amount
- `D` is delay
- `k` is the discount-rate parameter

Larger values of `k` indicate steeper discounting of delayed rewards.

Choices are modeled probabilistically using a logistic choice rule:

\[
P(LL) =
\frac{1}
{1 + e^{-\beta(V_{LL} - V_{SS})}}
\]

where `β` represents choice sensitivity. Larger values of `β` indicate that
choices are more consistently determined by differences in subjective value.

## Research Questions

This project asks whether:

1. Hyperbolic delay-discounting parameters can be recovered from simulated data.
2. Participant-level parameters can explain real intertemporal-choice behavior.
3. The model generalizes to held-out choices rather than simply fitting noise.
4. Individual differences in `k` and `β` correspond to interpretable behavioral
   differences.
5. Cognitive variables such as working memory, numeracy, and age are associated
   with delay-discounting behavior.

## Analysis Pipeline

| Notebook | Purpose |
| --- | --- |
| `01_discounting_model.ipynb` | Define and visualize hyperbolic subjective value |
| `02_choice_simulation.ipynb` | Simulate probabilistic intertemporal choices |
| `03_parameter_estimation.ipynb` | Estimate discounting parameters using likelihood |
| `04_parameter_recovery.ipynb` | Test whether simulated parameters can be recovered |
| `05_real_data_analysis.ipynb` | Fit the model to real participant choices |
| `06_cross_validation.ipynb` | Evaluate out-of-sample predictive performance |
| `07_individual_differences.ipynb` | Examine variation in `k`, `β`, and model performance |
| `08_cognitive_individual_differences.ipynb` | Relate discounting behavior to working memory, numeracy, and age |

## Data

### Intertemporal Choice Database

The primary model-validation analysis uses `Becker_2023_Exp1` from the
Intertemporal Choice Database compiled by Hannah Pongratz and Martin Schoemann.

After exclusion criteria were applied, the analysis contained:

- 159 participants
- 9,898 intertemporal-choice trials

The database contains standardized trial-level data from a large collection
of published intertemporal-choice studies.

Reference:

Pongratz, H., & Schoemann, M. (2026). *A large-scale dataset of choice and
response-time data in intertemporal choice*. Scientific Data, 13, 323.
DOI: 10.1038/s41597-026-06947-4

### AgeRisk

Cognitive individual-differences analyses use the AgeRisk dataset, a
cross-sectional adult lifespan dataset containing behavioral and cognitive
measures.

The delay-discounting analysis used:

- 200 participants
- 80 trials per participant
- 16,000 total intertemporal-choice trials
- automated operation-span working-memory scores
- numeracy scores
- age and demographic information

Raw datasets are not stored in this repository. See `data/README.md` for data
sources and setup instructions.

## Key Results

### Out-of-Sample Prediction

Five-fold within-participant cross-validation showed that the hyperbolic model
generalized to held-out choices.

Mean cross-validated classification accuracy was approximately **0.768**,
compared with **0.647** for a participant-specific majority-choice baseline.

The mean accuracy improvement was approximately **0.121**, with a bootstrap
95% confidence interval of approximately **[0.103, 0.139]**.

The model also improved probabilistic prediction:

- Mean model log loss: **0.511**
- Mean baseline log loss: **0.638**
- Mean log-loss improvement: **0.127**
- Bootstrap 95% CI: approximately **[0.102, 0.151]**

### Behavioral Meaning of Model Parameters

Estimated discount rate was strongly related to observed immediate-choice
behavior among participants with interior parameter estimates:

\[
r = 0.863,\quad R^2 = 0.745
\]

Participants with larger `k` values chose smaller-sooner rewards more often.

Choice sensitivity was also strongly associated with out-of-sample model
performance:

\[
r = 0.837
\]

for `log10(β)` versus cross-validated accuracy, and

\[
r = 0.719
\]

for `log10(β)` versus cross-validated log-loss improvement.

This supports the interpretation of `β` as a measure of behavioral
consistency or sensitivity to subjective-value differences.

### Cognitive Individual Differences

Working-memory capacity showed little evidence of a relationship with
immediate-choice behavior. Operation-span scores were essentially unrelated
to smaller-sooner choice rate:

\[
r = -0.045,\quad p = .530
\]

Numeracy showed a clearer relationship. Higher numeracy was associated with
fewer smaller-sooner choices:

\[
r = -0.290,\quad p < .001
\]

and with somewhat greater estimated choice sensitivity:

\[
r = 0.176,\quad p = .028
\]

The relationship between numeracy and smaller-sooner choice behavior remained
after controlling for age:

\[
r_{partial} = -0.293,\quad p < .001
\]

Age itself showed little relationship with delay-discounting behavior, although
older participants tended to score lower on both numeracy and working memory.

## Parameter Identification

Participant-level parameters were not equally well identified for every
participant.

In the AgeRisk analysis, all 200 optimization procedures converged, but 43
participants had at least one estimate at an optimization boundary.
Parameter-based cognitive analyses therefore focused primarily on the 157
participants with interior estimates.

This distinction is important: successful numerical optimization does not
necessarily imply that a behavioral parameter is precisely identified.

## Repository Structure

delay-discounting-analysis/
├── data/
│   ├── raw/                 # Local raw datasets; excluded from Git
│   ├── processed/           # Participant-level analysis outputs
│   └── README.md            # Data sources and setup
├── notebooks/
│   ├── 01_discounting_model.ipynb
│   ├── 02_choice_simulation.ipynb
│   ├── 03_parameter_estimation.ipynb
│   ├── 04_parameter_recovery.ipynb
│   ├── 05_real_data_analysis.ipynb
│   ├── 06_cross_validation.ipynb
│   ├── 07_individual_differences.ipynb
│   └── 08_cognitive_individual_differences.ipynb
├── .gitignore
├── README.md
└── requirements.txt

The project is implemented in Python using:
- NumPy
- pandas
- Matplotlib
- SciPy
- Jupyter
Participant parameters are estimated by maximizing the likelihood of observed
binary choices. Positive parameters are optimized in log space using bounded
numerical optimization.

Predictive performance is evaluated using participant-level five-fold
cross-validation, classification accuracy, binary log loss, majority-choice
baselines, and participant-level bootstrap confidence intervals.

Create and activate a Python environment, then install the project
dependencies: pip install -r requirements.txt

The model is intentionally simple. It assumes hyperbolic temporal discounting
and a logistic mapping from subjective-value differences to choice
probabilities.

Participant-level parameter estimates can also be difficult to identify when
choice patterns contain little information about one or both parameters.
Boundary estimates were therefore treated separately in individual-differences
analyses.

The cognitive analyses are correlational and do not establish causal effects
of working memory, numeracy, or age on intertemporal choice.

Future extensions could compare hyperbolic discounting against alternative
models such as exponential discounting, incorporate hierarchical parameter
estimation, or examine additional cognitive and psychological measures.

Author: Rubin Chang