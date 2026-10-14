# Data

Raw datasets are not included in this repository. Download them and place
them in `data/raw/` as shown below.

## Intertemporal Choice Database (notebooks 05–07)

Pongratz, H., & Schoemann, M. (2026). *A large-scale dataset of choice and
response-time data in intertemporal choice*. Scientific Data, 13, 323.
https://doi.org/10.1038/s41597-026-06947-4

Download `all_data.csv` from the project's OSF repository (OSF link: TODO)
and place it at:

```text
data/raw/all_data.csv
```

The analyses use the `Becker_2023_Exp1` study. Delay columns in this
dataset are named `ss_time` and `ll_time`.

## AgeRisk (notebook 08)

## Source: (citation and OSF link)
Pongratz, H., & Schoemann, M. (2026). A large-scale dataset of choice and
response-time data in intertemporal choice. *Scientific Data, 13*, 323.
https://doi.org/10.1038/s41597-026-06947-4

https://osf.io/3wsae/

Place the files like this:

```text
data/raw/agerisk/
├── delay_discounting/                 # one *_DelayDiscounting.csv per participant
├── automated_operation_span.csv
├── demographics.csv
├── numeracy.csv
└── participants_key.csv
```

## Processed outputs

Notebooks write participant-level results to `data/processed/`:

| File | Written by | Used by |
| --- | --- | --- |
| `becker_2023_parameters.csv` | 05 | 07 |
| `cross_validation_results.csv` | 06 | 07 |
