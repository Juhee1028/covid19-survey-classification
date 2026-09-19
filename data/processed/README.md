# Data Provenance and Access

## Source

The dataset used in this project is a 5,789-record subset of the public Nexoid COVID-19 Survival Calculator dataset.

- Original publisher: Nexoid, United Kingdom
- Original dataset: [Nexoid COVID-19 Survival Calculator dataset](https://www.covid19survivalcalculator.com/en/download)
- Original dataset size stated by Nexoid: 1,023,426 records
- Project subset: 5,789 records and 38 variables
- Dataset licence: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)

The subset sampling procedure is not available, so the exact 5,789 records cannot be reconstructed from the full public file.

## Licence and Attribution

CC BY 4.0 permits sharing and adaptation, including commercial use, provided that users:

1. Credit Nexoid as the dataset creator.
2. Link to the CC BY 4.0 licence.
3. State whether the data was subsetted, cleaned, transformed, or otherwise modified.
4. Do not imply that Nexoid endorses the analysis.

A suitable attribution is:

> Data adapted from the Nexoid COVID-19 Survival Calculator dataset, licensed under CC BY 4.0. This project uses a 5,789-record subset and applies additional cleaning and feature preparation. Nexoid does not endorse this analysis.

The repository's MIT License applies only to original code and documentation. It does not replace the CC BY 4.0 terms attached to the data.

## Privacy Decision

Nexoid states that the public dataset excludes email addresses, IP addresses, and full dates of birth. Age is grouped into 10-year bands, timestamps are randomly adjusted, and latitude and longitude are randomly offset. The dataset still contains health-related responses and approximate location fields.

For data minimisation, this repository does not publish the raw project subset. `Participant_ID`, `ip_latitude`, `ip_longitude`, and precomputed risk scores are excluded from modelling. If a processed derivative is published later, it should omit those fields and retain the Nexoid attribution above.

## Reproduce the Analysis

With access to the project dataset, place the file at `data/Dataset.csv` and run:

```bash
python src/train_models.py --data data/Dataset.csv
```

The analysis expects a binary target named `covid19_positive`. The CSV is ignored by Git so it is not accidentally committed.
