# Data Provenance and Processing

## UCI Online Retail

- Official page: https://archive.ics.uci.edu/dataset/352/online+retail
- DOI: 10.24432/C5BW33
- Licence: CC BY 4.0
- Official ZIP SHA-256 used for the verified run: `f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b`
- Source file: 541,909 rows and eight recorded columns.
- Audit: 5,268 exact duplicates; 135,080 missing customer IDs.
- Modelling data: 3,317 customers with a valid purchase before 1 September 2011; 1,952 purchased again during the following 100-day outcome window.

Valid purchase lines have a known customer ID, a non-cancellation invoice, positive quantity and positive unit price. Historical cancellations remain available only to derive cancellation-line rate. Future transactions never enter customer predictors.

## UCI Online Shoppers Purchasing Intention

- Official page: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- DOI: 10.24432/C5F88Q
- Licence: CC BY 4.0
- Official ZIP SHA-256 used for the verified run: `2972e6184d3ad7beaaa831d9fc2b059dc3ee29df69d1ec593c466a5cd8485d14`
- Source file: 12,330 rows, 17 predictors and one `Revenue` target.
- Audit: no missing values; 125 exact duplicates.
- Modelling data: 12,205 deduplicated sessions; 1,908 purchases (15.6%).

Numeric predictors are standardised inside each validation pipeline. Month, operating system, browser, region, traffic type, visitor type and weekend are one-hot encoded inside the pipeline.

## Raw Files

Raw files are intentionally excluded from version control. Run `python src/run_analysis.py` from the repository root to download them directly from UCI and reproduce all derived artefacts.
