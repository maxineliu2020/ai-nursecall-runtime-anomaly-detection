# DATA

This folder contains **input data** for the anomaly-detection experiments.

## Files

- `erm2-nwe9.csv`  
  Main subset used in the paper and course project. It is derived from the
  NYC 311 service-request dataset, with additional preprocessing and
  engineered fields (e.g., response time in hours, weekday, hour of day).

- `311_ServiceRequest_2010-Present_DataDictionary_Updated_2023.xlsx`  
  Official data dictionary from NYC OpenData describing all columns in the
  original 311 dataset.

> **Note:** The full raw NYC 311 CSV is not committed to this repository
> because it is large and can change over time. Please download it directly
> from NYC OpenData following the instructions in `docs/data_sources.md`.

## Expected Columns (in `erm2-nwe9.csv`)

The preprocessed CSV used by `src/experiment_real_plus.py` is expected to
contain at least the following columns:

- `resp_h` – response time in hours (continuous)
- `is_anomaly` – binary label (1 = anomalous / delayed, 0 = normal)
- `category` – agency or ticket category (e.g., NYPD, HPD, DSNY, …)
- `hour` – hour of day (0–23)
- `weekday` – weekday index (0=Mon … 6=Sun)

The script ignores additional columns and can be safely kept.


Data download link
311 Service Requests from 2010 to Present | NYC Open Data
https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9/about_data
