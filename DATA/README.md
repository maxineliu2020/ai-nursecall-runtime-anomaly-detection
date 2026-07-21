# Canonical data snapshot

`erm2-nwe9.csv` is the fixed 100,000-record NYC Open Data snapshot analysed in the manuscript. Raw 311 columns are retained; response time, hour, weekday, labels, and model features are derived at run time by `src/experiment_v2.py`.

## Integrity

```text
File: erm2-nwe9.csv
SHA-256: bf29e88716efe5cb4d6fd26e3fc4d46edf05f6545ea3481e99c46199197b97c0
Rows: 100,000
Created-date range: 2025-10-16 17:35:07 to 2025-10-26 01:49:41
```

The pipeline expects the source columns `created_date`, `closed_date`, and `agency`. It excludes records with missing closure timestamps or negative response intervals; zero-duration closures are retained and reported.

## Provenance

Source: City of New York, **311 Service Requests from 2010 to Present**, dataset identifier `erm2-nwe9`.

https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9

The accompanying Excel file is the official source data dictionary. Refer to NYC Open Data’s current terms of use when redistributing the snapshot.
