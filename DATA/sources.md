# Data provenance

## erm2-nwe9.csv (committed snapshot)
- Source: NYC Open Data, "311 Service Requests from 2010 to Present"
  https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9
- Snapshot: most recent 100,000 records at download time
  (created_date 2025-10-16 17:35:07 → 2025-10-26 01:49:41, sorted descending).
- Raw 311 columns are retained; response time and calendar features are derived
  at run time by `src/experiment_v2.py` (not precomputed in the CSV).
- License/terms: NYC Open Data terms of use.

## Using your own data
Provide a CSV with created/closed timestamp columns (any of the synonyms detected
by the loader) and optionally a category column. The pipeline derives
`response_time_h`, `hour`, `weekday`, and one-hot top-8 categories.
