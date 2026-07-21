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
The canonical Pipeline B script requires the source columns `created_date`,
`closed_date`, and `agency`. Rename equivalent fields before running the
manuscript analysis. The simplified Streamlit companion can detect common
created/closed date column variants, but that convenience loader is not part of
the canonical experiment. The pipeline derives `resp_h`, `hour`, `weekday`, and
one-hot top-eight agency indicators.
