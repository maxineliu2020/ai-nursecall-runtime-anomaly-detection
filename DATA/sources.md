# Using Your Own Data

To run the experiment on your own nurse-call or ticketing data:

1. Map your columns to the same logical fields:

creation timestamp → Created Date

closure or response timestamp → Closed Date

category / department → category

2. Follow the preprocessing steps above to produce:

resp_h, hour, weekday, category, is_anomaly

3. Save the result as DATA/erm2-nwe9.csv and run:
   python src/experiment_real_plus.py --data DATA/erm2-nwe9.csv

This ensures full reproducibility of the figures and metrics reported in
The course project.

