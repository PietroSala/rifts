# experiments_order/

Holds the CSV(s) defining the **dataset order** in which the Max-iAXp baseline
sweep is run. Generate one from the dashboard:

1. Open the Hasse tab.
2. Pick the axes / filter / scope you want.
3. Click **CSV (inverse topo)** — saves `hasse_inverse_topo.csv`.
4. Drop it (renamed however you like) into this directory.

The sweep script (`scripts/max_iaxp/sweep_progressive.py`) reads the first
`.csv` here (alphabetical) and uses its `dataset` column as the execution
order — least demanding first.
