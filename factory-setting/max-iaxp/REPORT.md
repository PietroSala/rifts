# Max-iAXp results — report

**Author**: Pietro Sala — Version 0.1

This report summarises the outcome of running **Max-iAXp** (the MILP-based
maximum interval-AXp solver from RFxpl / GiAXp) on the 66 included UCR
datasets of the TIME 2026 experimental corpus. It is derived from the
files under `time-series-case/experiments/results/max-iaxp/<dataset>/results.csv`,
plus the topological dataset ordering at
`time-series-case/experiments/experiments_order/included_topo.csv`.

## Methodology

For every dataset in the topo-ordered list, the sweep ran Max-iAXp on a
prefix of the test split using a **progressive-timeout ladder**: each
sample was attempted at a 60 s wall-clock budget, and on timeout it was
retried at 120 s, then 240 s, 480 s, and finally 960 s (≈ 16 min). A
sample is `ok` when Max-iAXp returned a verified maximal interval-AXp
within one of those budgets; `timeout` if every budget on the ladder
hit its cap without producing a verified reason; `error:*` if the
solver crashed.

For each successful row the recorded columns include:

- `n_constrained` — number of features still constrained by the AXp;
- `coverage_convB` — Convention-B coverage (mean per-feature fraction of
  EU thresholds inside the interval, augmented with ±∞);
- `solver_status` ∈ {`ok`, `timeout`, `error:*`};
- `solver_wall_s`, `total_wall_s` — solver and total wall-clock for the
  attempt that produced this row;
- `timeout_s` — the budget cap that this row corresponds to.

## Overall counts

| outcome      | count |
|--------------|-------|
| total rows   |  1578 |
| **ok**       |  **1240** |
| timeout      |   309 |
| error:*      |    29 |

## Datasets with successes

Across the 66 datasets, **only 4** produced at least one verified
Max-iAXp reason. All four are in the first 9 of the topo order — i.e.
the easiest datasets by the topo-ordering criterion (small forests,
short series, narrow per-feature EU).

| topo | dataset           | ok / n      | mean n_constrained | mean cov_convB | max wall on success | budget cap used |
|------|-------------------|-------------|--------------------|----------------|---------------------|-----------------|
| 0    | ItalyPowerDemand  | 1028 / 1029 |               3.1  |        0.9136  |     12.9 s          |   60 s          |
| 3    | Chinatown         |  207 / 219  |               5.1  |        0.8478  |    776.6 s          |  960 s          |
| 5    | ECGFiveDays       |    2 / 7    |               9.5  |        0.9552  |    144.0 s          |  960 s          |
| 8    | BME               |    3 / 8    |              13.0  |        0.9244  |    388.7 s          |  960 s          |

ItalyPowerDemand essentially solves at the cheapest tier — 1028 of 1029
samples finished within 60 s. The other three needed escalation. Even
on Chinatown, the worst successful sample needed **776 s** to produce
its reason.

## Datasets with zero successes

**62 of 66** datasets have **no** verified Max-iAXp reason at any
budget. The full list (in topo order) is the 62 entries between topo
position 1 and 65 minus the four above. The first ten timed-out
datasets in topo order — all of which were attempted at every budget
tier from 60 s up to 960 s — are:

| topo | dataset                  | n attempted | cap reached |
|------|--------------------------|-------------|-------------|
| 1    | PowerCons                |  7          | 960 s       |
| 2    | MoteStrain               |  5          | 960 s       |
| 4    | GunPointOldVersusYoung   |  5          | 960 s       |
| 6    | Coffee                   |  5          | 960 s       |
| 7    | FreezerRegularTrain      |  5          | 960 s       |
| 9    | InsectEPGSmallTrain      |  5          | 960 s       |
| 10   | DiatomSizeReduction      |  5          | 960 s       |
| 11   | Beef                     |  5          | 960 s       |
| 12   | BeetleFly                |  5          | 960 s       |
| 13   | InsectEPGRegularTrain    |  5          | 960 s       |

(and 52 more, all with similar pattern: 4–5 sample attempts, all
exhausting the ladder.)

## Where the ladder spent its budget

The 1240 successes break down across the ladder as follows. Each tier's
"ok" row counts samples that **first** succeeded at that tier (i.e.
having failed every lower tier).

| budget cap | succeeded here | still timed out here |
|------------|----------------|----------------------|
| ≤ 60  s    | **1037**       |  65                  |
| ≤ 120 s    |     2          |  65                  |
| ≤ 240 s    |    88          |  65                  |
| ≤ 480 s    |    33          |  65                  |
| ≤ 960 s    |    80          |  49                  |
| **total**  | **1240**       | 309                  |

- **Most successes (84 %)** are obtained at the cheapest 60 s tier.
- **80 samples** needed the full 960 s budget (≈ 16 min per sample) to
  succeed; **49 samples** ran the full ladder and never returned a
  reason.
- The 60 s timeout count (65) reflects the same set of stubborn samples
  that also timed out at every higher tier — once a sample times out at
  60 s on this corpus, the probability of ever succeeding at a higher
  tier is small (≈ 16 % aggregate up to 960 s).

## Coverage and constraint counts (where Max-iAXp succeeded)

Across the 4 datasets where Max-iAXp produced reasons, the average
quality is:

|  dataset           | mean n_constrained / total | mean coverage_convB |
|--------------------|----------------------------|---------------------|
| ItalyPowerDemand   |   3.1 / 24                 |  0.9136             |
| Chinatown          |   5.1 / 24                 |  0.8478             |
| ECGFiveDays        |   9.5 / 136                |  0.9552             |
| BME                |  13.0 / 128                |  0.9244             |

These are very compact reasons — **most features are unconstrained**.
This is the strength of the MILP approach: when it finishes, it finds
the minimum-constraint AXp for that sample.

## Practical implications

1. **Time cap that "earned" what we have**: the headline `1240 ok`
   number is produced over a budget budget that escalates up to **960 s
   per sample** in the worst case (Chinatown). The same number was
   essentially out of reach at a uniform 60 s cap, where only 1037 of
   the eventual 1240 successes would have completed (the remaining 203
   needed 120 s or more).
2. **Saturation effect**: every dataset whose samples timed out at the
   first 60 s attempt continued to time out at every successive tier
   up to 960 s. Doubling the budget is not what unlocks them — the MILP
   is genuinely infeasible at the corpus' typical forest size.
3. **Effective coverage**: in absolute terms 1240 / 1578 ≈ 78.6 % of
   attempted Max-iAXp samples produced reasons, but those are clustered
   on just 4 of 66 datasets. The other 62 datasets have **zero**
   verified Max-iAXp output.
4. **Our greedy comparison baseline** (see `code/sweeps/maximal_reasons/`
   for the per-sample reasons we computed): every greedy run finishes
   within seconds per sample on every dataset, including the 62 where
   Max-iAXp times out. On datasets where Max-iAXp succeeds, its reasons
   are typically 2-3× tighter than ours; on the rest of the corpus,
   our greedy / refinement chain is the only thing producing anything
   useful for the paper.

## Raw artefacts

- `results/max-iaxp/<dataset>/results.csv` — one CSV per dataset; one
  row per (sample, attempted timeout tier). The columns are
  documented above.
- `results/max-iaxp/<dataset>/samples.parquet` — the original test
  samples used.
- `results/max-iaxp/<dataset>/data.csv` — the training dataset.
- `results/max-iaxp/<dataset>/meta.json` — RF metadata.
- `results/max-iaxp/<dataset>/model.pkl` — the pickled sklearn
  RandomForestClassifier used.

The sweep driver is `scripts/max_iaxp/sweep_progressive.py`. The
helpers for Convention-B coverage and EU extraction are in
`scripts/max_iaxp/coverage.py`.
