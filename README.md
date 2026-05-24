# RIfTS

**R**eason **I**ntervals **f**or **T**ime **S**eries.

RIfTS computes the *maximum continuous reason* of a Random Forest classifier
at a sample: the largest hyper-rectangle of perturbations of the input that
preserves the forest classification. Each axis of the rectangle is an
EU-respecting half-open interval on one input feature, and the rectangle is
the optimum of a monotone cost function (default: EU-cell count) over the
lattice of EU-respecting interval constraint functions.

The pipeline has two phases:

1. a **greedy widening** that, starting from the EU cell around the input
   sample, expands one feature at a time, accepting every widening that
   preserves the reason property — this always terminates and always yields a
   maximal reason (a local fixed point of one-cell widening);
2. an optional **tableau-style branch-and-bound** that explores wider
   constrained ICFs from the greedy fixed point and certifies a maximum
   reason whenever it closes the tableau.

Both phases delegate reason verification to a CEGAR-driven vote-MILP that
takes an ICF and a target class and returns either `reason` (every realisable
leaf tuple under the ICF has majority label equal to the target class) or
`counter` together with an explicit counter-witness sample.

The default experimental scope is the
[UCR Univariate Time-Series Classification Archive](https://www.timeseriesclassification.com/)
(2018 release, 128 datasets, 109 after a variable-length / missing-value
filter). A Dash dashboard exposes the per-dataset forest, the EU profile, the
Max-iAXp baseline (Izza et al., IJCAI 2025) and the best RIfTS reason for any
sample.

## Layout

```
rifts-local/
├── README.md                  this file
├── download_ucr.sh            fetch the UCR Univariate 2018 archive
├── init.sh                    prepare current-state/ from factory-setting/ or from scratch
├── dataset/
│   ├── README.md              archive layout, excluded subset, citation
│   └── Univariate_ts/         populated by download_ucr.sh (128 sub-folders, ~800 MB)
├── factory-setting/           pristine ~1 GB snapshot of a fully-run state
│   ├── models/                109 tuned RandomForest .joblib files
│   ├── max-iaxp/              per-dataset Max-iAXp inputs + results.csv
│   ├── sweeps/
│   │   ├── maximal_reasons/sweep.db   greedy widening DB
│   │   └── refinements/sweep.db        tableau refinement DB
│   ├── metrics/               baseline.csv, reference.csv, model_stats.json, …
│   ├── hpo/optuna.db          Optuna HPO log
│   └── experiments_order/     topological order of the included datasets
├── current-state/             working state; created by init.sh, never tracked by git
├── src/                       algorithm core (drifts, tableau, cache)
├── scripts/                   sweep / dashboard / training scripts
│   └── _paths.py              path resolver shared by every script
└── tests/                     pytest suite for the algorithm core
```

## Prerequisites

* Python 3.10 (the algorithm core, the sweeps, the dashboard).
* `bash`, `curl`, `unzip`, `rsync` on `PATH`.
* Graphviz `dot` on `PATH` if you want the PNG export of the Hasse diagram
  from the dashboard.
* A MILP solver visible to PuLP (CBC, the PuLP default, is bundled with the
  package; Gurobi works automatically when its license is set).

## Setup

### 1. Fetch the UCR archive

```sh
./download_ucr.sh
```

Downloads `Univariate2018_ts.zip` (~125 MB) into `dataset/`, unpacks it into
`dataset/Univariate_ts/` (~800 MB, 128 sub-folders) and removes the zip.
Idempotent: subsequent calls detect the populated tree and exit silently.

### 2. Populate `current-state/`

`current-state/` is the working folder that every script reads from and
writes to. It is created by `init.sh`; both `factory-setting/` and `dataset/`
are read-only.

```sh
./init.sh                              # default: --from-factory
./init.sh --from-factory               # explicit form of the default
./init.sh --from-scratch
./init.sh --force [--from-{factory|scratch}]   # DESTRUCTIVE re-init
```

`init.sh` refuses to overwrite an existing `current-state/` unless `--force`
is passed. If `dataset/Univariate_ts/` is missing it bails out with a message
asking you to run `./download_ucr.sh` first.

#### `--from-factory` *(default)*

`current-state/` is rsync-cloned from `factory-setting/`. About 1 GB, under a
minute on a local SSD. When it returns:

* **Trained forests** — 109 hyperparameter-tuned RandomForest models in
  `current-state/models/`, one `.joblib` per UCR dataset.
* **Forest stats** — `current-state/metrics/model_stats.json` lists, per
  forest, the depth and leaf counts, the per-time-point `|EU(i)|` profile,
  the four complexity-axis values, and the dataset-wide ranks.
* **Inclusion gate** — `current-state/metrics/baseline.csv` (our forest's
  test accuracy) and `current-state/metrics/reference.csv` (TSF and 1NN-DTW
  reference accuracies) are present and consistent; the topological order of
  the included datasets is in `current-state/experiments_order/included_topo.csv`.
* **Max-iAXp baseline** — `current-state/max-iaxp/<dataset>/results.csv` and
  `samples.parquet` hold the Max-iAXp solver output (per sample: intervals,
  feature-count, EU-coverage, solver wall time, `ok` / `timeout` / `error`
  status) on every covered dataset.
* **Greedy widening sweep** — `current-state/sweeps/maximal_reasons/sweep.db`
  holds a `reasons` row per certified (dataset, sample) pair: thresholds and
  EU-cell positions for the maximal reason, plus the greedy run statistics.
* **Tableau-refinement sweep** —
  `current-state/sweeps/refinements/sweep.db` holds:
  - a `refinements` row per refinement attempt (started/found ρ, improvement
    flag, closure reason, elapsed time, and the threshold / position JSON of
    the reason found);
  - a `refinement_chain_summary` row per (dataset, sample) chain (greedy ρ,
    final maximum ρ, number of refinements, certified-maximum flag).
  The per-refinement *open-leaves* snapshot (the audit dump used to resume an
  interrupted refinement at the exact tableau leaf) is set to `NULL` in
  every refinement row of the bundled snapshot — this keeps the DB to about
  26 MB. If a script tries to resume a chain whose open-leaves snapshot is
  missing, it transparently restarts that chain from the root of the
  tableau; no state needed.
* **HPO log** — `current-state/hpo/optuna.db` is the Optuna study database
  with every trial of the 50 × 5-fold-CV grid that produced the forests.

You can immediately do any of:

```sh
python scripts/dashboard.py                      # http://127.0.0.1:8050
python scripts/greedy_cegar_sweep.py             # resume the greedy sweep
python scripts/refinement_doubling_sweeper.py --scope general --base-cap-s 60
python scripts/refinement_doubling_sweeper.py --scope axp     --base-cap-s 60
PYTHONPATH=src pytest -q tests                   # algorithm-core test suite
```

#### `--from-scratch`

`current-state/` is created as an empty 7-directory skeleton:

```
current-state/
├── models/
├── metrics/
├── hpo/
├── max-iaxp/
├── sweeps/
│   ├── maximal_reasons/
│   └── refinements/
└── experiments_order/
```

Every artefact has to be produced from the UCR archive. The recommended
sequence is

```sh
./init.sh --from-scratch
python scripts/run_all.py                     # 1. Optuna HPO + RandomForest training, one forest per dataset
python scripts/compute_model_stats.py         # 2. per-forest EU profile and leaf stats -> metrics/model_stats.json
python scripts/fetch_reference.py             # 3. TSF + 1NN-DTW reference accuracies -> metrics/reference.csv
python scripts/build_hasse.py                 # 4. complexity Hasse diagram + topological order
python scripts/max_iaxp/sweep.py              # 5. Max-iAXp baseline sweep
python scripts/greedy_cegar_sweep.py          # 6. greedy widening over the experimental scope
python scripts/refinement_doubling_sweeper.py --scope general --base-cap-s 60
python scripts/refinement_doubling_sweeper.py --scope axp     --base-cap-s 60
```

Expect this to take from a few hours (steps 1–4 on a modern laptop) to many
days (the Max-iAXp baseline times out at 960 s on most large forests; the
RIfTS sweeps are anytime and can be interrupted and resumed). Every script
checkpoints its progress on disk in `current-state/`, so a SIGINT / SIGTERM
can be picked up by re-launching the same command.

#### `--force`

`init.sh --force` deletes the existing `current-state/` before re-creating
it. Combine with `--from-factory` (default) to roll back to the bundled
snapshot, or with `--from-scratch` to wipe and restart from zero. The
factory snapshot and the UCR archive are never affected.

## Running the dashboard

```sh
python scripts/dashboard.py                 # default: 127.0.0.1:8050
python scripts/dashboard.py --port 8060     # avoid a collision with another instance
python scripts/dashboard.py --host 0.0.0.0  # expose on the local network (USE WITH CARE)
python scripts/dashboard.py --debug         # enable Dash debug mode
```

Two tabs:

* **Dataset explorer** — per-dataset forest summary (best HPO params, leaf
  count, unused-feature count, EU stats), the `|EU(i)|` bar chart across
  time-points, and a per-sample comparison panel. The comparison panel has
  two dropdowns (sample, RIfTS source) and shows two stacked plotly panels:
  the RIfTS reason on top (greedy or any improving refinement of the chain,
  selectable from the dropdown) and the Max-iAXp explanation underneath when
  the Max-iAXp solver succeeded for that sample. Each panel and the figure
  title are annotated with the EU-coverage `cov = ρ / N_EU ∈ [0, 1]`.
* **Hasse diagram** — interactive Hasse diagram of the complexity poset
  over the included datasets. Each axis has a per-axis role radio with four
  options: `dom+viz` (axis drives the partial order AND is drawn in the
  node label), `dom` (drives the order, not drawn), `viz` (drawn, does not
  drive), `none` (ignored). Downloads in SVG, PNG, DOT, CSV.

## Running the algorithm-core tests

```sh
PYTHONPATH=src pytest -q tests
```

## Path resolution

Every script imports `_paths` from `scripts/_paths.py`, which exposes:

| Constant       | Default                              | Purpose                                  |
|----------------|--------------------------------------|------------------------------------------|
| `RIFTS_ROOT`   | the directory containing this README  | located by walking up until both         |
|                |                                      | `dataset/` and `factory-setting/` exist  |
| `STATE_ROOT`   | `${RIFTS_ROOT}/current-state`         | every read / write the scripts perform   |
| `FACTORY_ROOT` | `${RIFTS_ROOT}/factory-setting`       | read-only reference snapshot             |
| `DATA_ROOT`    | `${RIFTS_ROOT}/dataset/Univariate_ts` | UCR archive root                         |
| `SRC_ROOT`     | `${RIFTS_ROOT}/src`                   | added to `sys.path` on import            |
| `SCRIPTS_ROOT` | `${RIFTS_ROOT}/scripts`               | added to `sys.path` on import            |

No environment variables are needed: a script can be invoked from any working
directory and `_paths` resolves the right roots from its own file location.

## Acknowledgements

### UCR / UEA archive

All experiments in this repository use the
[UCR / UEA Time Series Archive](https://www.timeseriesclassification.com/),
2018 univariate release, curated by Anthony Bagnall, Hoang Anh Dau, Eamonn
Keogh and collaborators. We are grateful to the authors and to every
contributor who donated, cleaned and documented the 128 datasets the archive
hosts; without them, almost every empirical part of this project would not
exist. If you use RIfTS in published work, please cite the archive paper:

> Dau, H.\,A., Bagnall, A., Kamgar, K., Yeh, C.-C.\,M., Zhu, Y., Gharghabi, S.,
> Ratanamahatana, C.\,A., Keogh, E. (2019). *The UCR Time Series Archive*.
> IEEE/CAA Journal of Automatica Sinica, 6(6):1293–1305.
> [doi:10.1109/JAS.2019.1911747](https://doi.org/10.1109/JAS.2019.1911747)

### Max-iAXp / RFxpl

Our Max-iAXp baseline is computed by
[RFxpl](https://github.com/izzayacine/RFxpl) (Yacine Izza, MIT licence), the
reference implementation of the line of work the baseline implements. RFxpl
is vendored under `third_party/RFxpl/` so the baseline runs out of the box
(see `THIRD_PARTY_NOTICES.md` for the licence text and the one modification
we made). The RFxpl authors ask that any work building on the codebase cite
the following two papers:

> Izza, Y., Marques-Silva, J. (2021). *On Explaining Random Forests with
> SAT*. Proceedings of the 30th International Joint Conference on Artificial
> Intelligence (IJCAI 2021), pages 2584–2591.
> [doi:10.24963/ijcai.2021/356](https://doi.org/10.24963/ijcai.2021/356)

> Izza, Y., Ignatiev, A., Stuckey, P.\,J., Marques-Silva, J. (2024).
> *Delivering Inflated Explanations*. Proceedings of the 38th AAAI
> Conference on Artificial Intelligence (AAAI 2024), pages 12744–12753.
> [doi:10.1609/aaai.v38i11.29170](https://doi.org/10.1609/aaai.v38i11.29170)
