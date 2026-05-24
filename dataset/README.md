# UCR Univariate Archive (2018 release)

This folder hosts the UCR/UEA *Univariate Time-Series Classification Archive*,
2018 release in the `.ts` format expected by `aeon` and `sktime`. The archive
is **not** vendored in the repository; we download it from the canonical
aeon-toolkit / timeseriesclassification.com mirror.

## How to populate

From the `rifts-local/` directory (one level above this folder), run

```
./download_ucr.sh
```

The script will, idempotently:

1. Download
   `https://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip`
   (~125 MB compressed) to `dataset/Univariate2018_ts.zip`.
2. Unpack it to `dataset/Univariate_ts/` (~800 MB extracted, 128 sub-folders,
   one per dataset, each containing `<name>_TRAIN.ts` and `<name>_TEST.ts`).
3. Remove the zip after a successful unpack.

If `dataset/Univariate_ts/` already exists and contains at least one populated
`<name>_TRAIN.ts` file, the script exits silently.

## What is here after the script runs

```
dataset/
├── README.md                  this file
└── Univariate_ts/             128 sub-folders, one per UCR dataset
    ├── ACSF1/
    │   ├── ACSF1.txt
    │   ├── ACSF1_TEST.ts
    │   └── ACSF1_TRAIN.ts
    ├── Adiac/
    │   ├── …
    └── …
```

## Datasets excluded from the experimental scope

Out of the 128 datasets in the archive, 19 are excluded from the experimental
scope of the paper (see `time-series-case/conference_paper.md`, §6) for two
reasons:

* **variable-length or missing-value series** (15): `AllGestureWiimoteX`,
  `AllGestureWiimoteY`, `AllGestureWiimoteZ`, `DodgerLoopDay`,
  `DodgerLoopGame`, `DodgerLoopWeekend`, `Fungi`, `GestureMidAirD1`,
  `GestureMidAirD2`, `GestureMidAirD3`, `GesturePebbleZ1`,
  `GesturePebbleZ2`, `MelbournePedestrian`, `PickupGestureWiimoteZ`,
  `PLAID`, `ShakeGestureWiimoteZ`. A point-feature random forest needs a
  fixed-length feature vector with no missing values.
* **HPO sweep failed silently** (4): `PigAirwayPressure`, `PigArtPressure`,
  `PigCVP`. We chose not to investigate further; out of scope.

The list lives in `EXCLUDED` in `scripts/load_ucr.py` once that script is
ported into RIfTS.

## Source

The archive is the work of the UCR / UEA time-series-classification community
(Anthony Bagnall, Hoang Anh Dau and collaborators), distributed via
[timeseriesclassification.com](https://www.timeseriesclassification.com/).
Cite as:

> Dau, H.\,A., Bagnall, A., Kamgar, K., Yeh, C.-C.\,M., Zhu, Y., Gharghabi, S.,
> Ratanamahatana, C.\,A., Keogh, E. (2019). *The UCR Time Series Archive*.
> IEEE/CAA Journal of Automatica Sinica, 6(6):1293–1305.
