# ICE_current

Data-analysis code for the Gorgon MOBO "iceburner" campaign (UROP project):
Pareto-front / hypervolume analysis of the multi-objective optimisation
runs, reconstruction of the physical DT ice-layer geometry from the
optimiser's design multipliers, and random-forest surrogate scans built
with [mille-feuille](https://github.com/aidancrilly/mille-feuille).

## Layout

```
iceburner/          reusable analysis package (import this from notebooks or scripts)
    scaling.py       similarity-scaling physics + DT ice-geometry reconstruction
    database.py      load the sqlite simulation database, filter outlier rows
    pareto.py        hypervolume history, Pareto-front extraction/inspection, plots
    surrogate.py      clean a mille-feuille State, scan a fitted surrogate, plot predictions
notebooks/          thin notebooks that call into iceburner and display results
data/               simulation database(s)
results/            generated outputs (e.g. the Pareto-point CSV)
mille-feuille/      git submodule: MOBO/surrogate library (fork, branch MOBO)
```

`pareto.py` and `database.py`/`scaling.py` implement the analysis for
`notebooks/01_database_and_pareto_analysis.ipynb` and
`notebooks/02_dt_ice_thickness_analysis.ipynb`. `surrogate.py` implements
the analysis for `notebooks/03_surrogate_scan.ipynb` and
`notebooks/04_surrogate_scan_filtered.ipynb`. All four notebooks used to
duplicate this logic (with small, silent divergences between copies); it
now lives in one place.

## Setup

```bash
git clone --recurse-submodules <this repo>
# or, if already cloned: git submodule update --init

conda create -n iceburner python=3.11
conda activate iceburner
pip install -e .                 # iceburner package + numpy/pandas/matplotlib/sklearn/torch/botorch
pip install -e mille-feuille      # the MOBO/surrogate library used by the surrogate notebooks
pip install -e ".[notebooks]"     # jupyter, if you want to run the notebooks
```

## Running the notebooks

Run from the `notebooks/` directory (they load data via a relative
`../data/...` path):

1. `01_database_and_pareto_analysis.ipynb` — loads the database, filters
   outliers, computes hypervolume history and the observed Pareto front,
   and writes `results/pareto_dominating_points.csv`.
2. `02_dt_ice_thickness_analysis.ipynb` — reads that CSV, reconstructs
   physical DT ice-layer geometry, and plots the Pareto trade-offs.
3. `03_surrogate_scan.ipynb` — fits a random-forest surrogate on the raw
   database and scans one input.
4. `04_surrogate_scan_filtered.ipynb` — filters known-bad simulations
   before fitting the surrogate, then runs several input scans, including
   a derived DT ice-thickness scan.

## Notes for anyone continuing this analysis

- The similarity-scaling constants (reference radii, Π parameter, laser
  energy, gas density at 20 MA) and their scaling exponents are now
  defined once in `iceburner/scaling.py`. A previous copy of this code
  used a laser-energy coefficient of `2.1e3` instead of `0.84e3`; `0.84e3`
  (the value used everywhere the Pareto-point inspection actually
  depended on it) is what's kept here.
- `dominating_points.ipynb`'s DT-ice-thickness-vs-current analysis
  previously derived `current_MA` from the normalised objective
  `Y_minus_current` (`= -current / 20`) without correcting for the
  normalisation, making every `current_MA` value 20x too small and
  silently breaking the "reference design" comparison. It now reads the
  already-correct `current` column directly.
