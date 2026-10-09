# Realising grid flexibility from workplace EV charging

Code, data and results for the paper:

> M. Almuwallad, *Realising grid flexibility from workplace electric vehicle charging under imperfect information and tariff signals*, submitted to *Applied Energy*.

The study replays 13,623 real charging sessions from two California workplaces in the open ACN-Data set (JPL and Caltech, July–December 2019). It compares uncontrolled charging, EDF and LLF heuristics, a peak-state receding-horizon LP (nominal, robust and "top-up" variants) and a hindsight optimum. The extensions cover a PV-aware controller, transformer hosting limits, a tariff-design sweep and a held-out 2021 validation. Bills follow the SCE TOU-EV-4 tariff (March 2019, as encoded in acnportal v0.3.3) with a demand charge of 15.51 USD/kW-month.

## Repository layout

| Path | Content |
|---|---|
| `acndc/` | Core package: cleaning, calibration, tariff, LP (HiGHS), simulator, metrics |
| `config.json` | Tariff, site capacities, model and experiment settings |
| `prepare_data.py` | Rebuilds `data/raw` and `data/processed` from the bundled extracts and checks their SHA-256 |
| `run_all.py` | Main experiments: `main`, `topup`, `sens`, `tariff`, `heldout` |
| `run_extensions.py` | Extensions: `pv`, `xfmr`, `tariff2` |
| `diagnose_unmet.py`, `flex_analysis.py` | Unmet-energy decomposition, grid-flexibility and PV-coincidence metrics |
| `make_*.py` | Figures and LaTeX tables for the paper |
| `data/sustaingym/` | ACN-Data session extracts (2019 and 2021) with `SHA256SUMS` |
| `results/` | Every metric CSV/JSON reported in the paper; `loads*/` hold the replayed 5-minute load profiles |
| `paper/` | LaTeX source of the manuscript (elsarticle), tables and figures (PDF and SVG) |
| `tests/` | Pipeline integrity tests on synthetic data |

## Reproduce

Python 3.11 or newer. The results were produced with Python 3.13, numpy 2.5, pandas 3.0, scipy 1.18, highspy 1.15 and pvlib 0.16.

```bash
pip install -r requirements.txt
python prepare_data.py            # build processed sessions from data/sustaingym (about 1 min)
python tests/test_pipeline.py     # integrity tests on synthetic data (about 30 s)

# Simulations (each writes CSV files to results/)
python run_all.py main            # all policies, July-December 2019
python run_all.py topup           # peak-state top-up variants
python run_all.py sens            # sensitivity analysis (August and November)
python run_all.py tariff          # demand-charge sweep 0-31.02 USD/kW-month
python run_all.py heldout         # out-of-sample check on July-August 2021
python run_extensions.py pv       # PV-aware control
python run_extensions.py xfmr     # transformer hosting limits (50% and 35%)
python run_extensions.py tariff2  # tariff archetypes (evening-peak TOU, flat energy)

# Analysis, figures and tables
python diagnose_unmet.py
python flex_analysis.py
python make_paper_assets.py
python make_tariff_figure.py
python make_revision_assets.py
python make_portrait_tables.py
python make_extension_assets.py
python make_compact_tables.py

# Manuscript
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

All scripts are run from the repository root. The full simulation set takes roughly 1–2 hours on one CPU core. You can skip the simulations: the analysis and figure scripts read the shipped `results/`.

## Data

The session data come from ACN-Data (Lee, Li & Low, *ACN-Data: Analysis and Applications of an Open EV Charging Dataset*, ACM e-Energy 2019, https://doi.org/10.1145/3307772.3328313). We use the extracts redistributed under CC-BY-4.0 in the SustainGym v0.1.7 package (PyPI). For unclaimed sessions, SustainGym fills the declared energy and departure with realised values. Because that would leak future information into an online controller, `acndc/sustaingym_source.py` discards those fields and lets the calibrated predictor supply them.

`acndc/pull.py` can also download sessions straight from the official ACN-Data API (token from https://ev.caltech.edu/dataset) to extend the data window.

## Licence

The code is released under the MIT licence (see `LICENSE`). The ACN-Data extracts in `data/sustaingym/` remain under CC-BY-4.0 and must be attributed to the ACN-Data authors as cited above.

## Citation

If you use this repository, please cite the paper (citation to be updated on publication) and ACN-Data. See `CITATION.cff`.

## Contact

Mansour Almuwallad, Yanbu Industrial College, Royal Commission for Jubail and Yanbu, Saudi Arabia. almuwalladm@rcjy.edu.sa · ORCID 0009-0002-9547-0621
