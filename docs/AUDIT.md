# Verification record

Automated checks were run by Codex for Yiming Zhang during release preparation
on 17–21 October 2026. The checks below distinguish saved-result verification
from small execution tests.

| Check | Result | Evidence |
| --- | --- | --- |
| Main tables, 17 October | 280 entries matched | [table_comparison_1017.csv](table_comparison_1017.csv) |
| Current onset tables, 21 October | 472 entries matched, including four BN annotations and 49 earliest-checkpoint qualifiers | [table_comparison_1021.csv](table_comparison_1021.csv) |
| Independent onset reconstruction | 784 cells and 10,976 fields matched; 435 final detections and 349 completed null cells | [tables_1021.json](audit/tables_1021.json) |
| Data validation and relocation | 5,061,086 records validated; 5,156 source-file hashes unchanged; only manifest paths changed | [MANIFEST.csv](../data/MANIFEST.csv), [folder map](data_layout.csv) |
| Maintained regression tests | 95 tests passed | `python -m pytest tests -q` |
| Cluster scripts versus audited release | 30 fixed-input return-value and random-draw comparisons matched | [arithmetic_1018.json](audit/arithmetic_1018.json) |
| Consolidated VRT | 18 recorded fixed-input cases cover full, small and reversed tests across dtypes and seeds | [test_compact_core.py](../tests/test_compact_core.py) |
| CPU and V100 execution | Three smoke jobs completed, covering tests, synthetic generation, model training steps and feature extraction | [smoke_1021.json](audit/smoke_1021.json) |
| Restored figure builds | Nine plotting tasks ran; 65 existing input files unchanged | [figures_1021.json](audit/figures_1021.json), [figure guide](../figures/README.md) |
| Execution archive | 295 files match `d50ebd7` byte for byte | [archive](../archive/README.md), annotated tag `as-run` |

The 472 comparisons cover Tables 1–3 and the nearest-neighbour, small-budget
and reversed-reference tables, including paired gaps. They do not count the
calibration, synthetic-power or runtime tables. The independent reconstruction
uses a separate standard-library reader without importing `results.py`; the
evidence records the manuscript and input hashes.

The maintained aggregate can be checked directly:

```bash
python -m pytest tests -q
python results.py validate --data data
python results.py onsets --data data --out results/onsets.csv
python - <<'PY'
import csv
with open('results/onsets.csv', newline='') as f:
    actual = list(csv.DictReader(f))
with open('docs/onsets_final.csv', newline='') as f:
    expected = list(csv.DictReader(f))
assert actual == expected
print('All 784 onset records match field for field.')
PY
```

The checks retain full validation; no `--no-validate` or protocol bypass is
used. Model training was not repeated. Cluster smoke tests use small generated
inputs and a NumPy version different from the original experiment environment.
Figure-source limitations, including the saved-asset reconstruction in
Appendix G, are documented in the figure guide.

The repository remains private. Author selection of code/data licenses,
the `v1.0-camera-ready` release tag and Zenodo DOI remain steps for public
release; no release date or DOI is asserted in `CITATION.cff` yet.
