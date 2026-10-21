# Validate the release

From the repository root, after installing `requirements.txt`:

```bash
python -m pytest tests -q
python results.py validate --data data
python results.py onsets --data data --out results/onsets.csv
python - <<'PYTHON'
import csv
from pathlib import Path

def read(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))

assert read("results/onsets.csv") == read("docs/onsets_final.csv")
print("All 784 onset records match.")
PYTHON
```

The current release passes 115 regression tests. Validation covers 5,156
manifest files and 5,061,086 result records. The onset comparison checks
saved results; it does not retrain the models. Earlier CPU and GPU smoke
runs checked small execution cases, including training and extraction.
