# Paper summaries

These CSVs aggregate the source records in the other data folders. Start with
`main_results.csv`; the full file-to-paper mapping is in the [data guide](../README.md).
`all_onsets.csv` retains all 784 individual onset records and adds full model names.

`null` means no sustained detection, and `not evaluated` means no available result.
Zoo ranges cover five models; `<=100` indicates detection at the first Phase Zoo
checkpoint. The small-sample file includes additional train-mode comparisons.

Regenerate from the repository root:

```bash
python results.py summaries --data data --out data/summaries
```
