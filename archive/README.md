# Original execution workflow

`as_run/` preserves the pre-consolidation release at commit `d50ebd7`, also
identified by the annotated `as-run` tag. Its 295 files are byte-identical to
that commit: 103 code files, 97 job scripts, 82 task files, four environment
records and nine tools. Treat this snapshot as read-only; maintained code
does not import it.

The task files retain the recorded experiment commands, seeds, budgets,
checkpoint ranges and output filenames. The job scripts contain the cluster
submission settings. Their original machine paths remain intact, so these
commands require adaptation to run elsewhere.

This is the audited release of the execution scripts, including validation
and portability fixes made before consolidation. It is not an untouched
cluster snapshot. [patches_vs_asrun_1018.diff](patches_vs_asrun_1018.diff)
records those earlier changes; [RELEASE_CHECKS_1018.md](RELEASE_CHECKS_1018.md)
records the historical review.

Use the nine modules in the repository root for new work. The October 18
comparison of the cluster scripts and audited release checked 30 fixed-input
cases, including returned values and sampled random draws. Consolidation is
covered separately by the 18 recorded VRT cases in
[test_compact_core.py](../tests/test_compact_core.py), baseline/model/synthetic
regression tests, unchanged result aggregates, and CPU/V100 execution smoke
tests. These checks do not rerun full model training.

The current verification record is in [docs/AUDIT.md](../docs/AUDIT.md).
