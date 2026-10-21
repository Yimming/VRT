"""
release_utils.py — shared guards for the released runners (added for the camera-ready code release, October 2026).

Problems found in the audits of the as-run scripts are fixed through this module without changing any test
arithmetic or random stream:
  1. Non-finite inputs, scores or statistics. A NaN or inf in the features, or a non-finite statistic, used to propagate
     silently: the permutation tests compared permuted statistics with a NaN observation, every comparison was False,
     and the test returned p = 1/(B + 1) as if it had rejected. validate_features(), check_scores(),
     check_statistics() and check_perm_statistic() now raise instead.
  2. Resume by row count. Several runners counted the rows already written for an epoch and reran the whole epoch when
     the count was short, which duplicated the seeds already present and could leave a seed missing while the count
     looked complete. ResumeIndex keys on (epoch, seed) (or (epoch, budget, seed)), checks that every counted row
     belongs to the current run (expect=) and carries valid values (finite, in range), and refuses conflicting or
     invalid rows instead of overwriting or skipping them.
  3. Appending to a damaged or foreign file. open_append() refuses an output file whose header differs from the
     runner's schema and moves a torn last line (a task killed in mid-write) into a sidecar before appending.
  4. Rounded p-values. Two runners wrote permutation p-values with four decimals; (count + 1)/2001 values round to
     0.0500 and then fail "< 0.05". exact_perm_p() recovers the exact value from a rounded one (unambiguous for
     B <= 9,999 permutations) and refuses values that are not close to the lattice; fmt_p() writes p with six
     decimals together with the integer count.
  5. Training resume index. resume_start_epoch() gives the loop index that matches a saved checkpoint_N.
"""
import csv
import math
import os

import numpy as np


# ----------------------------------------------------------------------------- 1. finiteness
def validate_features(*arrays, names=None, allow_zero_rows=False):
    """Raise ValueError if any array holds NaN or inf, or (unless allowed) a zero row that l2 normalization would map
    to the zero vector. Pass allow_zero_rows=True for unnormalized (raw) features, where a zero row is a valid
    point. Returns the arrays unchanged so that it can wrap a loader call."""
    names = names or [f"array{i}" for i in range(len(arrays))]
    for a, name in zip(arrays, names):
        a = np.asarray(a)
        if a.ndim != 2 or a.shape[0] == 0:
            raise ValueError(f"{name}: expected a non-empty 2-d array, got shape {a.shape}")
        if not np.isfinite(a).all():
            bad = int((~np.isfinite(a)).any(axis=1).sum())
            raise ValueError(f"{name}: {bad} rows contain NaN or inf")
        if not allow_zero_rows:
            norms = np.linalg.norm(a, axis=1)
            if (norms == 0).any():
                raise ValueError(f"{name}: {int((norms == 0).sum())} rows have zero norm; l2 normalization is undefined")
    return arrays if len(arrays) > 1 else arrays[0]


def check_scores(*score_arrays, what="c-hat"):
    """Refuse NaN or -inf scores. +inf is allowed: in float32 a query that coincides with one reference point has
    c-hat = +inf (see vrt_low_lib.compute_chat); a query coinciding with two or more (duplicate) reference points gives
    NaN, which this check catches. The median statistics are checked separately (check_statistics,
    check_perm_statistic), so a median that becomes +inf or NaN because too many scores are +inf stops the run too."""
    for c in score_arrays:
        c = np.asarray(c)
        if np.isnan(c).any() or np.isneginf(c).any():
            raise FloatingPointError(f"{what}: {int(np.isnan(c).sum())} NaN and {int(np.isneginf(c).sum())} -inf values; "
                                     "check the input features (duplicate or non-finite points)")


def check_statistics(obs, perm_vals=None, what="statistic"):
    """Raise FloatingPointError if the observed or any permuted statistic is not finite. Call it before computing the
    permutation p-value; a NaN observation would otherwise give p = 1/(B + 1)."""
    if not np.isfinite(obs):
        raise FloatingPointError(f"{what}: observed value is {obs!r}")
    if perm_vals is not None:
        perm_vals = np.asarray(perm_vals)
        if not np.isfinite(perm_vals).all():
            raise FloatingPointError(f"{what}: {int((~np.isfinite(perm_vals)).sum())} of {perm_vals.size} permuted values are not finite")


def check_perm_statistic(value, what="permuted statistic"):
    """Guard for a permutation loop that compares one permuted value at a time (the VRT median tests): refuse a
    non-finite value instead of letting the comparison with the observation silently evaluate to False."""
    if not np.isfinite(value):
        raise FloatingPointError(f"{what} is {value!r}; the test is undefined for these scores")
    return value


# ----------------------------------------------------------------------------- 2. resume by key
def repair_tail(path, sidecar_suffix=".torn"):
    """If the file does not end with a newline, its last line was cut (a task killed in mid-write): move the fragment
    to <path>.torn (appended) and truncate the file. Returns the fragment ('' when nothing was done)."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return ""
    with open(path, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return ""
    raw = open(path, "rb").read()
    cut = raw.rfind(b"\n") + 1
    frag = raw[cut:]
    with open(path + sidecar_suffix, "ab") as side:
        side.write(frag + b"\n")
    with open(path, "wb") as f:
        f.write(raw[:cut])
    return frag.decode("utf-8", errors="replace")


def read_header(path):
    with open(path, newline="") as f:
        return next(csv.reader(f), None)


class ResumeIndex:
    """Rows already present in an output CSV, keyed on (epoch, seed) or (epoch, budget, seed).

    index = ResumeIndex(path, key_cols=("epoch", "seed"), value_cols=("p",), expect={"tag": args.tag, "n_perm": B},
                        ranges={"p": (0, 1)}, header=HEADER)
    todo = [s for s in seeds if not index.has(ep, s)]
    index.report()      # raises on conflicting duplicates, invalid rows or rows of another run; else a one-line summary

    A row counts as complete only when every key column parses, every column in `expect` equals the current run's
    configuration, every value column parses as a finite float inside its range (columns in `nan_ok` may be 'nan',
    columns in `discrete` must take one of the listed values), and `row_check(row)` (if given) returns None. Rows of
    another configuration (another tag, budget, sample size or permutation count) are counted in .foreign and make
    report() refuse by default (foreign="refuse"): an output file holds one run; write to a new path instead of mixing
    runs. Rows that fail the value checks are counted in .invalid and make report() refuse as well. A torn last line
    is moved to <path>.torn first (repair_tail), and a header that differs from the runner's schema is refused."""

    def __init__(self, path, key_cols=("epoch", "seed"), value_cols=(), int_cols=("epoch", "seed"), expect=None,
                 ranges=None, nan_ok=(), discrete=None, row_check=None, header=None, foreign="refuse"):
        self.path, self.key_cols, self.value_cols = path, tuple(key_cols), tuple(value_cols)
        self.expect, self.ranges, self.nan_ok = dict(expect or {}), dict(ranges or {}), set(nan_ok)
        self.discrete, self.row_check, self.foreign_policy = dict(discrete or {}), row_check, foreign
        self.rows, self.duplicates, self.conflicts, self.invalid, self.foreign = {}, 0, [], [], 0
        self.foreign_examples = []
        self.repaired = repair_tail(path)
        self.exists = os.path.exists(path) and os.path.getsize(path) > 0
        if not self.exists:
            return
        if header is not None and read_header(path) != list(header):
            raise RuntimeError(f"{path}: header {read_header(path)} differs from the runner's schema {list(header)}; "
                               "write to a new output file")
        with open(path, newline="") as f:
            for n, r in enumerate(csv.DictReader(f), start=2):
                try:
                    key = tuple(int(r[c]) if c in int_cols else r[c] for c in self.key_cols)
                except (KeyError, ValueError, TypeError):
                    self.invalid.append((n, "key does not parse"))
                    continue
                mismatch = [c for c, v in self.expect.items() if not _same(r.get(c), v)]
                if mismatch:
                    self.foreign += 1
                    if len(self.foreign_examples) < 3:
                        self.foreign_examples.append((n, {c: r.get(c) for c in mismatch}))
                    continue
                val, problem = [], None
                for c in self.value_cols:
                    s = r.get(c)
                    if s is None:
                        problem = f"{c} missing (torn line)"; break
                    if c in self.nan_ok and s.strip().lower() == "nan":
                        val.append(s); continue
                    try:
                        x = float(s)
                    except ValueError:
                        problem = f"{c}={s!r} does not parse"; break
                    if c in self.discrete:
                        if x not in self.discrete[c]:
                            problem = f"{c}={s} is not one of {sorted(self.discrete[c])}"; break
                    else:
                        lo, hi = self.ranges.get(c, (-math.inf, math.inf))
                        if not math.isfinite(x) or not lo <= x <= hi:
                            problem = f"{c}={s} is not finite or out of range [{lo}, {hi}]"; break
                    val.append(s)
                if problem is None and self.row_check is not None:
                    problem = self.row_check(r)
                if problem:
                    self.invalid.append((n, f"{key}: {problem}"))
                    continue
                val = tuple(val)
                if key in self.rows:
                    self.duplicates += 1
                    if val != self.rows[key]:
                        self.conflicts.append(key)
                    continue
                self.rows[key] = val

    def has(self, *key):
        return tuple(key) in self.rows

    def epochs_complete(self, seeds, budgets=None):
        """epochs for which every (seed[, budget]) key is present"""
        eps = {k[0] for k in self.rows}
        if budgets is None:
            return {e for e in eps if all((e, s) in self.rows for s in seeds)}
        return {e for e in eps if all((e, b, s) in self.rows for b in budgets for s in seeds)}

    def report(self, label=""):
        msg = f"resume{(' ' + label) if label else ''}: {len(self.rows)} keys present"
        if self.repaired:
            msg += f", torn last line moved to {self.path}.torn"
        if self.duplicates:
            msg += f", {self.duplicates} duplicate rows ignored"
        problems = []
        if self.foreign:
            text = f"{self.foreign} rows belong to another run (e.g. line {self.foreign_examples[0][0]}: {self.foreign_examples[0][1]})"
            if self.foreign_policy == "refuse":
                problems.append(text + "; an output file holds one run, write to a new --out path")
            else:
                msg += ", " + text + ", ignored"
        if self.conflicts:
            problems.append(f"{len(self.conflicts)} keys have conflicting duplicates, e.g. {self.conflicts[:5]}")
        if self.invalid:
            problems.append(f"{len(self.invalid)} invalid rows, e.g. {self.invalid[:3]}")
        if problems:
            raise RuntimeError(msg + "; " + "; ".join(problems) + "; clean the file before resuming")
        return msg


def _same(stored, expected):
    """compare a stored CSV field with the current run's value, numerically when both are numbers"""
    if stored is None:
        return False
    try:
        return float(stored) == float(expected)
    except (TypeError, ValueError):
        return str(stored) == str(expected)


def count_consistent(p_col, count_col, nperm_col):
    """row_check for runners that write p with its integer permutation count: B must be a positive integer, the count
    an integer in [0, B], and p equal to (count + 1)/(B + 1) within the six-decimal rounding of the file."""
    def check(r):
        try:
            B, c, p = int(r[nperm_col]), int(r[count_col]), float(r[p_col])
        except (KeyError, TypeError, ValueError):
            return f"{count_col}/{nperm_col} missing or not integers"
        if B < 1 or not 0 <= c <= B:
            return f"{count_col}={c} outside 0..{nperm_col}={B}"
        if abs(p - (c + 1) / (B + 1)) > 5e-7 + 1e-12:
            return f"{p_col}={p} disagrees with ({count_col} + 1)/({nperm_col} + 1) = {(c + 1) / (B + 1):.6f}"
        return None
    return check


def open_append(path, header):
    """Open an output CSV for appending. A new or empty file gets the header. An existing file must carry exactly this
    header (otherwise RuntimeError: write to a new file instead of mixing schemas) and a torn last line is moved to
    <path>.torn first, so that the new rows never attach to a fragment. Returns (file, writer)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    if not new:
        old = read_header(path)
        if old != list(header):
            raise RuntimeError(f"{path}: existing header {old} differs from {list(header)}; write to a new output file")
        repair_tail(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(header)
        f.flush()
    return f, w


# ----------------------------------------------------------------------------- 3. p-values
def perm_p(count, n_perm):
    """(count + 1) / (n_perm + 1) as a float"""
    return (count + 1) / (n_perm + 1)


def exact_perm_p(p_rounded, n_perm, decimals=4):
    """Recover the exact permutation p-value k / (n_perm + 1) from a value rounded to `decimals` decimals. The rounding
    error is at most 0.5 x 10^-decimals; multiplied by n_perm + 1 it must stay below 0.5 for the recovery to be
    unambiguous (four decimals: n_perm <= 9,999). Raises ValueError when p is outside (0, 1] or further from the
    lattice than the rounding allows, so that arbitrary numbers are not projected onto it."""
    if (n_perm + 1) * 0.5 * 10 ** (-decimals) >= 0.5:
        raise ValueError(f"{decimals} decimals do not identify k for n_perm = {n_perm}")
    if not (0 < p_rounded <= 1):
        raise ValueError(f"p = {p_rounded!r} is not a permutation p-value")
    x = p_rounded * (n_perm + 1)
    k = int(round(x))
    if k < 1 or abs(x - k) > (n_perm + 1) * 0.5 * 10 ** (-decimals) + 1e-9:
        raise ValueError(f"p = {p_rounded!r} is not within rounding distance of a multiple of 1/{n_perm + 1}")
    return k / (n_perm + 1)


def fmt_p(p):
    """six decimals; exact for permutation p-values with n_perm + 1 <= 999,999 (see exact_perm_p)"""
    return f"{p:.6f}"


# ----------------------------------------------------------------------------- 4. training resume
def resume_start_epoch(save_dir, epochs, fmt="checkpoint_{:03d}.pt"):
    """Loop index to resume a training run from, and the checkpoint to load (or None).

    checkpoint_N holds the model after N completed epochs, so training resumes with loop index N and the next
    checkpoint written is checkpoint_{N+1}. The as-run training script resumed with N + 1, which skipped one epoch
    index and never wrote checkpoint_{N+1}."""
    for ep in range(epochs, -1, -1):
        p = os.path.join(save_dir, fmt.format(ep))
        if os.path.exists(p):
            return ep, p
    return 0, None
