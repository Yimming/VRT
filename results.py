"""Validate released results, export clean CSVs, and reproduce paper summaries.

Run ``python results.py --help`` for commands. Only aggregation is performed;
model checkpoints are not needed.
"""
import argparse
import collections
import csv
import glob
import hashlib
import io
import math
import os
from pathlib import Path
import re
import shutil
import statistics as st
import sys

from common import exact_perm_p

FINITE, PVAL, BINARY, UNIT, NAN_OR_UNIT, INT, PVAL_OR_EMPTY, ANY, APPROX_P, FEATURES = range(10)


INT_KEYS = {"epoch", "seed", "n_samples", "n_perm", "M", "k", "rep", "C", "scale_int"}


PROTOCOL = {"reversed": {"direction": {"test_ref"}, "n_perm": {2000}},
            "reversed/large": {"M": {2000}, "k": {60}}, "reversed/small": {"M": {500}, "k": {15}},
            "mmd": {"n_perm": {2000}}, "mmd_raw": {"n_perm": {2000}}, "ed": {"n_perm": {2000}}, "ed_raw": {"n_perm": {2000}},
            "knn": {"k": {3}, "n_perm": {2000}}}


class SchemaError(ValueError):
    pass


def _check(kind, s):
    """True when the string s is an admissible value of the given kind"""
    if kind == ANY:
        return True
    if kind == PVAL_OR_EMPTY and s == "":
        return True
    if s is None or s == "":
        return False
    if kind == NAN_OR_UNIT and s.strip().lower() == "nan":
        return True
    if kind == FEATURES:                                      # a word, not a number: checked before the float conversion
        return s in ("l2", "raw")
    try:
        x = float(s)
    except ValueError:
        return False
    if kind == INT:
        return x == int(x)
    if not math.isfinite(x):
        return False
    if kind in (PVAL, PVAL_OR_EMPTY):
        return 0 < x <= 1
    if kind == BINARY:
        return x in (0.0, 1.0)
    if kind in (UNIT, NAN_OR_UNIT, APPROX_P):
        return 0 <= x <= 1
    return True


KINDS = {FINITE: "finite", PVAL: "p in (0, 1]", BINARY: "0 or 1", UNIT: "in [0, 1]", NAN_OR_UNIT: "nan or in [0, 1]",
         INT: "integer", PVAL_OR_EMPTY: "p in (0, 1] or empty", ANY: "any", APPROX_P: "p in [0, 1]", FEATURES: "l2 or raw"}


TAG = r"(?P<tag>[A-Za-z0-9_]+?)"


CHUNK = r"(_c\d+(-\d+)?)?"


SSL_DS = r"(?P<ds>cifar10|cifar100)(?P<bn>_bntrain)?"


SSL_STATS = {"vrt", "vrt_low", "vrt_rev", "mmd", "ed", "c2st", "mmd_raw", "ed_raw", "mmdagg", "schilling", "henze"}


def ssl_values(stat):
    """value kinds of one SimCLR row: MMDAgg writes a 0/1 decision, C2ST an accuracy and a normal-approximation p,
    Schilling a normal-approximation p, the others a permutation p with acc = nan"""
    if stat == "mmdagg":
        return {"acc": NAN_OR_UNIT, "p": BINARY}
    if stat == "c2st":
        return {"acc": UNIT, "p": APPROX_P}
    if stat == "schilling":
        return {"acc": NAN_OR_UNIT, "p": APPROX_P}
    return {"acc": NAN_OR_UNIT, "p": PVAL}


SCHEMAS = {
    "vrt_fwd": dict(group="vrt_fwd", pattern=rf"^vrt_fwd_{TAG}_(?P<budget>large|small){CHUNK}\.csv$",
                    key=("tag", "budget", "epoch", "seed"), match={"tag": "tag", "budget": "budget"},
                    values={"dc": FINITE, "p": PVAL}),
    "reversed": dict(group="reversed", pattern=rf"^vrt_rev_{TAG}_(?P<budget>large|small){CHUNK}\.csv$",
                     key=("tag", "budget", "direction", "epoch", "seed", "M", "k"), match={"tag": "tag", "budget": "budget"},
                     values={"delta_c": FINITE, "p": PVAL}, optional={"count": INT, "n_perm": INT, "p_exact": PVAL},
                     cross=[("p", "count", "n_perm"), ("p_exact", "p", 2000)]),
    "mmd_p2000": dict(group="mmd", pattern=rf"^mmd_{TAG}_n(?P<n>\d+){CHUNK}\.csv$",
                      key=("tag", "n_samples", "n_perm", "epoch", "seed"), match={"tag": "tag", "n_samples": "n"},
                      values={"mmd2": FINITE, "p": PVAL}, optional={"features": FEATURES}),
    "mmd_raw_p2000": dict(group="mmd_raw", pattern=rf"^mmd_{TAG}_n(?P<n>\d+){CHUNK}\.csv$",
                          key=("tag", "n_samples", "n_perm", "epoch", "seed"), match={"tag": "tag", "n_samples": "n"},
                          values={"mmd2": FINITE, "p": PVAL}, optional={"features": FEATURES}),
    "ed_p2000": dict(group="ed", pattern=rf"^ed_{TAG}_n(?P<n>\d+){CHUNK}\.csv$",
                     key=("tag", "n_samples", "n_perm", "epoch", "seed"), match={"tag": "tag", "n_samples": "n"},
                     values={"ed_val": FINITE, "p": PVAL}, optional={"features": FEATURES}),
    "ed_raw_p2000": dict(group="ed_raw", pattern=rf"^ed_{TAG}_n(?P<n>\d+){CHUNK}\.csv$",
                         key=("tag", "n_samples", "n_perm", "epoch", "seed"), match={"tag": "tag", "n_samples": "n"},
                         values={"ed_val": FINITE, "p": PVAL}, optional={"features": FEATURES}),
    "knn_v2": dict(group="knn", pattern=rf"^knn_{TAG}_n(?P<n>\d+)_part\d*{CHUNK}\.csv$",
                   key=("tag", "n_samples", "k", "epoch", "seed"), match={"tag": "tag", "n_samples": "n"},
                   values={"p_schilling": APPROX_P, "p_henze": PVAL},
                   optional={"henze_count": INT, "n_perm": INT, "p_henze_exact": PVAL, "features": FEATURES},
                   cross=[("p_henze", "henze_count", "n_perm"), ("p_henze_exact", "p_henze", 2000)]),
    "c2st": dict(group="base", pattern=rf"^c2st_{TAG}_b(?P<budgets>[\d-]+){CHUNK}\.csv$",
                 key=("model", "epoch", "budget", "stat", "seed"),
                 match={"model": "tag", "budget": lambda m: {"b" + b for b in m["budgets"].split("-")}, "stat": {"c2st"}},
                 values={"acc": UNIT, "p": APPROX_P}, optional={"features": FEATURES}),
    "bntrain_c2st": "c2st",
    "mmdagg": dict(group="base", pattern=rf"^mmdagg_{TAG}_b(?P<n>\d+){CHUNK}\.csv$",
                   key=("model", "epoch", "budget", "stat", "seed"),
                   match={"model": "tag", "budget": lambda m: {"b" + m["n"]}, "stat": {"mmdagg"}},
                   values={"acc": NAN_OR_UNIT, "p": BINARY}, optional={"features": FEATURES}),
    "bntrain_mmdagg": "mmdagg",
    "ssl": dict(group="ssl", pattern=rf"^ssl_{SSL_DS}_(?P<group>[a-z]+){CHUNK}(_e\d+)?\.csv$",
                key=("model", "epoch", "budget", "stat", "seed"),
                match={"model": lambda m: {m["ds"] + (m["bn"] or "")}, "stat": SSL_STATS, "budget": {"b1500", "b10000", "b5000"}},
                values=ssl_values),
    "ssl_bntrain": dict(group="ssl", alternatives=["ssl", "sslagg"]),
    "sslagg": dict(group="ssl", pattern=rf"^sslagg_{SSL_DS}_(s\d+-\d+_e\d+(_\d+)?|\d+-\d+_e\d+)\.csv$",
                   key=("model", "epoch", "budget", "stat", "seed"),
                   match={"model": lambda m: {m["ds"] + (m["bn"] or "")}, "stat": {"mmdagg"}, "budget": {"b5000"}},
                   values={"acc": NAN_OR_UNIT, "p": BINARY}),
    "ssl_baselines": dict(group="ssl", alternatives=["sslagg"],
                          legacy=r"^(sslbase|sslvrt|sslseeds)_(cifar10|cifar100)_\d+-\d+\.csv$"),
    "mmd_on_chat": dict(group="mmd_on_chat", pattern=rf"^mmdchat_{TAG}{CHUNK}\.csv$",
                        key=("tag", "epoch", "seed"), match={"tag": "tag"},
                        values={"dc": FINITE, "mmd_c": FINITE, "p": PVAL}, optional={"vrt_p": PVAL_OR_EMPTY, "n_inf": INT}),
    "chat_stats": dict(group="chat_stats", pattern=rf"^chatstats_{TAG}{CHUNK}(fix)?\.csv$",
                       key=("tag", "epoch", "seed", "stat"), match={"tag": "tag"},
                       values=lambda stat: {"obs": FINITE, "p": PVAL_OR_EMPTY if stat.startswith(("tr_", "te_")) else PVAL}),
    "ssl_nll": dict(group="ssl_nll", pattern=rf"^ssl_nll_(?P<ds>cifar10|cifar100){CHUNK}\.csv$",
                    key=("epoch",), match={}, file_key=("ds",),
                    values={"tr_eval": FINITE, "te_eval": FINITE, "gap_eval": FINITE, "tr_train": FINITE, "te_train": FINITE,
                            "gap_train": FINITE}),
    "synth": dict(group="synth", alternatives=["synth_clusters", "synth_matched", "synth_plain", "synth_bandwidth"]),
    "synth_clusters": dict(group="synth", pattern=r"^fig4a_clusters\.csv$", key=("cond", "C", "test", "rep"), match={},
                           file_key=("basename",), values={"p": PVAL}),
    "synth_matched": dict(group="synth", pattern=r"^fig4a_matched\.csv$", key=("cond", "test", "budget", "rep"), match={},
                          file_key=("basename",), values={"p": PVAL}),
    "synth_plain": dict(group="synth", pattern=r"^(fig4a_all|fig4a_radial|synth_radial_results)\.csv$",
                        key=("cond", "test", "rep"), match={}, file_key=("basename",), values={"p": PVAL}),
    "synth_bandwidth": dict(group="synth", pattern=r"^mmd_bandwidth\.csv$", key=("cond", "scale", "test", "rep"), match={},
                            file_key=("basename",), values={"p": PVAL, "sigma": FINITE, "d_within": FINITE, "d_between": FINITE}),
    "timing": dict(group="timing", pattern=r"^tests_timing_[A-Za-z0-9_]+\.csv$|^mmdagg_np_jax_[A-Za-z0-9_]+\.csv$",
                   key=("tag", "test", "budget", "backend", "device", "seed"), match={}, file_key=("basename",), unique=False,
                   values={"seconds": FINITE}, optional={"result": ANY, "threads": INT}),
}


NESTED = {"null", "null_trtr"}


def null_cifar_values(method):
    if method == "mmdagg":
        return {"p": BINARY, "acc": NAN_OR_UNIT, "sec": FINITE}
    if method in ("c2st", "schilling"):
        return {"p": APPROX_P, "acc": NAN_OR_UNIT if method == "schilling" else UNIT, "sec": FINITE}
    return {"p": PVAL, "acc": NAN_OR_UNIT, "sec": FINITE}


SCHEMAS.update({
    "null_typeI": dict(group="null_typeI", pattern=r"^typeI_verbatim_[A-Za-z0-9_]+\.csv$", key=("seed",), match={},
                       file_key=("basename",), values={"dc": FINITE, "p": PVAL}),
    "null_cifar_base": dict(group="null_cifar_base", pattern=r"^base_null_[A-Za-z0-9_]+_r\d+-\d+(_official)?\.csv$",
                            key=("arch", "bn_mode", "rep", "net_seed", "split_seed", "seed", "split", "method", "budget"),
                            match={}, file_key=("basename",), values=lambda method: null_cifar_values(method), stat_col="method"),
    "null_cifar_rule": dict(group="null_cifar_rule", pattern=r"^vrt_rule_[A-Za-z0-9_]+_n\d+-\d+\.csv$",
                            key=("arch", "bn_mode", "net", "net_seed", "budget", "seed"), match={}, file_key=("basename",),
                            values={"dc": FINITE, "p": PVAL}),
    "null_cifar_vrt": dict(group="null_cifar_vrt", pattern=r"^vrt_null_[A-Za-z0-9_]+_r\d+-\d+\.csv$",
                           key=("arch", "bn_mode", "rep", "net_seed", "split_seed", "vrt_seed", "split", "budget", "D"), match={},
                           file_key=("basename",), values={"dc": FINITE, "p": PVAL, "cos": FINITE, "feat_sec": FINITE}),
    "official_check": dict(group="official_check", pattern=r"^official_check_[A-Za-z0-9_]+_r\d+-\d+\.csv$",
                           key=("dataset", "features", "rep", "split", "method"), match={}, file_key=("basename",),
                           values={"p": PVAL}),
})


def resolve(family):
    """the schema dict(s) of a family folder: a list of candidate schemas and the legacy pattern (or None)"""
    s = SCHEMAS.get(family)
    if s is None:
        return [], None
    if isinstance(s, str):
        s = SCHEMAS[s]
    if "alternatives" in s:
        return [SCHEMAS[a] for a in s["alternatives"]], s.get("legacy")
    return [s], s.get("legacy")


def schema_for(family, basename, sub=None):
    """(schema, regex match) for a file, ('legacy', None) for a recognised legacy file, (None, None) when unknown.
    `sub` is the subfolder inside a nested Type-I family (null/<sub>/<file> uses the schema of <sub>)."""
    if family in NESTED:
        if sub:
            cands, legacy = resolve(sub)
        else:
            cands, legacy = resolve("null_typeI")
    else:
        cands, legacy = resolve(family)
    for sc in cands:
        m = re.match(sc["pattern"], basename)
        if m:
            routing = sub if family in NESTED and sub else family
            if routing in ("ssl", "ssl_bntrain", "ssl_baselines"):
                # the onset loader reads eval-mode SimCLR files from ssl/ and ssl_baselines/ and train-mode files from
                # ssl_bntrain/ only; a file in the wrong folder would validate and then be silently unused, so refuse it
                is_train = bool(m.groupdict().get("bn"))
                if is_train != (routing == "ssl_bntrain"):
                    mode = "train" if is_train else "eval"
                    expected = "ssl_bntrain" if is_train else ("ssl_baselines" if basename.startswith("sslagg_") else "ssl")
                    raise SchemaError(f"{family}/{basename}: {mode}-mode SimCLR file belongs in {expected}/")
            return sc, m
    if legacy and re.match(legacy, basename):
        return "legacy", None
    return None, None


def read_rows(path):
    """header and rows of a CSV with NUL bytes removed; torn lines (field count differs) are returned separately"""
    raw = open(path, "rb").read()
    n_nul = raw.count(b"\x00")
    text = raw.replace(b"\x00", b"").decode("utf-8")
    lines = text.splitlines()
    if not lines:
        return None, [], [], n_nul, 0
    header = next(csv.reader([lines[0]]))
    rows, torn, repeated_header = [], [], 0
    for n, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        f = next(csv.reader([line]))
        if f == header:
            repeated_header += 1
            continue
        if len(f) != len(header):
            torn.append((n, line))
            continue
        rows.append(f)
    return header, rows, torn, n_nul, repeated_header


def _cross_problem(spec, f, idx):
    """one cross-field rule: (p_col, count_col, nperm_col) requires B >= 1, 0 <= count <= B and p == (count + 1)/(B + 1)
    within six-decimal rounding; (exact_col, p_col, B) requires exact == round(p x (B + 1))/(B + 1) (the recovery of a
    four-decimal p for the fixed B of the historical runs). Rules whose columns are absent are skipped."""
    a, b, c = spec
    if isinstance(c, int):                                    # (exact_col, p_col, B)
        if a not in idx or b not in idx:
            return None
        try:
            exact, p = float(f[idx[a]]), float(f[idx[b]])
        except ValueError:
            return f"{a}/{b} do not parse"
        k = round(p * (c + 1))
        if abs(exact - k / (c + 1)) > 1e-8 or abs(p * (c + 1) - k) > (c + 1) * 0.5e-4 + 1e-9:
            return f"{a}={exact} does not match {b}={p} on the 1/{c + 1} lattice"
        return None
    if b not in idx or c not in idx:
        return None
    try:
        B, cnt, p = int(f[idx[c]]), int(f[idx[b]]), float(f[idx[a]])
    except ValueError:
        return f"{b}/{c} are not integers"
    if B < 1 or not 0 <= cnt <= B:
        return f"{b}={cnt} outside 0..{c}={B}"
    if abs(p - (cnt + 1) / (B + 1)) > 5e-7 + 1e-12:
        return f"{a}={p} disagrees with ({b} + 1)/({c} + 1) = {(cnt + 1) / (B + 1):.6f}"
    return None


def validate_rows(schema, match, header, rows, path, protocol=None):
    """Check every row of one file against its schema. Returns (key_idx, records) with records = [(key, values)] in
    file order (verbatim duplicates kept, so that the caller can count them); raises SchemaError on the first
    problems (header without the required columns, a row that does not parse, an integer key that is not an integer,
    a value outside its kind, a cross-field inconsistency, a row that disagrees with the file name, or, when
    `protocol` is given, a row whose protocol columns differ from the fixed values). Integer key columns are
    normalised (int), so '042' and '42' are one key."""
    missing = [c for c in schema["key"] if c not in header]
    vals = schema["values"]
    fixed_vals = vals if isinstance(vals, dict) else None
    if fixed_vals:
        missing += [c for c in fixed_vals if c not in header]
    if missing:
        raise SchemaError(f"{path}: header {header} lacks the columns {missing}")
    idx = {c: i for i, c in enumerate(header)}
    key_cols = list(schema["key"])
    file_key = tuple(match[g] if g != "basename" else os.path.basename(path) for g in schema.get("file_key", ()))
    checks = []
    for col, want in schema.get("match", {}).items():
        if col not in idx:
            raise SchemaError(f"{path}: header lacks the column {col}")
        allowed = {match[want]} if isinstance(want, str) else (want(match) if callable(want) else set(want))
        checks.append((idx[col], col, allowed))
    proto = []
    if protocol:
        for col, allowed in protocol.items():
            if col in idx:
                proto.append((idx[col], col, allowed))
    optional = {c: k for c, k in schema.get("optional", {}).items() if c in idx}
    cross = [spec for spec in schema.get("cross", []) if spec[0] in idx]
    records, problems = [], []
    for n, f in enumerate(rows, start=2):
        problem = None
        for i, col, allowed in checks:
            if f[i] not in allowed:
                problem = f"{col}={f[i]!r} does not belong to this file (allowed {sorted(allowed)})"; break
        if problem is None:
            for i, col, allowed in proto:
                try:
                    v = int(f[i]) if col in INT_KEYS else f[i]
                except ValueError:
                    v = f[i]
                if v not in allowed:
                    problem = f"{col}={f[i]!r} is not the protocol value {sorted(allowed)}"; break
        key = None
        if problem is None:
            key = []
            for c in key_cols:
                v = f[idx[c]]
                if c in INT_KEYS:
                    try:
                        v = int(v)
                    except ValueError:
                        problem = f"{c}={v!r} is not an integer"; break
                key.append(v)
        if problem is None:
            kinds = fixed_vals if fixed_vals is not None else vals(f[idx[schema.get("stat_col", "stat")]])
            for col, kind in list(kinds.items()) + list(optional.items()):
                if not _check(kind, f[idx[col]]):
                    problem = f"{col}={f[idx[col]]!r} is not {KINDS[kind]}"; break
        if problem is None:
            for spec in cross:
                problem = _cross_problem(spec, f, idx)
                if problem:
                    break
        if problem:
            problems.append((n, problem))
            if len(problems) >= 5:
                break
            continue
        key = file_key + tuple(key) + (() if schema.get("unique", True) else (n,))
        values = tuple(f[idx[c]] for c in kinds) + tuple(f[idx[c]] for c in optional)
        records.append((key, values))
    if problems:
        raise SchemaError(f"{path}: {len(problems)}+ invalid rows, e.g. " + "; ".join(f"line {n}: {msg}" for n, msg in problems))
    return [idx[c] for c in key_cols], records


def protocol_for(schema, match):
    """fixed protocol values for a file: the group's rule plus the budget-specific rule (reversed large / small)"""
    rules = dict(PROTOCOL.get(schema["group"], {}))
    budget = match.get("budget") if match else None
    if budget:
        rules.update(PROTOCOL.get(f"{schema['group']}/{budget}", {}))
    return rules or None


def validate_tree(root, families=None, verbose=False, protocol=True):
    """Validate every recognised CSV under root (all result families or the given ones) and check that no key occurs
    with two different values anywhere in a group; with protocol=True (default) rows must also carry the fixed protocol
    values of the 50-seed runs (PROTOCOL). Returns a summary dict; raises SchemaError on the first problem, also when
    the tree holds no result file at all."""
    seen = {}
    summary = {"files": 0, "records": 0, "legacy": 0, "unknown": [], "torn": 0, "cross_file_duplicates": 0}
    if not os.path.isdir(root):
        raise SchemaError(f"{root} is not a folder")
    for family in sorted(os.listdir(root)):
        fdir = os.path.join(root, family)
        if not os.path.isdir(fdir) or (families and family not in families):
            continue
        if family not in SCHEMAS and family not in NESTED:
            continue
        paths = sorted(Path(fdir).rglob("*.csv")) if family in NESTED else sorted(Path(fdir).glob("*.csv"))
        for file_path in paths:
            name, path = file_path.name, str(file_path)
            sub = file_path.relative_to(fdir).parts[0] if file_path.parent != Path(fdir) else None
            schema, m = schema_for(family, name, sub)
            if schema is None:
                summary["unknown"].append(os.path.join(family, name)); continue
            if schema == "legacy":
                summary["legacy"] += 1; continue
            header, rows, torn, n_nul, _ = read_rows(path)
            if header is None:
                continue
            if torn:
                raise SchemaError(f"{path}: {len(torn)} torn lines, e.g. line {torn[0][0]}: {torn[0][1][:80]!r}")
            _, records = validate_rows(schema, m.groupdict(), header, rows, path,
                                       protocol=protocol_for(schema, m.groupdict()) if protocol else None)
            group = schema["group"]
            for key, values in records:
                k = (group,) + key
                prev = seen.get(k)
                if prev is None:
                    seen[k] = (hash(values), path)
                elif prev[0] != hash(values):
                    raise SchemaError(f"key {key} of group {group} has different values in {prev[1]} and {path}")
                elif prev[1] != path:
                    summary["cross_file_duplicates"] += 1
            summary["files"] += 1; summary["records"] += len(records)
            if verbose:
                print(f"  {family}/{name}: {len(records)} records")
    if summary["unknown"]:
        raise SchemaError(f"{len(summary['unknown'])} files with unrecognised names, e.g. {summary['unknown'][:5]}")
    if summary["files"] == 0:
        raise SchemaError(f"{root}: no result file of the families {sorted(families) if families else 'known to the schemas'} found")
    return summary


EXCLUDE_DIRS = {"c100_nan_bug": "SimCLR CIFAR-100 MMD on c-hat before the +inf cap (NaN statistics, p = 1/2001)",
                "dup_backup": "c-hat diagnostic files before the duplicate-row cleanup (both concurrent jobs' rows)",
                "_superseded": "10-seed kNN runs superseded by knn_v2",
                "null10": "10-seed null runs superseded by the 50-seed null folders"}


FOUR_DECIMAL = {"reversed": ("p", "p_exact", "count", "n_perm"), "knn_v2": ("p_henze", "p_henze_exact", "henze_count", "n_perm")}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_columns(family, header, rows, path):
    """Add the exact-p column to a four-decimal family. New-format files (integer count and n_perm columns) get
    p_exact = (count + 1)/(n_perm + 1); as-run files, which hold p with four decimals and no count, get
    round(p x 2001)/2001, the recovery for the 2,000 permutations of every historical run of these two families
    (the task files pass --n_perm 2000). The consistency of p with count/n_perm and of an existing exact column with p
    is checked by the schema's cross-field rules before this function runs."""
    if family not in FOUR_DECIMAL:
        return header, rows, ""
    pcol, newcol, ccol, ncol = FOUR_DECIMAL[family]
    if pcol not in header or newcol in header:
        return header, rows, ""
    pi = header.index(pcol)
    if ccol in header and ncol in header:
        ci, ni = header.index(ccol), header.index(ncol)
        rows = [f + [f"{(int(f[ci]) + 1) / (int(f[ni]) + 1):.9f}"] for f in rows]
        return header + [newcol], rows, f"added {newcol} from {ccol}"
    try:
        rows = [f + [f"{exact_perm_p(float(f[pi]), 2000):.9f}"] for f in rows]
    except ValueError as e:
        raise SchemaError(f"{path}: {e}") from e
    return header + [newcol], rows, f"added {newcol}"


class Exporter:
    def __init__(self, src, dst, dry_run, allow_unknown):
        self.src, self.dst, self.dry_run, self.allow_unknown = src, dst, dry_run, allow_unknown
        self.seen = {}                                        # (group, key) -> (hash(values), relpath)
        self.manifest, self.excluded, self.cleaned = [], [], []
        self.n_records = self.n_cross_dup = 0

    def record(self, rel, size, digest, info):
        self.manifest.append([rel, info.get("rows", ""), size, digest, info.get("nul_bytes", 0), info.get("torn", 0),
                              info.get("dup_rows", 0), info.get("note", "")])
        if info.get("nul_bytes") or info.get("torn") or info.get("dup_rows") or info.get("note"):
            self.cleaned.append((rel, info))

    def copy(self, s, d, rel, note):
        if not self.dry_run:
            os.makedirs(os.path.dirname(d), exist_ok=True); shutil.copy2(s, d)
        self.record(rel, os.path.getsize(s), "" if self.dry_run else sha256(d), {"note": note})

    def export_csv(self, s, d, rel, family, schema, match):
        header, rows, torn, n_nul, rep_header = read_rows(s)
        info = {"nul_bytes": n_nul, "torn": len(torn), "dup_rows": rep_header}
        if header is None:
            info.update(rows=0, note="empty file")
            if not self.dry_run:
                os.makedirs(os.path.dirname(d), exist_ok=True); open(d, "w").close()
            self.record(rel, 0, "" if self.dry_run else sha256(d), info)
            return
        _, records = validate_rows(schema, match, header, rows, s)
        kept, seen_here = [], set()
        for f, (key, values) in zip(rows, records):
            t = tuple(f)
            if t in seen_here:
                info["dup_rows"] += 1; continue
            seen_here.add(t)
            k = (schema["group"],) + key
            prev = self.seen.get(k)
            if prev is None:
                self.seen[k] = (hash(values), rel)
            elif prev[0] != hash(values):
                raise SchemaError(f"{rel}: key {key} occurs with different values here and in {prev[1]}")
            else:
                self.n_cross_dup += 1
            kept.append(f)
        header, kept, note = exact_columns(family, header, kept, s)
        info.update(rows=len(kept), note=note)
        self.n_records += len(kept)
        if not self.dry_run:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            with open(d, "w", newline="") as out:
                w = csv.writer(out); w.writerow(header); w.writerows(kept)
        self.record(rel, os.path.getsize(s) if self.dry_run else os.path.getsize(d), "" if self.dry_run else sha256(d), info)

    def run(self):
        for root, dirs, files in os.walk(self.src):
            rel_root = os.path.relpath(root, self.src)
            family = None if rel_root == "." else rel_root.split(os.sep)[0]
            for dname in [x for x in dirs if x in EXCLUDE_DIRS]:
                self.excluded.append((os.path.join(rel_root, dname) if rel_root != "." else dname, EXCLUDE_DIRS[dname]))
                dirs.remove(dname)
            parts = [] if rel_root == "." else rel_root.split(os.sep)
            sub = parts[1] if len(parts) > 1 else None
            for name in sorted(files):
                rel = os.path.join(rel_root, name) if rel_root != "." else name
                s, d = os.path.join(root, name), os.path.join(self.dst, rel)
                if not name.endswith(".csv"):
                    self.copy(s, d, rel, "copied verbatim (not a result CSV)"); continue
                schema, m = schema_for(family, name, sub) if family in SCHEMAS or family in NESTED else (None, None)
                if schema == "legacy":
                    self.copy(s, d, rel, "legacy file of the 10-seed stage, copied verbatim, not used for onset summaries"); continue
                if schema is None:
                    if not self.allow_unknown:
                        raise SchemaError(f"{rel}: no schema recognises this file name (pass --allow_unknown to copy it verbatim)")
                    self.copy(s, d, rel, "unrecognised name, copied verbatim, not validated"); continue
                self.export_csv(s, d, rel, sub if family in NESTED and sub else family, schema, m.groupdict())
        if not self.dry_run and self.manifest:
            os.makedirs(self.dst, exist_ok=True)
            with open(os.path.join(self.dst, "MANIFEST.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["path", "rows", "bytes", "sha256", "nul_bytes_removed", "torn_lines_dropped", "duplicate_rows_dropped", "note"])
                w.writerows(self.manifest)
            with open(os.path.join(self.dst, "EXCLUDED.txt"), "w") as f:
                for dname, why in self.excluded:
                    f.write(f"{dname}\t{why}\n")


class ResultRoutes:

    """Read checkpoint results from one data directory; cache each CSV once."""

    def __init__(self, data):
        self.BF = str(Path(data).expanduser().resolve())
        self.SSLB = str(Path(self.BF) / "ssl_baselines")
        self._files = {}

    CIFAR = ["v7", "r50", "ablA", "ablB", "ablC", "ablD"] + [f"zoo_c10_m{i}" for i in range(5)] + [f"zoo_c100_m{i}" for i in range(5)]

    OTHER = ["phase_bs8", "phase_bs128", "phase_bs1024", "wp", "imgnet_simple", "imgnet_adv"]

    SSL = {"simclr_c10": "cifar10", "simclr_c100": "cifar100"}

    def parse(self, f, pcol, acccol):
        """one file, parsed once: (model, stat, budget) -> epoch -> seed -> (p, acc); files without those columns key on None"""
        key = (f, pcol, acccol)
        if key in self._files:
            return self._files[key]
        out = collections.defaultdict(lambda: collections.defaultdict(dict))
        raw = open(f, "rb").read().replace(b"\x00", b"")
        for r in csv.DictReader(io.StringIO(raw.decode())):
            try:
                k = (r.get("model"), r.get("stat"), r.get("budget"))
                acc = float(r[acccol]) if acccol else float("nan")
                out[k][int(r["epoch"])].setdefault(int(r["seed"]), (float(r[pcol]), acc))
            except (KeyError, ValueError):
                continue
        self._files[key] = out
        return out

    def read(self, patterns, pcol="p", acccol=None, filt=None):
        """epoch -> {seed: (p, acc)} over the files of the patterns, rows matching filt (model / stat / budget)"""
        d = collections.defaultdict(dict)
        for pat in patterns:
            for f in sorted(glob.glob(pat)):
                for (model, stat, budget), eps in self.parse(f, pcol, acccol).items():
                    if filt and any({"model": model, "stat": stat, "budget": budget}.get(k) != v for k, v in filt.items()):
                        continue
                    for e, seeds in eps.items():
                        for sd, v in seeds.items():
                            d[e].setdefault(sd, v)
        return d

    def cell(self, tag, mode, stat, N):
        """epoch -> {seed: (p, acc)}; mode eval or train"""
        if tag in self.SSL:
            ds = self.SSL[tag]
            lab = ds if mode == "eval" else f"{ds}_bntrain"
            if stat == "mmdagg" and N == 5000:
                pats = [f"{self.SSLB}/sslagg_{ds}_*.csv"] if mode == "eval" else [f"{self.BF}/ssl_bntrain/sslagg_{ds}_bntrain_*.csv"]
                return self.read(pats, filt={"model": lab, "budget": "b5000", "stat": "mmdagg"})
            st = {"vrt": "vrt" if N == 10000 else "vrt_low"}.get(stat, stat)
            pats = [f"{self.BF}/ssl/ssl_{ds}_*_c*.csv"] if mode == "eval" else [f"{self.BF}/ssl_bntrain/ssl_{ds}_bntrain_*_c*.csv"]
            return self.read(pats, "p", "acc", filt={"model": lab, "stat": st, "budget": f"b{N}"})
        t = tag if mode == "eval" else f"{tag}_bntrain"
        b = "large" if N == 10000 else "small"
        if stat == "vrt":
            return self.read([f"{self.BF}/vrt_fwd/vrt_fwd_{t}_{b}_c*.csv"])
        if stat == "vrt_rev":
            return self.read([f"{self.BF}/reversed/vrt_rev_{t}_{b}_c*.csv"])
        if stat in ("mmd", "ed", "mmd_raw", "ed_raw"):
            base = {"mmd": "mmd_p2000", "ed": "ed_p2000", "mmd_raw": "mmd_raw_p2000", "ed_raw": "ed_raw_p2000"}[stat]
            return self.read([f"{self.BF}/{base}/{stat.split('_')[0]}_{t}_n{N}_c*.csv"])
        if stat in ("henze", "schilling"):
            return self.read([f"{self.BF}/knn_v2/knn_{t}_n{N}_part*.csv"], f"p_{stat}")
        if stat == "c2st":
            pats = [f"{self.BF}/c2st/c2st_{t}_b*_c*.csv"] + ([f"{self.BF}/bntrain_c2st/c2st_{t}_b*_c*.csv"] if mode == "train" else [])
            return self.read(pats, "p", "acc", filt={"model": t, "budget": f"b{N}"})
        if stat == "mmdagg":
            pats = [f"{self.BF}/mmdagg/mmdagg_{t}_b{N}_c*.csv"] + ([f"{self.BF}/bntrain_mmdagg/mmdagg_{t}_b{N}_c*.csv"] if mode == "train" else [])
            return self.read(pats)
        raise ValueError(stat)

    def checkpoints(self, tag):
        if tag in self.SSL:
            return list(range(1, 1001))
        d = self.cell(tag, "eval", "vrt", 10000)
        return sorted(d)

    STATS_LARGE = [("vrt", 10000), ("mmd", 10000), ("mmdagg", 5000), ("c2st", 10000), ("ed", 10000),
                   ("mmd_raw", 10000), ("ed_raw", 10000), ("henze", 10000), ("schilling", 10000), ("vrt_rev", 10000)]

    STATS_SMALL = [("vrt", 1500), ("mmd", 1500), ("mmdagg", 1500), ("c2st", 1500), ("ed", 1500),
                   ("henze", 1500), ("schilling", 1500), ("vrt_rev", 1500)]


SEEDS = set(range(42, 92))


FAMILIES_READ = ["vrt_fwd", "reversed", "mmd_p2000", "ed_p2000", "mmd_raw_p2000", "ed_raw_p2000", "knn_v2", "c2st",
                 "bntrain_c2st", "mmdagg", "bntrain_mmdagg", "ssl", "ssl_bntrain", "ssl_baselines"]


TABLE_STATS = ["vrt", "mmd", "mmdagg", "c2st", "ed"]


GROUPS = [("ResNet20-FRN (CIFAR-10)", ["v7"]),
          ("ResNet-18 C100 Zoo (x5)", [f"zoo_c100_m{i}" for i in range(5)]),
          ("ResNet-18 C10 Zoo (x5)", [f"zoo_c10_m{i}" for i in range(5)]),
          ("ResNet-50 (CIFAR-100)", ["r50"]),
          ("Phase Zoo bs8", ["phase_bs8"]), ("Phase Zoo bs128", ["phase_bs128"]), ("Phase Zoo bs1024", ["phase_bs1024"]),
          ("SimCLR CIFAR-10", ["simclr_c10"]), ("SimCLR CIFAR-100", ["simclr_c100"]),
          ("A: baseline", ["ablA"]), ("B: 10x wd", ["ablB"]), ("C: strong aug", ["ablC"]), ("D: wd + aug", ["ablD"]),
          ("5-layer CNN (CIFAR-10)", ["wp"]),
          ("DeepResNet50 simple (ImageNet)", ["imgnet_simple"]), ("DeepResNet50 adv. (ImageNet)", ["imgnet_adv"])]


NO_BN = {"v7", "wp"}


def decide(stat, vals, where=""):
    """decision rule of the paper; refuses values that cannot be a p-value, decision or accuracy"""
    ps = [p for p, _ in vals]
    if stat == "mmdagg":
        if any(p not in (0.0, 1.0) for p in ps):
            raise ValueError(f"{where}: MMDAgg decisions must be 0 or 1")
        return sum(ps) / len(ps) >= 0.6
    lo = 0.0 if stat in ("c2st", "schilling") else 1e-12          # normal-approximation p may round to 0; permutation p > 0
    if any(not (lo <= p <= 1) for p in ps):
        raise ValueError(f"{where}: p-values outside (0, 1] (NaN would count as 'not rejected')")
    if stat == "c2st":
        accs = [a for _, a in vals]
        if any(not (0 <= a <= 1) for a in accs):
            raise ValueError(f"{where}: C2ST accuracies outside [0, 1]")
        return st.median(ps) < 0.05 and st.median(accs) > 0.55
    return st.median(ps) < 0.05


def exact_p_cell(routes, exact_perm_p, tag, mode, stat, N):
    """epoch -> {seed: p_exact} for the two four-decimal families of the CIFAR-style models. New-format rows carry the
    integer permutation count and n_perm, and p_exact = (count + 1)/(n_perm + 1) is taken from them; as-run rows hold
    p with four decimals and no count, and the exact value is round(p x 2001)/2001 (every historical run of these two
    families used 2,000 permutations, which the protocol validation enforces where the column exists)."""
    t = tag if mode == "eval" else f"{tag}_bntrain"
    if stat == "vrt_rev":
        pats, pcol, ccol = [f"{routes.BF}/reversed/vrt_rev_{t}_{'large' if N == 10000 else 'small'}_c*.csv"], "p", "count"
    else:
        pats, pcol, ccol = [f"{routes.BF}/knn_v2/knn_{t}_n{N}_part*.csv"], "p_henze", "henze_count"
    out = {}
    for pat in pats:
        for f in sorted(glob.glob(pat)):
            raw = open(f, "rb").read().replace(b"\x00", b"")
            for r in csv.DictReader(io.StringIO(raw.decode())):
                ep, sd = int(r["epoch"]), int(r["seed"])
                if r.get(ccol) not in (None, "") and r.get("n_perm") not in (None, ""):
                    px = (int(r[ccol]) + 1) / (int(r["n_perm"]) + 1)
                else:
                    px = exact_perm_p(float(r[pcol]), 2000)
                out.setdefault(ep, {}).setdefault(sd, px)
    return out


def analyse(routes, exact_perm_p, tag, mode, stat, N, cps, stored_p):
    d = routes.cell(tag, mode, stat, N)
    if not d:
        return None
    comp = sorted(e for e in d if set(d[e]) >= SEEDS)
    partial = [e for e in sorted(d) if e not in comp]
    row = {"tag": tag, "mode": mode, "stat": stat, "budget": N, "n_epochs": len(comp), "first_epoch": "",
           "last_epoch": "", "last_checkpoint": cps[-1] if cps else "", "onset": "", "gap_before_onset": "",
           "unevaluated_after_onset": "", "borderline_epochs": "", "partial_epochs": len(partial), "status": ""}
    if not comp:                                            # results exist but no epoch has all 50 seeds: say so
        row["status"] = f"no complete epoch ({len(partial)} epochs with fewer than 50 seeds)"
        return row
    vals = {e: [d[e][s] for s in sorted(SEEDS)] for e in comp}
    if stat in ("vrt_rev", "henze") and not stored_p:
        if tag in routes.SSL:                                # SimCLR files hold six-decimal p (exact on the 1/2001 lattice)
            vals = {e: [(exact_perm_p(p, 2000), a) for p, a in v] for e, v in vals.items()}
        else:
            px = exact_p_cell(routes, exact_perm_p, tag, mode, stat, N)
            vals = {e: [(px[e][s], a) for s, (p, a) in zip(sorted(SEEDS), v)] for e, v in vals.items()}
    rej = {e: decide(stat, vals[e], f"{tag} {mode} {stat} {N} epoch {e}") for e in comp}
    e1 = next((e for i, e in enumerate(comp) if all(rej[x] for x in comp[i:])), None)
    borderline = [e for e in comp if stat != "mmdagg" and st.median(p for p, _ in vals[e]) == 0.05]
    row.update(first_epoch=comp[0], last_epoch=comp[-1], onset="" if e1 is None else e1,
               borderline_epochs=";".join(map(str, borderline)))
    if e1 is not None and cps:
        earlier = [e for e in comp if e < e1]
        e0 = earlier[-1] if earlier else None
        row["gap_before_onset"] = sum(1 for c in cps if (e0 is None or c > e0) and c < e1 and c not in comp)
        row["unevaluated_after_onset"] = sum(1 for c in cps if c > e1 and c not in comp)
        row["status"] = "final" if row["gap_before_onset"] == 0 and row["unevaluated_after_onset"] == 0 else "provisional"
    elif cps:
        row["unevaluated_after_onset"] = sum(1 for c in cps if c > comp[-1])
        row["status"] = "null, last checkpoint evaluated" if row["unevaluated_after_onset"] == 0 else "null, provisional"
    return row


def fmt(onsets):
    """paper format for a group of models: null, N, min--max, with (k) when fewer than all detect"""
    present = [o for o in onsets if o is not None]
    if not present:
        return "null"
    s = str(min(present)) if min(present) == max(present) else f"{min(present)}--{max(present)}"
    return s + (f" ({len(present)})" if len(present) < len(onsets) else "")


def print_tables(rows):
    look = {(r["tag"], r["mode"], r["stat"], r["budget"]): (None if r["onset"] == "" else r["onset"]) for r in rows}

    def val(tags, mode, stat):
        N = 5000 if stat == "mmdagg" else 10000
        got = [look.get((t, mode, stat, N), "missing") for t in tags]
        if "missing" in got:
            return "missing"
        return fmt(got)

    print("\nTable 1 (eval mode): " + " | ".join(TABLE_STATS))
    for label, tags in GROUPS:
        print(f"  {label:32s} " + " | ".join(val(tags, "eval", s) for s in TABLE_STATS))
    print("\nTable 2 (eval -> train, BatchNorm models): " + " | ".join(TABLE_STATS))
    for label, tags in GROUPS:
        if tags[0] in NO_BN:
            continue
        print(f"  {label:32s} " + " | ".join(f"{val(tags, 'eval', s)} -> {val(tags, 'train', s)}" for s in TABLE_STATS))
    print("\nTable 3 (unnormalized features, eval -> train): mmd_raw | ed_raw")
    for label, tags in GROUPS:
        cells = []
        for s in ("mmd_raw", "ed_raw"):
            cells.append(val(tags, "eval", s) if tags[0] in NO_BN else f"{val(tags, 'eval', s)} -> {val(tags, 'train', s)}")
        print(f"  {label:32s} " + " | ".join(cells))


def export_results(a):
    src, dst = os.path.abspath(os.path.expanduser(a.src)), os.path.abspath(os.path.expanduser(a.dst))
    if not os.path.isdir(src):
        raise SystemExit(f"{src} is not a folder")
    if not a.dry_run and os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} exists and is not empty; export into a new folder so that stale files cannot survive")
    if not os.path.isdir(os.path.join(src, "ssl_baselines")):
        print("note: no ssl_baselines folder inside src; the SimCLR eval-mode MMDAgg files will be missing", file=sys.stderr)
    # the export is written to a private partial folder beside dst and moved into place only when it is complete, so a
    # refused export never leaves half a tree behind and never deletes anything it did not create
    partial = None if a.dry_run else f"{dst}.partial-{os.getpid()}"
    ex = Exporter(src, dst if a.dry_run else partial, a.dry_run, a.allow_unknown)
    try:
        ex.run()
    except SchemaError as e:
        if partial and os.path.isdir(partial):
            shutil.rmtree(partial, ignore_errors=True)
        raise SystemExit(f"export stopped: {e}")
    if not ex.manifest:
        if partial and os.path.isdir(partial):
            shutil.rmtree(partial, ignore_errors=True)
        raise SystemExit(f"{src}: no files found, nothing exported")
    if partial:
        if os.path.isdir(dst):
            os.rmdir(dst)                                     # it was empty (checked above)
        os.rename(partial, dst)
    print(f"{len(ex.manifest)} files {'would be ' if a.dry_run else ''}exported to {dst}; {ex.n_records} records; "
          f"{ex.n_cross_dup} keys repeated in another file with the same values; {len(ex.excluded)} folders excluded")
    for rel, info in ex.cleaned:
        if info.get("note", "").startswith(("added", "p ", "copied", "legacy", "unrecognised")) and not (
                info.get("nul_bytes") or info.get("torn") or info.get("dup_rows")):
            continue                                        # routine notes are in the manifest
        print(f"  cleaned {rel}: {info}")
    notes = {}
    for rel, info in ex.cleaned:
        notes[info.get("note", "")] = notes.get(info.get("note", ""), 0) + 1
    for note, n in sorted(notes.items()):
        if note:
            print(f"  {n} files: {note}")


def load_routes(data):
    return ResultRoutes(data), exact_perm_p


def compute_onsets(data, *, stored_p=False, validate=True, protocol=True):
    """Return the paper's onset records, preserving incomplete checkpoints explicitly."""
    if validate:
        try:
            summary = validate_tree(str(data), families=FAMILIES_READ, protocol=protocol)
        except SchemaError as e:
            raise SchemaError(f"result files not valid: {e}") from e
        print(f"validated {summary['files']} files, {summary['records']} records; "
              f"{summary['legacy']} legacy files skipped")
    routes = ResultRoutes(data)
    rows = []
    for tag in routes.CIFAR + routes.OTHER + list(routes.SSL):
        cps = routes.checkpoints(tag)
        if not cps:
            print(f"{tag}: no VRT results found, skipped", file=sys.stderr)
            continue
        for mode in ("eval", "train"):
            for stat, budget in routes.STATS_LARGE + routes.STATS_SMALL:
                row = analyse(routes, exact_perm_p, tag, mode, stat, budget, cps, stored_p)
                if row:
                    rows.append(row)
    if not rows:
        raise SchemaError(f"{data}: no onset cells found")
    return rows


def write_records(path, rows):
    if not rows:
        raise SchemaError("no records to write")
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def onset_report(args):
    rows = compute_onsets(args.data, stored_p=args.stored_p, validate=not args.no_validate,
                          protocol=not args.no_protocol_check)
    write_records(args.out, rows)
    gaps = sum(r["gap_before_onset"] not in ("", 0) for r in rows)
    tails = sum(r["unevaluated_after_onset"] not in ("", 0) for r in rows)
    partial = sum(bool(r["partial_epochs"]) for r in rows)
    print(f"{len(rows)} cells written to {args.out}; {gaps} with gaps before onset, "
          f"{tails} with unevaluated tails, {partial} with incomplete seed sets")
    for row in rows:
        if row["n_epochs"] == 0:
            print(f"  incomplete: {row['tag']} {row['mode']} {row['stat']} {row['budget']}: {row['status']}")
    print_tables(rows)


def manifest_check(root):
    """Verify every payload listed in the release manifest, including non-CSV files."""
    root = Path(root).expanduser().resolve()
    path = root / "MANIFEST.csv"
    if not path.is_file():
        raise SchemaError(f"{path}: manifest not found")
    count, records = 0, 0
    seen = set()
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            rel = row["path"]
            if rel in seen:
                raise SchemaError(f"{path}: duplicate manifest path {rel}")
            seen.add(rel)
            p = (root / rel).resolve()
            if not p.is_relative_to(root) or not p.is_file():
                raise SchemaError(f"{rel}: missing file or path outside the data folder")
            if p.stat().st_size != int(row["bytes"]) or sha256(p) != row["sha256"]:
                raise SchemaError(f"{rel}: size or SHA-256 does not match the manifest")
            count += 1
            records += int(row["rows"]) if row["rows"] else 0
    print(f"manifest verified: {count} files, {records} recorded result rows")
    return count, records


def validate_report(args):
    summary = validate_tree(args.data)
    print(f"validated {summary['files']} CSV files, {summary['records']} records; "
          f"{summary['legacy']} legacy files skipped; "
          f"{summary['cross_file_duplicates']} equal keys across files")
    if not args.no_manifest:
        manifest_check(args.data)


def csv_records(pattern):
    for path in sorted(glob.glob(str(pattern))):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                yield path, row


def add_unique(group, key, value, context):
    if key in group and group[key] != value:
        raise SchemaError(f"{context}: conflicting values for {key}")
    group.setdefault(key, value)


def type_i_report(args):
    """Single-test null rates and the separate 50-seed decision-rule control."""
    families = ["null_cifar_vrt", "null_cifar_base", "null_cifar_rule", "official_check"]
    validate_tree(args.data, families=families)
    counts = collections.defaultdict(dict)
    for path, row in csv_records(Path(args.data) / "null_cifar_vrt/vrt_null_*.csv"):
        if row["budget"] == "large":
            key = (row["split"], row["arch"], row["bn_mode"], "vrt")
            add_unique(counts[key], int(row["rep"]), float(row["p"]) < .05, path)
    for path, row in csv_records(Path(args.data) / "null_cifar_base/base_null_*.csv"):
        method, p = row["method"], float(row["p"])
        key = (row["split"], row["arch"], row["bn_mode"], method)
        add_unique(counts[key], int(row["rep"]), p >= .5 if method == "mmdagg" else p < .05, path)
        if method == "c2st":
            rule_key = (row["split"], row["arch"], row["bn_mode"], "c2st_acc55")
            add_unique(counts[rule_key], int(row["rep"]), p < .05 and float(row["acc"]) > .55, path)
    if not counts:
        raise SchemaError("no CIFAR null results found")
    arches = ["r18_c10", "r18_c100", "r50_c100", "simclr_c10", "frn_c10"]
    methods = ["vrt", "mmd", "mmdagg", "c2st", "ed"]
    for split, label in [("resplit", "Table 5: independent random partitions"),
                         ("official", "Appendix C: original train/test split (shared test images)")]:
        print(f"\n{label}: rejections/repetitions (percent), eval mode")
        print("model | " + " | ".join(methods))
        for arch in arches:
            cells = []
            for method in methods:
                hits = counts.get((split, arch, "eval", method), {})
                if not hits:
                    raise SchemaError(f"missing null cell: {split}, {arch}, {method}")
                n, k = len(hits), sum(hits.values())
                cells.append(f"{k}/{n} ({100*k/n:.1f}%)")
            print(arch + " | " + " | ".join(cells))
    print("C2ST here uses p < 0.05 without the held-out accuracy threshold.")
    networks = collections.defaultdict(dict)
    for path, row in csv_records(Path(args.data) / "null_cifar_rule/vrt_rule_*.csv"):
        key = (row["arch"], row["bn_mode"], row["budget"], int(row["net"]))
        add_unique(networks[key], int(row["seed"]), float(row["p"]), path)
    if networks:
        groups = collections.defaultdict(list)
        for (arch, mode, budget, net), values in sorted(networks.items()):
            if set(values) != SEEDS:
                raise SchemaError(f"{arch}/{mode}/{budget}/net {net}: expected seeds 42 through 91")
            groups[(arch, mode, budget)].append(st.median(values.values()) < .05)
        print("\n50-seed decision rule on untrained networks: flagged/networks")
        for key, hits in sorted(groups.items()):
            print(" / ".join(key) + f": {sum(hits)}/{len(hits)}")
    controls = collections.defaultdict(dict)
    for path, row in csv_records(Path(args.data) / "official_check/official_check_*.csv"):
        key = (row["dataset"], row["features"], row["split"], row["method"])
        add_unique(controls[key], int(row["rep"]), float(row["p"]) < .05, path)
    if controls:
        print("\nSeparate pixel-feature / shuffled-extraction controls (not the Appendix C figure)")
        for key, hits in sorted(controls.items()):
            print(" / ".join(key) + f": {sum(hits.values())}/{len(hits)}")
    if args.out:
        rows = [dict(split=s, arch=a, mode=m, method=t, rejections=sum(v.values()), repetitions=len(v))
                for (s, a, m, t), v in sorted(counts.items())]
        write_records(args.out, rows)


def supplementary_onsets(args):
    rows = compute_onsets(args.data)
    look = {(r["tag"], r["mode"], r["stat"], r["budget"]):
            None if r["onset"] == "" else r["onset"] for r in rows}
    if args.command == "small-sample":
        print("\nTable 7: 1,500 points per split; eval mode | " + " | ".join(TABLE_STATS))
        for label, tags in GROUPS:
            cells = []
            for method in TABLE_STATS:
                values = [look.get((t, "eval", method, 1500), "missing") for t in tags]
                cells.append("missing" if "missing" in values else fmt(values))
            print(label + " | " + " | ".join(cells))
        selected = [r for r in rows if r["budget"] == 1500 and r["stat"] in TABLE_STATS]
    else:
        print("\nTable 8: eval mode, 10,000 points per split | standard | reversed | delay")
        for label, tags in GROUPS:
            forward = [look.get((t, "eval", "vrt", 10000)) for t in tags]
            reverse = [look.get((t, "eval", "vrt_rev", 10000)) for t in tags]
            gaps = [b-a for a, b in zip(forward, reverse) if a is not None and b is not None]
            print(label + f" | {fmt(forward)} | {fmt(reverse)} | {fmt(gaps)}")
        zoo = [t for t in ResultRoutes.CIFAR if t.startswith("zoo_")]
        delays = [look[(t, "eval", "vrt_rev", 10000)] - look[(t, "eval", "vrt", 10000)] for t in zoo]
        print(f"Ten zoo models: median delay {st.median(delays):g} epochs")
        selected = [r for r in rows if r["budget"] == 10000 and r["stat"] in {"vrt", "vrt_rev"}]
    if args.out:
        write_records(args.out, selected)


def score_transform_report(args):
    validate_tree(args.data, families=["mmd_on_chat", "vrt_fwd", "mmd_p2000", "ssl"])
    routes = ResultRoutes(args.data)
    result = []
    print("Appendix I: model | VRT | MMD on c-hat | MMD on embeddings | complete checkpoints")
    for tag in routes.CIFAR + routes.OTHER + list(routes.SSL):
        by_epoch = collections.defaultdict(dict)
        for path, row in csv_records(Path(args.data) / f"mmd_on_chat/mmdchat_{tag}_c*.csv"):
            value = (float(row["p"]), float(row["dc"]), row.get("vrt_p", ""))
            add_unique(by_epoch[int(row["epoch"])], int(row["seed"]), value, path)
        if not by_epoch:
            raise SchemaError(f"{tag}: no MMD-on-score results")
        complete = sorted(e for e in by_epoch if set(by_epoch[e]) >= SEEDS)
        rejects = {e: st.median(by_epoch[e][s][0] for s in SEEDS) < .05 for e in complete}
        onset = next((e for i, e in enumerate(complete) if all(rejects[c] for c in complete[i:])), None)
        cps = routes.checkpoints(tag)
        vrt_row = analyse(routes, exact_perm_p, tag, "eval", "vrt", 10000, cps, False)
        emb_row = analyse(routes, exact_perm_p, tag, "eval", "mmd", 10000, cps, False)
        missing_tail = [e for e in cps if onset is not None and e > onset and e not in complete]
        partial = len(set(by_epoch) - set(complete))
        comparison = []
        threshold_crossings = 0
        if tag in routes.SSL:
            original = routes.cell(tag, "eval", "vrt", 10000)
            for ep, values in by_epoch.items():
                for seed, (_, _, check_p) in values.items():
                    if check_p and ep in original and seed in original[ep]:
                        reference_p = original[ep][seed][0]
                        comparison.append(abs(float(check_p) - reference_p))
                        threshold_crossings += (float(check_p) < .05) != (reference_p < .05)
            tolerance, compared = 1e-6, "VRT p of the check seed"
        else:
            dc = {}
            for path, row in csv_records(Path(args.data) / f"vrt_fwd/vrt_fwd_{tag}_large_c*.csv"):
                dc[(int(row["epoch"]), int(row["seed"]))] = float(row["dc"])
            for ep, values in by_epoch.items():
                for seed, (_, score, _) in values.items():
                    if (ep, seed) in dc:
                        comparison.append(abs(score-dc[(ep, seed)]))
            tolerance, compared = 1e-5, "median score difference"
        output = dict(tag=tag, vrt_onset=vrt_row["onset"], mmd_score_onset="" if onset is None else onset,
                      mmd_embedding_onset=emb_row["onset"], n_epochs=len(complete), partial_epochs=partial,
                      unevaluated_after_onset=len(missing_tail), comparison=compared,
                      compared_pairs=len(comparison), mismatches=sum(d > tolerance for d in comparison),
                      maximum_difference=max(comparison, default=0.0), threshold_crossings=threshold_crossings)
        result.append(output)
        print(f"{tag} | {output['vrt_onset']} | {output['mmd_score_onset']} | "
              f"{output['mmd_embedding_onset']} | {len(complete)}/{len(cps)}")
        print(f"  {compared}: {len(comparison)} pairs, {output['mismatches']} differences above {tolerance:g}; "
              f"max difference {output['maximum_difference']:g}, {threshold_crossings} cross p = 0.05")
    paired = [r for r in result if r["vrt_onset"] != "" and r["mmd_score_onset"] != ""]
    earlier = sum(r["mmd_score_onset"] < r["vrt_onset"] for r in paired)
    equal = sum(r["mmd_score_onset"] == r["vrt_onset"] for r in paired)
    later = sum(r["mmd_score_onset"] > r["vrt_onset"] for r in paired)
    print(f"MMD on c-hat versus VRT: {earlier} earlier, {equal} equal, {later} later "
          f"among {len(paired)} models detected by both")
    if args.out:
        write_records(args.out, result)


def runtime_report(args):
    records = []
    for tag in ["r50_ep61", "v7_ep38"]:
        path = Path(args.data) / "timing" / f"tests_timing_{tag}_vrt64.csv"
        if not path.is_file():
            raise SchemaError(f"{path}: float64 paper timings not found")
        schema, match = schema_for("timing", path.name)
        header, rows, torn, _, _ = read_rows(path)
        if torn:
            raise SchemaError(f"{path}: torn timing rows")
        validate_rows(schema, match.groupdict(), header, rows, path)
        by_budget = collections.defaultdict(dict)
        for row in map(lambda fields: dict(zip(header, fields)), rows):
            if row["test"] != "vrt":
                continue
            if row["tag"] != tag or row["device"] != "cpu" or row["backend"] != "numpy" or int(row["threads"]) != 8:
                raise SchemaError(f"{path}: unexpected runtime setting")
            add_unique(by_budget[int(row["budget"])], int(row["seed"]), float(row["seconds"]), path)
        for budget in [1500, 10000]:
            times = by_budget[budget]
            if set(times) != set(range(42, 52)):
                raise SchemaError(f"{path}, {budget}: expected the ten seeds 42 through 51")
            records.append(dict(model=tag, budget=budget, median_seconds=st.median(times.values()), seeds=len(times)))
    print("Table 9: VRT float64, CPU, eight threads; median seconds over ten seeds")
    print("points per split | ResNet-50 | ResNet20-FRN")
    for budget in [1500, 10000]:
        values = [r["median_seconds"] for r in records if r["budget"] == budget]
        print(f"{budget} | " + " | ".join(f"{v:.2f}" for v in values))
    if args.out:
        write_records(args.out, records)


def synthetic_load(path, key):
    """{key(row): [p, ...]}"""
    out = collections.defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            out[key(r)].append(float(r["p"]))
    return out


def synthetic_cell(ps, latex=False):
    """count of rejections; the percentage is shown with one decimal (199/200 is 99.5%, not 100%)"""
    if not ps:
        return "-"
    n = sum(p < 0.05 for p in ps)
    return f"{n}/{len(ps)}" if latex else f"{n:4d}/{len(ps):<4d} ({100 * n / len(ps):5.1f}%)"


def synthetic_pct(ps):
    """percentage of rejections with one decimal (the paper's text rounds to integers where it says so)"""
    return round(100 * sum(p < 0.05 for p in ps) / len(ps), 1) if ps else None


def synthetic_report(a):
    root = Path(a.data).expanduser().resolve()
    clusters = root / "synth/fig4a_clusters.csv"
    matched = root / "synth/fig4a_matched.csv"
    if not matched.is_file():
        matched = Path(__file__).resolve().parent / "figures/fig_data_s50/fig4a_matched.csv"
    bw_path = str(root / "synth/mmd_bandwidth.csv")
    for path in (clusters, matched, Path(bw_path)):
        if not path.is_file():
            raise SchemaError(f"{path}: synthetic input missing")
        schema, match = schema_for("synth", path.name)
        header, rows, torn, _, _ = read_rows(path)
        if torn:
            raise SchemaError(f"{path}: torn result rows")
        _, records = validate_rows(schema, match.groupdict(), header, rows, path)
        if len({key for key, _ in records}) != len(records):
            raise SchemaError(f"{path}: repeated synthetic repetition keys")
    CL = synthetic_load(clusters, lambda r: (int(r["C"]), r["cond"], r["test"]))
    MA = synthetic_load(matched, lambda r: (int(r["budget"]), r["cond"], r["test"]))
    BW = synthetic_load(bw_path, lambda r: (r["cond"], float(r["scale"]), r["test"]))

    print("== Synthetic power panel (Figure 3(a) shows groups 1, 2 and 4), percent rejections (VRT, MMD)")
    fig = [("within-class concentration, training +5%", CL[(10, "tight_0.05", "VRT")], CL[(10, "tight_0.05", "MMD")]),
           ("concentrated subpopulation, 10% of training at 3 kappa_w", CL[(10, "memo_0.10", "VRT")], CL[(10, "memo_0.10", "MMD")]),
           ("single-cloud concentration, kappa 250 -> 256.7", MA[(5000, "conc_0.3", "VRT")], MA[(5000, "conc_0.3", "MMD")]),
           ("single-cloud mean direction, 2 degrees", MA[(5000, "mean_2deg", "VRT")], MA[(5000, "mean_2deg", "MMD")]),
           ("radial spread, log-sd 0.05 -> 0.20", MA[(5000, "S2_radial_tail", "VRT")], MA[(5000, "S2_radial_tail", "MMD")])]
    for name, v, m in fig:
        print(f"   {name:58s} VRT {synthetic_cell(v)}   MMD {synthetic_cell(m)}")

    print("\n== Class-cluster model (D = 500, N = 5,000 per sample, kappa_w = 2,500, matched budget, 2,000 permutations)")
    rows_cl = [(10, "null", "Null, both samples kappa_w"), (10, "tight_0.05", "Within-class concentration, training +5%"),
               (10, "memo_0.10", "Concentrated subpopulation, 10% at 3 kappa_w"), (10, "shift_4", "Per-class location shift, 4 degrees"),
               (100, "null", "Null, C = 100"), (100, "tight_0.05", "Within-class concentration +5%, C = 100"),
               (100, "memo_0.10", "Concentrated subpopulation 10%, C = 100"), (100, "shift_4", "Per-class shift 4 degrees, C = 100")]
    for C, c, name in rows_cl:
        print(f"   C={C:3d} {name:48s} VRT {synthetic_cell(CL[(C, c, 'VRT')])}   MMD {synthetic_cell(CL[(C, c, 'MMD')])}"
              f"   median p VRT {st.median(CL[(C, c, 'VRT')]) if CL[(C, c, 'VRT')] else float('nan'):.3f}"
              f" MMD {st.median(CL[(C, c, 'MMD')]) if CL[(C, c, 'MMD')] else float('nan'):.3f}")
    print("   tightening sweep (C = 10):")
    for rel in ("0.03", "0.05", "0.07", "0.10", "0.30"):
        c = "tight_" + rel
        print(f"      +{int(float(rel) * 100):2d}%   VRT {synthetic_cell(CL[(10, c, 'VRT')])}   MMD {synthetic_cell(CL[(10, c, 'MMD')])}")

    print("\n== Single cloud (fig4a_matched.csv), rejections out of 200 at each budget; VRT, MMD, norm tests")
    conds = [("null_vmf", "Null, vMF(250) against vMF(250)"), ("conc_0.3", "Concentration, kappa 250 -> 256.7"),
             ("mean_2deg", "Mean direction, 2 degrees"), ("mix_0.05", "Mixture, 5% of vMF(750)"),
             ("S0_null", "Radial null"), ("S1_radial_mean", "Radial scale x1.03"), ("S2_radial_tail", "Radial spread 0.05 -> 0.20"),
             ("S3_angular", "Concentration with radii"), ("S4_angular_plus_radial", "Concentration with radii and scale x1.03")]
    for b in (5000, 1500, 500):
        print(f"   budget {b}")
        for c, name in conds:
            parts = [f"{t} {synthetic_cell(MA[(b, c, t)])}" for t in ("VRT", "MMD", "NORM-MED", "NORM-DISP") if MA.get((b, c, t))]
            print(f"      {name:44s} " + "   ".join(parts))

    print("\n== Fixed-bandwidth control (C = 10, mmd_bandwidth.csv)")
    sig = [float(r["sigma"]) for r in csv.DictReader(open(bw_path))]
    dw = [float(r["d_within"]) for r in csv.DictReader(open(bw_path))]
    db = [float(r["d_between"]) for r in csv.DictReader(open(bw_path))]
    print(f"   median-heuristic sigma {st.median(sig):.3f} (min {min(sig):.3f}, max {max(sig):.3f}); median within-class distance "
          f"{st.median(dw):.3f}; between-class {st.median(db):.3f}")
    scales = sorted({k[1] for k in BW if k[2] == "MMD"})
    for c in ("null", "tight_0.05"):
        parts = [f"VRT {synthetic_cell(BW[(c, 0.0, 'VRT')])}"] + [f"MMD x{s:g} {synthetic_cell(BW[(c, s, 'MMD')])}" for s in scales]
        print(f"   {c:12s} " + "   ".join(parts))

    if a.latex:
        print("\n== LaTeX rows (rejections out of 200; VRT & MMD)")
        print("% class-cluster table")
        for C, c, name in rows_cl:
            print(f"{name} & {synthetic_cell(CL[(C, c, 'VRT')], True)} & {synthetic_cell(CL[(C, c, 'MMD')], True)} \\\\")
        print("% tightening sweep")
        print("Training tightening (percent) & " + " & ".join(f"+{int(float(r) * 100)}" for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("VRT & " + " & ".join(synthetic_cell(CL[(10, 'tight_' + r, 'VRT')], True) for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("MMD & " + " & ".join(synthetic_cell(CL[(10, 'tight_' + r, 'MMD')], True) for r in ("0.03", "0.05", "0.07", "0.10", "0.30")) + " \\\\")
        print("% single cloud, budget 5000")
        for c, name in conds[:7]:
            parts = [synthetic_cell(MA[(5000, c, t)], True) if MA.get((5000, c, t)) else "--" for t in ("VRT", "MMD")]
            print(f"{name} & " + " & ".join(parts) + " \\\\")
        print("% bandwidth control")
        for c in ("null", "tight_0.05"):
            print(f"{c} & {synthetic_cell(BW[(c, 0.0, 'VRT')], True)} & " + " & ".join(synthetic_cell(BW[(c, s, 'MMD')], True) for s in scales) + " \\\\")


def parse_seed_spec(spec):
    seeds = set()
    for part in spec.split(","):
        if "-" in part:
            low, high = map(int, part.split("-", 1))
            if low > high:
                raise SchemaError(f"invalid seed range {part}")
            seeds.update(range(low, high+1))
        else:
            seeds.add(int(part))
    return seeds


def aggregate_new_runs(paths, seeds):
    """Aggregate evaluate.py output separately from the historical result families."""
    expected = set(seeds)
    required = {"model", "epoch", "method", "budget", "features", "protocol", "seed", "n_perm", "backend",
                "statistic", "p", "accuracy", "decision"}
    valid_methods = {"vrt", "reversed", "mmd", "ed", "c2st", "mmdagg", "schilling", "henze", "mmd-scores"}
    groups = collections.defaultdict(lambda: collections.defaultdict(dict))
    checkpoints = collections.defaultdict(set)
    for filename in paths:
        with open(filename, newline="") as f:
            reader = csv.DictReader(f)
            if not required.issubset(reader.fieldnames or ()):
                raise SchemaError(f"{filename}: missing columns {sorted(required-set(reader.fieldnames or ())) }")
            for line, row in enumerate(reader, 2):
                where = f"{filename}:{line}"
                try:
                    epoch, seed, budget, perms = map(int, (row["epoch"], row["seed"], row["budget"], row["n_perm"]))
                except (TypeError, ValueError) as e:
                    raise SchemaError(f"{where}: invalid epoch, seed, budget or n_perm") from e
                method = row["method"]
                if method not in valid_methods or row["features"] not in {"l2", "raw"} or perms < 1 or budget < 1:
                    raise SchemaError(f"{where}: invalid method, feature mode, budget or permutation count")
                if seed not in expected:
                    raise SchemaError(f"{where}: seed {seed} outside the requested seed set")
                if not row["model"] or not row["protocol"] or not row["backend"]:
                    raise SchemaError(f"{where}: empty model, protocol or backend")
                key = (row["model"], method, budget, row["features"], row["protocol"], perms, row["backend"])
                try:
                    if method == "mmdagg":
                        p = float(row["decision"])
                        if p not in (0.0, 1.0):
                            raise ValueError("MMDAgg decision must be 0 or 1")
                        if row["p"] != "":
                            raise ValueError("MMDAgg has a binary decision, not a p-value")
                    else:
                        p = float(row["p"])
                        low = 0 if method in {"c2st", "schilling"} else 1e-12
                        if not low <= p <= 1:
                            raise ValueError("invalid p-value")
                    acc = float(row["accuracy"]) if method == "c2st" else None
                    if acc is not None and not 0 <= acc <= 1:
                        raise ValueError("invalid held-out accuracy")
                    if row["statistic"] and not math.isfinite(float(row["statistic"])):
                        raise ValueError("non-finite statistic")
                except (TypeError, ValueError) as e:
                    raise SchemaError(f"{where}: {e}") from e
                add_unique(groups[key][epoch], seed, (p, acc, row["statistic"], row["decision"]), where)
                checkpoints[row["model"]].add(epoch)
    if not groups:
        raise SchemaError("no new-run records found")
    output = []
    for key, epochs in sorted(groups.items()):
        model, method, budget, features, protocol, perms, backend = key
        complete = sorted(e for e in epochs if set(epochs[e]) == expected)
        partial = sorted(set(epochs)-set(complete))
        decisions = {e: decide(method, [value[:2] for value in epochs[e].values()], f"{model}/{method}/{e}") for e in complete}
        onset = next((e for i, e in enumerate(complete) if all(decisions[x] for x in complete[i:])), None)
        observed_checkpoints = sorted(checkpoints[model])
        earlier = [e for e in complete if onset is not None and e < onset]
        previous = earlier[-1] if earlier else None
        before = [e for e in observed_checkpoints if onset is not None and e < onset
                  and (previous is None or e > previous) and e not in complete]
        after = [e for e in observed_checkpoints if onset is not None and e > onset and e not in complete]
        missing = sorted(set(observed_checkpoints)-set(complete))
        status = "no complete epoch" if not complete else "null on evaluated checkpoints" if onset is None else "detected on evaluated checkpoints"
        if before or after or partial:
            status += "; incomplete coverage"
        output.append(dict(model=model, method=method, budget=budget, features=features, protocol=protocol,
                           n_perm=perms, backend=backend, n_epochs=len(complete),
                           onset="" if onset is None else onset, first_epoch=complete[0] if complete else "",
                           last_epoch=complete[-1] if complete else "", partial_epochs=len(partial),
                           missing_seed_epochs=";".join(map(str, partial)),
                           missing_checkpoints=";".join(map(str, missing)), gap_before_onset=len(before),
                           unevaluated_after_onset=len(after), status=status))
    return output


def new_runs_report(args):
    rows = aggregate_new_runs(args.input, parse_seed_spec(args.seeds))
    write_records(args.out, rows)
    print(f"{len(rows)} fresh-run summaries written to {args.out}")
    print("Missing checkpoints are relative to epochs present for that model in the supplied files; "
          "unprovided training checkpoints cannot be inferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    help_text = {
        "onsets": "recompute all 784 historical onset cells and Tables 1–3",
        "validate": "validate historical results and verify manifest hashes",
        "type-i": "Table 5, Appendix C and untrained-network controls",
        "synthetic": "Table 6 and Appendix D synthetic comparisons",
        "small-sample": "Table 7 and the 1,500-point onset records",
        "reversed": "Table 8 and reference-direction comparisons",
        "score-transform": "Appendix I: VRT, MMD on scores and MMD on embeddings",
        "runtime": "Table 9: the final float64 VRT timings",
    }
    for command, description in help_text.items():
        p = sub.add_parser(command, help=description)
        p.add_argument("--data", default="data", help="released result directory")
        if command == "onsets":
            p.add_argument("--out", default="results/onsets.csv")
            p.add_argument("--stored-p", "--stored_p", action="store_true", dest="stored_p")
            p.add_argument("--no-validate", "--no_validate", action="store_true", dest="no_validate")
            p.add_argument("--no-protocol-check", "--no_protocol_check", action="store_true", dest="no_protocol_check")
        elif command == "validate":
            p.add_argument("--no-manifest", action="store_true", help="validate new files without a release manifest")
        elif command == "synthetic":
            p.add_argument("--latex", action="store_true", help="also print LaTeX table rows")
        else:
            p.add_argument("--out", help="also write the computed records to CSV")
    p = sub.add_parser("export", help="validate and clean raw historical results into a new directory")
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--dry-run", "--dry_run", action="store_true", dest="dry_run")
    p.add_argument("--allow-unknown", "--allow_unknown", action="store_true", dest="allow_unknown")
    p = sub.add_parser("new-runs", help="aggregate the separate CSV format produced by evaluate.py")
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--out", default="results/new_onsets.csv")
    p.add_argument("--seeds", default="42-91")
    args = parser.parse_args()
    commands = {"onsets": onset_report, "validate": validate_report, "export": export_results,
                "type-i": type_i_report, "synthetic": synthetic_report, "small-sample": supplementary_onsets,
                "reversed": supplementary_onsets, "score-transform": score_transform_report,
                "runtime": runtime_report, "new-runs": new_runs_report}
    try:
        commands[args.command](args)
    except SchemaError as e:
        parser.exit(1, f"results: {e}\n")


if __name__ == "__main__":
    main()
