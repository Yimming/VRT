"""
result_schemas.py — what a valid result file of each family looks like, shared by export_results.py and make_onsets.py.

A schema fixes, per result family (the folder name), the file-name pattern, the columns that identify a record (the
key), the columns of the file name that every row must agree with (a row of another model or budget inside a file is
an error), the value columns with their admissible values (permutation p in (0, 1], a 0/1 MMDAgg decision, a C2ST
accuracy in [0, 1], finite statistics), and the `group` under which keys must be unique across all files of the export
(c2st and bntrain_c2st share one key space, mmd and mmd_raw do not). validate_rows() applies a schema to the parsed
rows of one file and returns the records (key, values) or raises SchemaError with the first offending lines.
Legacy files (the 10-seed stage of the SimCLR runs, never read by the harvest) are recognised and copied without
validation.
"""
import csv
import io
import math
import os
import re

FINITE, PVAL, BINARY, UNIT, NAN_OR_UNIT, INT, PVAL_OR_EMPTY, ANY, APPROX_P, FEATURES = range(10)
# key columns that are integers: parsed as int before anything is compared, so '042' and '42' are one seed and 'oops'
# is an error rather than a silently skipped row
INT_KEYS = {"epoch", "seed", "n_samples", "n_perm", "M", "k", "rep", "C", "scale_int"}
# fixed protocol of the 50-seed runs, checked by make_onsets (--no_protocol_check to skip): a row with another value
# of these columns is a different experiment and must not be merged into a cell
PROTOCOL = {"reversed": {"direction": {"test_ref"}, "n_perm": {2000}},
            "reversed/large": {"M": {2000}, "k": {60}}, "reversed/small": {"M": {500}, "k": {15}},
            "mmd": {"n_perm": {2000}}, "mmd_raw": {"n_perm": {2000}}, "ed": {"n_perm": {2000}}, "ed_raw": {"n_perm": {2000}},
            "knn": {"k": {3}, "n_perm": {2000}}}
# PVAL is a permutation p-value, (count + 1)/(B + 1) > 0; APPROX_P a normal-approximation p-value (C2ST, Schilling), which
# can round to 0.000000


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
CHUNK = r"(_c\d+(-\d+)?)?"                              # epoch chunk of one array task, _c<lo>-<hi> or _c<epoch>; absent in the Type-I null runs
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


# Type-I null runs: null/ and null_trtr/ mirror the main families in subfolders (same runners, tags imgnet_rand_*),
# plus the verbatim ImageNet null files at their top level
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
        if family not in SCHEMAS:
            continue
        for name in sorted(os.listdir(fdir)):
            if not name.endswith(".csv"):
                continue
            path = os.path.join(fdir, name)
            schema, m = schema_for(family, name)
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


if __name__ == "__main__":
    import sys
    s = validate_tree(sys.argv[1], verbose="-v" in sys.argv)
    print(s)
