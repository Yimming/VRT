"""
export_results.py — build the clean, verified result set for the code release from the raw per-seed CSVs.

Input: the raw result tree of the 50-seed runs (the s50 folder; its ssl_baselines subfolder holds the SimCLR eval-mode
MMDAgg files). Output: a new, empty destination folder with the same layout, every CSV validated and cleaned, plus
MANIFEST.csv (path, rows, bytes, sha256, what was cleaned) and EXCLUDED.txt (folders left out and why).

Validation (tools/result_schemas.py), per file: the file name must match the pattern of its family; every row must
belong to the file (model, budget or sample size as in the file name), parse, and carry admissible values (permutation
p in (0, 1], MMDAgg decision 0 or 1, C2ST accuracy in [0, 1], finite statistics); across the whole export no key may
occur with two different values (c2st and bntrain_c2st share one key space, mmd and mmd_raw do not). Any violation
stops the export with the file and line. Cleaning, logged per file in MANIFEST.csv: NUL bytes removed (eleven as-run
files hold zero-filled regions between rows, left by concurrent appends; the rows themselves are complete), torn lines
dropped (none in the as-run data), rows repeated verbatim kept once (one file), repeated header lines dropped. The two
families written with four-decimal permutation p-values (reversed VRT, Henze) get an extra column p_exact /
p_henze_exact = round(p x 2001)/2001 when they do not carry the integer count; files of the new format (count, n_perm
columns) are checked for p == (count + 1)/(n_perm + 1) instead.
Folders excluded: c100_nan_bug (SimCLR CIFAR-100 MMD on c-hat before the +inf cap), dup_backup (c-hat diagnostic files
before the duplicate cleanup), _superseded (10-seed kNN runs), null10 (10-seed null runs). Legacy files of the 10-seed
SimCLR stage in ssl_baselines (sslbase_*, sslvrt_*, sslseeds_*) are copied verbatim and marked; the harvest never reads
them. Files whose names no schema recognises stop the export unless --allow_unknown copies them verbatim (marked).
Usage:
  python tools/export_results.py --src ~/Downloads/s50/ssl_data --dst data
  python tools/export_results.py --src ~/Downloads/s50/ssl_data --dst data --dry_run
"""
import argparse
import csv
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
from release_utils import exact_perm_p  # noqa: E402
from result_schemas import NESTED, SCHEMAS, SchemaError, read_rows, schema_for, validate_rows  # noqa: E402

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
                    self.copy(s, d, rel, "legacy file of the 10-seed stage, copied verbatim, not read by the harvest"); continue
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True, help="a new or empty folder")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--allow_unknown", action="store_true", help="copy files with unrecognised names instead of stopping")
    a = ap.parse_args()
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


if __name__ == "__main__":
    main()
