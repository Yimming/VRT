"""
Tests of results.py export and schema validation on synthetic result folders: NUL-filled regions, torn
lines, repeated header lines, verbatim duplicates, conflicting keys inside a file and across files (export refused),
rows that do not belong to their file, invalid values (NaN p, MMDAgg decision 2), unrecognised names, the exact-p
column for the two four-decimal families, excluded folders, legacy files, a non-empty destination and the manifest.
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOL = os.path.join(ROOT, "results.py")
SEEDS = range(42, 92)


def run(src, dst, *extra):
    return subprocess.run([sys.executable, TOOL, "export", "--src", str(src), "--dst", str(dst), *extra],
                          capture_output=True, text=True)


def write(path, text, binary=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(text.encode() if binary is None else binary)


def make_tree(src):
    # vrt_fwd: clean file plus a NUL region, a torn line and a repeated header, as left by concurrent appends
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c0-9.csv",
          "tag,budget,epoch,seed,dc,p\nr50,large,0,42,+0.100000,0.500250\nr50,large,0,43,+0.200000,0.400300\n")
    body = ("tag,budget,epoch,seed,dc,p\n" "r50,large,1,42,+0.100000,0.300350\n").encode()
    body += b"\x00" * 64 + b"r50,large,1,43,+0.2" + b"\n" + b"r50,large,1,43,+0.200000,0.200400\n"
    body += b"tag,budget,epoch,seed,dc,p\n" + b"r50,large,1,42,+0.100000,0.300350\n"      # second job: header + same row
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c10-19.csv", "", body)
    # reversed and knn_v2: four-decimal p on the 1/2001 lattice
    write(src / "reversed" / "vrt_rev_zoo_c10_m3_large_c0-9.csv",
          "tag,budget,direction,epoch,seed,M,k,delta_c,p\n"
          "zoo_c10_m3,large,test_ref,30,42,2000,60,-0.01,0.0470\nzoo_c10_m3,large,test_ref,30,43,2000,60,-0.02,0.0530\n")
    write(src / "knn_v2" / "knn_r50_n10000_part_c0-39.csv",
          "tag,n_samples,k,epoch,seed,p_schilling,p_henze\nr50,10000,3,83,42,0.0412,0.0500\n")
    # MMDAgg decisions, a legacy SimCLR file, excluded folders and a non-csv file
    write(src / "mmdagg" / "mmdagg_r50_b5000_c129-129.csv",
          "model,epoch,budget,stat,seed,acc,p\nr50,129,b5000,mmdagg,42,nan,1\nr50,129,b5000,mmdagg,43,nan,0\n")
    write(src / "ssl_baselines" / "sslbase_cifar10_0-99.csv", "model,epoch,budget,stat,seed,acc,p\ncifar10,5,b8000,c2st,42,0.5,0.4\n")
    write(src / "knn_v2" / "_superseded" / "knn_r50_n10000_part0.csv", "tag\nold\n")
    write(src / "mmd_on_chat" / "c100_nan_bug" / "x.csv", "tag\nold\n")
    write(src / "chat_stats" / "dup_backup" / "x.csv", "tag\nold\n")
    write(src / "null10" / "x.csv", "tag\nold\n")
    write(src / "ssl" / "notes.txt", "hello\n")


def test_export_cleans_validates_and_documents(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src)
    r = run(src, dst)
    assert r.returncode == 0, r.stderr + r.stdout
    rows = list(csv.DictReader(open(dst / "vrt_fwd" / "vrt_fwd_r50_large_c10-19.csv")))
    assert [(x["epoch"], x["seed"], x["p"]) for x in rows] == [("1", "42", "0.300350"), ("1", "43", "0.200400")]
    rev = list(csv.DictReader(open(dst / "reversed" / "vrt_rev_zoo_c10_m3_large_c0-9.csv")))
    assert [x["p_exact"] for x in rev] == [f"{94 / 2001:.9f}", f"{106 / 2001:.9f}"]
    knn = list(csv.DictReader(open(dst / "knn_v2" / "knn_r50_n10000_part_c0-39.csv")))
    assert knn[0]["p_henze_exact"] == f"{100 / 2001:.9f}" and knn[0]["p_schilling"] == "0.0412"
    assert not (dst / "knn_v2" / "_superseded").exists() and not (dst / "null10").exists()
    assert not (dst / "mmd_on_chat" / "c100_nan_bug").exists() and not (dst / "chat_stats" / "dup_backup").exists()
    assert (dst / "ssl" / "notes.txt").read_text() == "hello\n"
    man = {x["path"]: x for x in csv.DictReader(open(dst / "MANIFEST.csv"))}
    m = man[os.path.join("vrt_fwd", "vrt_fwd_r50_large_c10-19.csv")]
    assert m["nul_bytes_removed"] == "64" and m["torn_lines_dropped"] == "1" and m["duplicate_rows_dropped"] == "2"
    assert m["rows"] == "2" and len(m["sha256"]) == 64
    assert man[os.path.join("ssl_baselines", "sslbase_cifar10_0-99.csv")]["note"].startswith("legacy")
    excluded = (dst / "EXCLUDED.txt").read_text()
    assert "null10" in excluded and "_superseded" in excluded and "c100_nan_bug" in excluded and "dup_backup" in excluded
    assert "cleaned vrt_fwd" in r.stdout
    # a second export into the same (non-empty) folder is refused
    r2 = run(src, dst)
    assert r2.returncode != 0 and "not empty" in r2.stderr


def test_export_refuses_conflicts_within_and_across_files(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c0-9.csv",
          "tag,budget,epoch,seed,dc,p\nr50,large,0,42,+0.1,0.500250\nr50,large,0,42,+0.1,0.600200\n")
    r = run(src, dst)
    assert r.returncode != 0 and "different values" in r.stderr
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c0-9.csv", "tag,budget,epoch,seed,dc,p\nr50,large,0,42,+0.1,0.500250\n")
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c10-19.csv", "tag,budget,epoch,seed,dc,p\nr50,large,0,42,+0.1,0.600200\n")
    r = run(src, dst)
    assert r.returncode != 0 and "different values" in r.stderr and "vrt_fwd_r50_large_c0-9.csv" in r.stderr
    # c2st and bntrain_c2st share one key space
    write(src / "vrt_fwd" / "vrt_fwd_r50_large_c10-19.csv", "tag,budget,epoch,seed,dc,p\nr50,large,1,42,+0.1,0.600200\n")
    write(src / "c2st" / "c2st_r50_bntrain_b1500_c0-9.csv", "model,epoch,budget,stat,seed,acc,p\nr50_bntrain,0,b1500,c2st,42,0.6,0.01\n")
    write(src / "bntrain_c2st" / "c2st_r50_bntrain_b1500_c0-9.csv", "model,epoch,budget,stat,seed,acc,p\nr50_bntrain,0,b1500,c2st,42,0.6,0.02\n")
    r = run(src, dst)
    assert r.returncode != 0 and "different values" in r.stderr


def test_export_refuses_foreign_rows_invalid_values_and_unknown_names(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "vrt_fwd" / "vrt_fwd_v7_large_c0-9.csv", "tag,budget,epoch,seed,dc,p\nr50,large,0,42,+0.1,0.5\n")
    r = run(src, dst)
    assert r.returncode != 0 and "does not belong to this file" in r.stderr
    write(src / "vrt_fwd" / "vrt_fwd_v7_large_c0-9.csv", "tag,budget,epoch,seed,dc,p\nv7,large,0,42,+0.1,nan\n")
    r = run(src, dst)
    assert r.returncode != 0 and "is not p in (0, 1]" in r.stderr
    write(src / "vrt_fwd" / "vrt_fwd_v7_large_c0-9.csv", "tag,budget,epoch,seed,dc,p\nv7,large,0,42,+0.1,0.5\n")
    write(src / "mmdagg" / "mmdagg_v7_b5000_c0-9.csv", "model,epoch,budget,stat,seed,acc,p\nv7,0,b5000,mmdagg,42,nan,2\n")
    r = run(src, dst)
    assert r.returncode != 0 and "is not 0 or 1" in r.stderr
    write(src / "mmdagg" / "mmdagg_v7_b5000_c0-9.csv", "model,epoch,budget,stat,seed,acc,p\nv7,0,b5000,mmdagg,42,nan,1\n")
    write(src / "mmdagg" / "a.csv", "model,epoch,budget,stat,seed,acc,p\nv7,0,b5000,mmdagg,42,nan,1\n")
    r = run(src, dst)
    assert r.returncode != 0 and "no schema recognises" in r.stderr
    r = run(src, dst, "--allow_unknown")
    assert r.returncode == 0, r.stderr
    man = {x["path"]: x for x in csv.DictReader(open(dst / "MANIFEST.csv"))}
    assert man[os.path.join("mmdagg", "a.csv")]["note"].startswith("unrecognised")


def test_type_one_families_are_validated(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "null" / "typeI_verbatim_paper_full.csv", "seed,dc,p\n0,2.039851,0.210895\n")
    write(src / "null" / "vrt_fwd" / "vrt_fwd_imgnet_rand_eval_large.csv",
          "tag,budget,epoch,seed,dc,p\nimgnet_rand_eval,large,0,42,+0.1,0.5\n")
    write(src / "null_trtr" / "knn_v2" / "knn_imgnet_rand_eval_trtr_n10000_part.csv",
          "tag,n_samples,k,epoch,seed,p_schilling,p_henze\nimgnet_rand_eval_trtr,10000,3,0,42,0.0412,0.0500\n")
    write(src / "null_cifar_base" / "base_null_r50_c100_eval_cpu_r0-49.csv",
          "arch,bn_mode,rep,net_seed,split_seed,seed,split,method,budget,p,acc,sec\n"
          "r50_c100,eval,0,1000,1000,100000,resplit,mmdagg,5000,0,nan,3\nr50_c100,eval,0,1000,1000,100000,resplit,c2st,10000,0.18,0.5045,0\n")
    write(src / "null_cifar_vrt" / "vrt_null_frn_c10_eval_r0-99.csv",
          "arch,bn_mode,rep,net_seed,split_seed,vrt_seed,split,budget,D,dc,p,cos,feat_sec\n"
          "frn_c10,eval,0,1000,1000,100000,resplit,large,64,-3e-4,0.978511,0.9966,1.8\n")
    write(src / "official_check" / "official_check_pixels_cifar10_r0-49.csv", "dataset,features,rep,split,method,p\ncifar10,pixels,0,official,mmd,0.435282\n")
    r = run(src, dst)
    assert r.returncode == 0, r.stderr
    man = {x["path"]: x for x in csv.DictReader(open(dst / "MANIFEST.csv"))}
    assert man[os.path.join("null_trtr", "knn_v2", "knn_imgnet_rand_eval_trtr_n10000_part.csv")]["note"] == "added p_henze_exact"
    assert all(man[k]["rows"] for k in man if k.endswith(".csv"))
    # an invalid MMDAgg decision in the CIFAR null family is refused like everywhere else
    write(src / "null_cifar_base" / "base_null_r50_c100_eval_cpu_r0-49.csv",
          "arch,bn_mode,rep,net_seed,split_seed,seed,split,method,budget,p,acc,sec\nr50_c100,eval,0,1000,1000,100000,resplit,mmdagg,5000,0.5,nan,3\n")
    r = run(src, tmp_path / "dst2")
    assert r.returncode != 0 and "is not 0 or 1" in r.stderr


def test_features_column_of_the_released_runners_is_accepted(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    write(src / "mmd_p2000" / "mmd_r50_n10000_c0-9.csv",
          "tag,n_samples,n_perm,epoch,seed,mmd2,p,features\nr50,10000,2000,0,42,1e-3,0.500250,l2\n")
    write(src / "mmd_raw_p2000" / "mmd_r50_n10000_c0-9.csv",
          "tag,n_samples,n_perm,epoch,seed,mmd2,p,features\nr50,10000,2000,0,42,1e-3,0.500250,raw\n")
    write(src / "knn_v2" / "knn_r50_n10000_part_c0-9.csv",
          "tag,n_samples,k,epoch,seed,p_schilling,p_henze,henze_count,n_perm,features\nr50,10000,3,0,42,0.0412,0.049975,99,2000,raw\n")
    write(src / "c2st" / "c2st_r50_b1500_c0-9.csv", "model,epoch,budget,stat,seed,acc,p,features\nr50,0,b1500,c2st,42,0.6,0.01,l2\n")
    r = run(src, dst)
    assert r.returncode == 0, r.stderr
    knn = list(csv.DictReader(open(dst / "knn_v2" / "knn_r50_n10000_part_c0-9.csv")))
    assert knn[0]["p_henze_exact"] == f"{100 / 2001:.9f}" and knn[0]["features"] == "raw"
    write(src / "c2st" / "c2st_r50_b1500_c0-9.csv", "model,epoch,budget,stat,seed,acc,p,features\nr50,0,b1500,c2st,42,0.6,0.01,both\n")
    r = run(src, tmp_path / "dst2")
    assert r.returncode != 0 and "is not l2 or raw" in r.stderr


def test_simclr_file_in_the_wrong_folder_is_refused(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    row = "model,epoch,budget,stat,seed,acc,p\n{m},5,b10000,vrt,42,nan,0.5\n"
    write(src / "ssl" / "ssl_cifar10_fast_c1-100.csv", row.format(m="cifar10"))
    write(src / "ssl_bntrain" / "ssl_cifar10_bntrain_fast_c1-100.csv", row.format(m="cifar10_bntrain"))
    assert run(src, dst).returncode == 0
    write(src / "ssl" / "ssl_cifar100_bntrain_fast_c1-100.csv", row.format(m="cifar100_bntrain"))   # train file under ssl/
    r = run(src, tmp_path / "dst2")
    assert r.returncode != 0 and "belongs in ssl_bntrain/" in r.stderr
    os.remove(src / "ssl" / "ssl_cifar100_bntrain_fast_c1-100.csv")
    write(src / "ssl_bntrain" / "sslagg_cifar100_s42-91_e100.csv",                                  # eval sslagg under ssl_bntrain/
          "model,epoch,budget,stat,seed,acc,p\ncifar100,100,b5000,mmdagg,42,nan,1\n")
    r = run(src, tmp_path / "dst3")
    assert r.returncode != 0 and "belongs in ssl_baselines/" in r.stderr


def test_dry_run_writes_nothing(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make_tree(src)
    r = run(src, dst, "--dry_run")
    assert r.returncode == 0 and not dst.exists() and "would be exported" in r.stdout
