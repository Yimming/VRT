"""vrt_low_lib.py - VERBATIM copy of the scoring and test functions of ksweep15_fill.py (the rebuttal's
small-budget VRT script), importable without running its argparse block.
ksweep15_fill.py - small-budget VRT (k=15) on the full epoch grid.
Protocol: M=500 train reference + 1,000 train queries + 1,500 test queries,
k=15, 2,000 permutations, seeds 42-51, two-sided median test. Scoring
copied verbatim from the supplementary phase_multiseed_vrt.py.
Output: ksweep15_{unit}_{lo}-{hi}.csv, columns model,epoch,k,stat,seed,T,p."""
import numpy as np
from scipy.special import betainc, betaln

def log_cap_volume(theta, D):
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        log_V[~safe] = const + (D-1) * np.log(np.maximum(np.sin(theta[~safe]), 1e-300))
    return log_V

def compute_chat(query, ref, D, k, batch_size=1024):
    """c-hat of every query against the reference set (k nearest reference points by cosine).
    Arithmetic unchanged from the runs. Note on dtype (release audit): with float32 inputs a query that coincides with a
    reference point has angle 0 and sin 0 = 0; the 1e-300 floor below is then cast to float32 and becomes 0, so the log
    is -inf and the query's c-hat is +inf. With float64 inputs the floor holds and c-hat is large but finite. The VRT
    median tests accept isolated +inf values (they only rank last); kernel tests on c-hat must not (see
    ssl_mmd_on_chat.cap_infinite). A query that coincides with two or more reference points (duplicate reference
    points) gives two -inf cap volumes in a row, dL = -inf - (-inf) = NaN and a NaN c-hat even for finite inputs;
    check_scores() rejects NaN and -inf, and the median statistics are checked in vrt_low / vrt_full."""
    Nq = query.shape[0]; theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s+batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D); dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None,:] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)

def check_scores(*cs):
    """release fix: refuse NaN or -inf c-hat values (non-finite input, or a query coinciding with duplicate reference
    points); +inf is allowed, see compute_chat"""
    for c in cs:
        if np.isnan(c).any() or np.isneginf(c).any():
            raise FloatingPointError(f"c-hat contains {int(np.isnan(c).sum())} NaN and {int(np.isneginf(c).sum())} -inf values; "
                                     "check the input features (duplicate or non-finite points)")


def check_median_stat(x, what):
    """release fix: a median difference that is NaN (inf - inf) or infinite (more than half of one side's scores +inf)
    would make every permutation comparison False and return p = 1/(n_perm + 1); stop instead. The check does not touch
    the random stream (the permutations are drawn exactly as in the runs)."""
    if not np.isfinite(x):
        raise FloatingPointError(f"{what} is {x!r}; the median test is undefined for these scores")
    return x


def vrt_low(e_tr, e_te, D, k=15, M=500, n_qtr=1000, n_qte=1500,
            n_perm=2000, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(e_tr))
    ref = e_tr[perm[:M]]
    q_tr = e_tr[perm[M:M+min(n_qtr, len(e_tr)-M)]]
    pte = rng.permutation(len(e_te))
    q_te = e_te[pte[:min(n_qte, len(e_te))]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(q_te, ref, D, k)
    check_scores(c_tr, c_te)
    obs = check_median_stat(np.median(c_te) - np.median(c_tr), "observed median difference")
    pooled = np.concatenate([c_tr, c_te]); na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        d = check_median_stat(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]]), "permuted median difference")
        if abs(d) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)

def vrt_full(emb_tr, emb_te, D, k=60, M=2000, n_qtr=8000, n_perm=2000, seed=42):
    """VERBATIM body of supcon_fill_vrt.vrt_test (the paper's large-budget protocol): reference and train
    queries drawn from one permutation of the train split, ALL test points as test queries, no test-side draw."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_tr))
    ref = emb_tr[perm[:M]]
    q_tr = emb_tr[perm[M:M + min(n_qtr, len(emb_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(emb_te, ref, D, k)
    check_scores(c_tr, c_te)
    obs = check_median_stat(np.median(c_te) - np.median(c_tr), "observed median difference")
    pooled = np.concatenate([c_tr, c_te])
    na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        d = check_median_stat(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]]), "permuted median difference")
        if abs(d) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)

def l2n(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

