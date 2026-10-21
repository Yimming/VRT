"""VRT scoring and permutation tests on normalized embeddings."""

import numpy as np
from scipy.special import betainc, betaln

from common import check_scores, check_perm_statistic, l2n


def log_cap_volume(theta, D):
    """Log volume of a spherical cap, including the small-volume limit."""
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
    """Score each query using its k nearest reference points; preserve input dtype."""
    if k > ref.shape[0]:
        raise ValueError(f"k={k} exceeds reference size M={ref.shape[0]}")
    Nq = query.shape[0]
    theta_all = []
    for s in range(0, Nq, batch_size):
        e = min(s + batch_size, Nq)
        cos = query[s:e] @ ref.T
        pidx = np.argpartition(-cos, k - 1, axis=1)[:, :k]
        top = np.take_along_axis(cos, pidx, axis=1)
        top = np.take_along_axis(top, np.argsort(-top, axis=1), axis=1)
        theta_all.append(np.arccos(np.clip(top, -1, 1)))
    theta = np.concatenate(theta_all, 0)
    L = log_cap_volume(theta, D)
    dL = np.diff(L, axis=1)
    r = np.arange(1, k, dtype=np.float64)
    Z = np.maximum(r[None, :] * dL, 0.0)
    return np.maximum(Z.mean(axis=1), 1e-12)


def perm_p(c_tr, c_te, n_perm, rng):
    """Two-sided median test, continuing the supplied sampling random stream."""
    check_scores(c_tr, c_te)
    obs = check_perm_statistic(np.median(c_te) - np.median(c_tr), "observed median difference")
    pooled = np.concatenate([c_tr, c_te])
    na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        d = check_perm_statistic(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]]), "permuted median difference")
        if abs(d) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def vrt_full(emb_tr, emb_te, D, k=60, M=2000, n_qtr=8000, n_perm=2000, seed=42):
    """Draw disjoint train reference/queries and use all supplied test points."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_tr))
    ref = emb_tr[perm[:M]]
    q_tr = emb_tr[perm[M:M + min(n_qtr, len(emb_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(emb_te, ref, D, k)
    return perm_p(c_tr, c_te, n_perm, rng)


def vrt_low(e_tr, e_te, D, k=15, M=500, n_qtr=1000, n_qte=1500,
            n_perm=2000, seed=42):
    """Small-budget protocol, including its separate test-query draw."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(e_tr))
    ref = e_tr[perm[:M]]
    q_tr = e_tr[perm[M:M+min(n_qtr, len(e_tr)-M)]]
    pte = rng.permutation(len(e_te))
    q_te = e_te[pte[:min(n_qte, len(e_te))]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(q_te, ref, D, k)
    return perm_p(c_tr, c_te, n_perm, rng)


def vrt_reversed(emb_tr, emb_te, D, M, k, n_q, n_perm, seed):
    """Draw reference points from test, then sample train queries."""
    rng = np.random.RandomState(seed)
    pte = rng.permutation(len(emb_te))
    ref = emb_te[pte[:M]]
    q_te = emb_te[pte[M:M + n_q]]
    ptr = rng.permutation(len(emb_tr))
    q_tr = emb_tr[ptr[:n_q]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(q_te, ref, D, k)
    return perm_p(c_tr, c_te, n_perm, rng)


vrt_test = vrt_full
check_median_stat = check_perm_statistic
