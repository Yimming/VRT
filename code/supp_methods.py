"""
methods.py — Two-sample test implementations
==============================================
VRT:    Proposed method. Permutation-based, p = (count+1)/(n_perm+1)
MMD:    Gretton et al., JMLR 2012. Gaussian RBF + median heuristic
MMDAgg: Schrab et al., JMLR 2023. Aggregated kernel, binary output
ED:     Székely & Rizzo, 2004. Euclidean energy distance
C2ST:   Lopez-Paz & Oquab, ICLR 2017. Held-out classifier accuracy
"""

import numpy as np
from scipy.special import betainc, betaln
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier
try:
    from mmdagg.np import mmdagg as mmdagg_fn
except ImportError:
    mmdagg_fn = None


# ═══════════════════════════════════════════════════════════════
# VRT
# ═══════════════════════════════════════════════════════════════

def log_cap_volume(theta, D):
    """Log spherical cap volume, numerically stable for any D."""
    sin2 = np.sin(theta)**2
    I = betainc((D-1)/2.0, 0.5, sin2)
    V = np.where(theta <= np.pi/2, 0.5*I, 1.0 - 0.5*I)
    safe = V > 1e-300
    log_V = np.empty_like(V)
    log_V[safe] = np.log(V[safe])
    if np.any(~safe):
        a = (D-1) / 2.0
        const = -np.log(2.0) - np.log(a) - betaln(a, 0.5)
        sin_vals = np.maximum(np.sin(theta[~safe]), 1e-300)
        log_V[~safe] = const + (D-1) * np.log(sin_vals)
    return log_V


def compute_chat(query, ref, D, k, batch_size=1024):
    """Compute per-point VRT scores for query points against reference."""
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


def vrt_test(emb_tr, emb_te, D, k=60, M=2000, n_qtr=8000,
             n_perm=2000, seed=42):
    """VRT two-sample test with training embeddings as reference.

    Parameters
    ----------
    emb_tr : array, shape (N_train, D)
        L2-normalized training embeddings on S^{D-1}.
    emb_te : array, shape (N_test, D)
        L2-normalized test embeddings on S^{D-1}.
    D : int
        Embedding dimension.
    k : int
        Number of nearest neighbors for scoring.
    M : int
        Reference set size (drawn from emb_tr).
    n_qtr : int
        Training query set size (disjoint from reference).
    n_perm : int
        Number of permutations for p-value.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dc : float
        Observed median score difference, median(c_te) - median(c_tr).
    p_value : float
        Two-sided permutation p-value via Phipson & Smyth (2010).
    """
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_tr))
    ref = emb_tr[perm[:M]]
    q_tr = emb_tr[perm[M:M + min(n_qtr, len(emb_tr) - M)]]
    c_tr = compute_chat(q_tr, ref, D, k)
    c_te = compute_chat(emb_te, ref, D, k)
    obs = np.median(c_te) - np.median(c_tr)
    pooled = np.concatenate([c_tr, c_te])
    na = len(c_tr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ═══════════════════════════════════════════════════════════════
# MMD (Gretton et al., JMLR 2012)
# ═══════════════════════════════════════════════════════════════

def mmd_test(emb_tr, emb_te, n_sub=1500, n_perm=1000, seed=42):
    """MMD two-sample test with Gaussian RBF kernel and median heuristic."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr))
    n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    pooled = np.concatenate([X, Y])
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)]
    d2 = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2 * sub @ sub.T
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, np.inf)
    vals = d2[np.isfinite(d2) & (d2 > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) > 0 else 1.0

    def rbf(A, B, s):
        d2 = np.sum(A**2, 1, keepdims=True) + np.sum(B**2, 1, keepdims=True).T - 2 * A @ B.T
        return np.exp(-np.maximum(d2, 0.0) / (2 * s**2))

    def mmd2(X, Y, s):
        n, m = len(X), len(Y)
        Kxx = rbf(X, X, s); np.fill_diagonal(Kxx, 0)
        Kyy = rbf(Y, Y, s); np.fill_diagonal(Kyy, 0)
        Kxy = rbf(X, Y, s)
        return float(Kxx.sum() / (n * (n-1)) + Kyy.sum() / (m * (m-1)) - 2 * Kxy.sum() / (n * m))

    obs = mmd2(X, Y, sigma)
    pooled_all = np.concatenate([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled_all))
        if mmd2(pooled_all[p[:n]], pooled_all[p[n:]], sigma) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ═══════════════════════════════════════════════════════════════
# MMDAgg (Schrab et al., JMLR 2023)
# ═══════════════════════════════════════════════════════════════

def mmdagg_test(emb_tr, emb_te, n_sub=1500, seed=42):
    """MMDAgg test. Returns binary: 1 = reject, 0 = not reject."""
    if mmdagg_fn is None:
        raise ImportError("mmdagg is required. Install with: pip install mmdagg")
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]].astype(np.float32)
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]].astype(np.float32)
    return int(mmdagg_fn(X, Y, seed=seed))


# ═══════════════════════════════════════════════════════════════
# Energy Distance (Székely & Rizzo, 2004)
# ═══════════════════════════════════════════════════════════════

def energy_distance_test(emb_tr, emb_te, n_sub=1500, n_perm=500, seed=42):
    """Energy distance two-sample test."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]

    def ed_stat(X, Y):
        n, m = len(X), len(Y)
        xy = sum(np.sqrt(((X[s:min(s+400,n), None, :] - Y[None, :, :])**2).sum(2)).sum()
                 for s in range(0, n, 400)) / (n * m)
        xx = sum(np.sqrt(((X[s:min(s+400,n), None, :] - X[None, :, :])**2).sum(2)).sum()
                 for s in range(0, n, 400)) / (n * n)
        yy = sum(np.sqrt(((Y[s:min(s+400,m), None, :] - Y[None, :, :])**2).sum(2)).sum()
                 for s in range(0, m, 400)) / (m * m)
        return float(2 * xy - xx - yy)

    obs = ed_stat(X, Y)
    pooled = np.concatenate([X, Y])
    n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if ed_stat(pooled[p[:n]], pooled[p[n:]]) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


# ═══════════════════════════════════════════════════════════════
# C2ST (Lopez-Paz & Oquab, ICLR 2017)
# ═══════════════════════════════════════════════════════════════

def c2st_test(emb_tr, emb_te, n_sub=8000, seed=42):
    """C2ST-NN: classifier two-sample test (1 hidden layer, 20 ReLU units, Adam). Returns (accuracy, baseline, p-value)."""
    rng = np.random.RandomState(seed)
    n_each = min(n_sub, len(emb_tr), len(emb_te))
    X_p = emb_tr[rng.permutation(len(emb_tr))[:n_each]]
    X_q = emb_te[rng.permutation(len(emb_te))[:n_each]]
    X = np.concatenate([X_p, X_q])
    y = np.array([0] * n_each + [1] * n_each)
    shuf = rng.permutation(len(X))
    X, y = X[shuf], y[shuf]
    n_tr = len(X) // 2
    X_train, y_train = X[:n_tr], y[:n_tr]
    X_test, y_test = X[n_tr:], y[n_tr:]
    n_te = len(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation='relu', solver='adam', max_iter=100, random_state=seed)
    clf.fit(X_train, y_train)
    acc = float((clf.predict(X_test) == y_test).mean())
    se = np.sqrt(1.0 / (4.0 * n_te))
    z = (acc - 0.5) / se
    p_val = float(1.0 - norm.cdf(z))
    return acc, 0.5, p_val
