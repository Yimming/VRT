"""Two-sample baselines used in the paper and its controls."""

import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier

from common import check_statistics, l2n

try:
    from mmdagg.np import mmdagg as mmdagg_fn
except ImportError:
    mmdagg_fn = None



def median_sigma(pooled, rng):
    """Median RBF bandwidth using a random pooled subsample."""
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)].astype(np.float64)
    sq = np.sum(sub ** 2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * sub @ sub.T, 0.0)
    np.fill_diagonal(d2, np.inf)
    vals = d2[np.isfinite(d2) & (d2 > 0)]
    return max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) > 0 else 1.0


def rbf_matrix(Z, sigma, backend="torch"):
    """Float64 pooled RBF matrix, with a zero diagonal."""
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64)).to(dev)
        sq = (T * T).sum(1)
        K = T @ T.T
        K.mul_(-2.0).add_(sq[:, None]).add_(sq[None, :])
        K.clamp_(min=0.0).mul_(-1.0 / (2 * sigma ** 2)).exp_()
        K.fill_diagonal_(0.0)
        return K
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T), 0.0)
    K = np.exp(-d2 / (2 * sigma ** 2))
    np.fill_diagonal(K, 0.0)
    return K


def mmd2_from_matrix(K, P, n, m):
    """Unbiased MMD squared for each labelling column in P."""
    is_torch = not isinstance(K, np.ndarray)
    if is_torch:
        import torch
        P = torch.as_tensor(P, dtype=K.dtype, device=K.device)
    else:
        P = np.asarray(P, dtype=np.float64)
    tot = K.sum()
    KP = K @ P
    S1all = KP.sum(0)
    Sxx = (P * KP).sum(0)
    Syy = tot - 2.0 * S1all + Sxx
    Sxy = S1all - Sxx
    val = Sxx / (n * (n - 1)) + Syy / (m * (m - 1)) - 2.0 * Sxy / (n * m)
    return val.cpu().numpy() if is_torch else val


def mmd_test_fast(emb_tr, emb_te, n_sub=1500, n_perm=2000, seed=42, backend="torch"):
    """MMD permutation test; return (statistic, p-value)."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr)); n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    pooled = np.concatenate([X, Y])
    if not np.isfinite(pooled).all():
        raise ValueError("mmd_test_fast: the subsample contains NaN or inf")
    sigma = median_sigma(pooled, rng)
    N, n, m = len(pooled), n_x, n_y
    K = rbf_matrix(pooled, sigma, backend)
    perms = np.stack([rng.permutation(N) for _ in range(n_perm)])
    P = np.zeros((N, n_perm + 1), dtype=np.float64)
    P[:n, 0] = 1.0
    P[perms[:, :n].T, np.arange(1, n_perm + 1)[None, :]] = 1.0
    val = mmd2_from_matrix(K, P, n, m)
    obs, perm_vals = float(val[0]), val[1:]
    check_statistics(obs, perm_vals, "MMD^2")
    p = float((np.sum(perm_vals >= obs) + 1) / (n_perm + 1))
    return obs, p


def mmd_test_original(emb_tr, emb_te, n_sub=1500, n_perm=1000, seed=42):
    """Direct kernel recomputation, retained for numerical comparisons."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr))
    n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]

    pooled = np.concatenate([X, Y])
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)]
    d2 = np.sum(sub**2,1,keepdims=True) + np.sum(sub**2,1,keepdims=True).T - 2*sub@sub.T
    d2 = np.maximum(d2, 0.0)
    np.fill_diagonal(d2, np.inf)
    vals = d2[np.isfinite(d2) & (d2 > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) > 0 else 1.0
    def rbf(A, B, s):
        d2 = np.sum(A**2,1,keepdims=True)+np.sum(B**2,1,keepdims=True).T-2*A@B.T
        d2 = np.maximum(d2, 0.0)
        return np.exp(-d2/(2*s**2))
    def mmd2(X, Y, s):
        n, m = len(X), len(Y)
        Kxx = rbf(X,X,s); np.fill_diagonal(Kxx,0)
        Kyy = rbf(Y,Y,s); np.fill_diagonal(Kyy,0)
        Kxy = rbf(X,Y,s)
        return float(Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*Kxy.sum()/(n*m))
    obs = mmd2(X, Y, sigma)
    pooled_all = np.concatenate([X, Y]); n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled_all))
        if mmd2(pooled_all[p[:n]], pooled_all[p[n:]], sigma) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def pairwise_dist(Z, backend="numpy"):
    """Float64 Euclidean distance matrix with a zero diagonal."""
    if backend == "torch":
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float64)).to(dev)
        sq = (T * T).sum(1)
        d = T @ T.T
        d.mul_(-2.0).add_(sq[:, None]).add_(sq[None, :])
        d.clamp_(min=0.0).sqrt_()
        d.fill_diagonal_(0.0)
        return d
    Z = np.ascontiguousarray(Z, dtype=np.float64)
    sq = (Z * Z).sum(1)
    d2 = sq[:, None] + sq[None, :] - 2.0 * (Z @ Z.T)
    np.maximum(d2, 0.0, out=d2)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return 0.5 * (d + d.T)


def energy_from_matrix(Dm, P, n1, n2):
    """Energy statistic for each labelling column in P."""
    is_torch = not isinstance(Dm, np.ndarray)
    if is_torch:
        import torch
        P = torch.as_tensor(P, dtype=Dm.dtype, device=Dm.device)
    else:
        P = np.asarray(P, dtype=np.float64)
    S_all = Dm.sum()
    DP = Dm @ P
    S1all = DP.sum(0)
    S11 = (P * DP).sum(0)
    S22 = S_all - 2.0 * S1all + S11
    S12 = S1all - S11
    E = 2.0 * S12 / (n1 * n2) - S11 / (n1 * n1) - S22 / (n2 * n2)
    return E.cpu().numpy() if is_torch else E


def energy_perm_test(X, Y, n_perm, rng, backend="numpy", return_perms=False):
    """Energy permutation test using the supplied random stream."""
    n1, n2 = len(X), len(Y)
    N = n1 + n2
    Dm = pairwise_dist(np.vstack([X, Y]), backend)
    perms = np.stack([rng.permutation(N) for _ in range(n_perm)])
    P = np.zeros((N, n_perm + 1), dtype=np.float64)
    P[:n1, 0] = 1.0
    P[perms[:, :n1].T, np.arange(1, n_perm + 1)[None, :]] = 1.0
    E = energy_from_matrix(Dm, P, n1, n2)
    E_obs, E_perm = float(E[0]), E[1:]
    check_statistics(E_obs, E_perm, "energy distance")
    p = float((np.sum(E_perm >= E_obs) + 1) / (n_perm + 1))
    return (E_obs, p, E_perm, perms) if return_perms else (E_obs, p, E_perm)


def energy_test_fast(emb_tr, emb_te, n_sub=1500, n_perm=500, seed=42, backend="numpy"):
    """Energy test; return (statistic, p-value)."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("energy_test_fast: the subsample contains NaN or inf")
    E_obs, p, _ = energy_perm_test(X, Y, n_perm, rng, backend)
    return E_obs, p


def energy_distance_test_original(emb_tr, emb_te, n_sub=1500, n_perm=500, seed=42):
    """Direct distance recomputation, retained for numerical comparisons."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    def ed_stat(X, Y):
        n, m = len(X), len(Y)
        xy = 0.0
        for s in range(0, n, 400):
            e = min(s+400, n)
            xy += np.sqrt(((X[s:e,None,:]-Y[None,:,:])**2).sum(2)).sum()
        xy /= (n*m)
        xx = 0.0
        for s in range(0, n, 400):
            e = min(s+400, n)
            xx += np.sqrt(((X[s:e,None,:]-X[None,:,:])**2).sum(2)).sum()
        xx /= (n*n)
        yy = 0.0
        for s in range(0, m, 400):
            e = min(s+400, m)
            yy += np.sqrt(((Y[s:e,None,:]-Y[None,:,:])**2).sum(2)).sum()
        yy /= (m*m)
        return float(2*xy - xx - yy)
    obs = ed_stat(X, Y)
    pooled = np.concatenate([X, Y]); n = len(X)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if ed_stat(pooled[p[:n]], pooled[p[n:]]) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def energy_naive(X, Y):
    """Energy statistic from direct pairwise distances."""
    from scipy.spatial.distance import cdist
    return 2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean()


def knn_graph(Z, k, backend="torch"):
    """Build the pooled kNN graph, excluding each point by index."""
    if backend == "sklearn":
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(Z)
        _, idx_full = nbrs.kneighbors(Z)

        not_self = idx_full != np.arange(len(Z))[:, None]
        sel = np.argsort(~not_self, axis=1, kind="stable")[:, :k]
        return np.take_along_axis(idx_full, sel, axis=1)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    T = torch.from_numpy(np.ascontiguousarray(Z, dtype=np.float32)).to(dev)
    sq = (T * T).sum(1)
    n = T.shape[0]; out = np.empty((n, k), dtype=np.int64)
    step = 4096
    with torch.no_grad():
        for s in range(0, n, step):
            e = min(s + step, n)
            d2 = sq[s:e, None] + sq[None, :] - 2.0 * (T[s:e] @ T.T)
            d2[torch.arange(e - s), torch.arange(s, e)] = float("inf")
            out[s:e] = torch.topk(d2, k, dim=1, largest=False).indices.cpu().numpy()
    return out


def mutual_pairs(idx):
    """Count unordered pairs with reciprocal nearest-neighbour edges."""
    n = idx.shape[0]
    back = idx[idx]
    mutual_edge = (back == np.arange(n)[:, None, None]).any(axis=2)
    return int(mutual_edge.sum()) // 2


def knn_conditional_moments(idx, n1, n2, k):
    """Conditional mean and variance of the within-group edge count."""
    n = n1 + n2
    if n <= 3 or min(n1, n2) < 1 or not 1 <= k < n or idx.shape != (n, k):
        raise ValueError("expected n > 3, nonempty groups, and an (n, k) neighbour graph")
    M = mutual_pairs(idx)
    indegree = np.bincount(idx.ravel(), minlength=n).astype(np.float64)
    v = 2.0 * M / (n * k)
    c = np.sum((indegree - k) ** 2) / (n * k)
    a = 4.0 * (n1 - 1) * (n2 - 1) / ((n - 2) * (n - 3))
    prefactor = k * n1 * n2 / (n - 1)
    term1 = a * (1.0 + v - 2.0 * k / (n - 1))
    term2 = (1.0 - a) * c
    mean_count = k * (n1 * (n1 - 1) + n2 * (n2 - 1)) / (n - 1)
    var_count = prefactor * (term1 + term2)
    tol = 64 * np.finfo(float).eps * prefactor * (abs(term1) + abs(term2))
    if abs(var_count) <= tol:
        var_count = 0.0
    elif var_count < 0:
        raise ValueError("negative conditional variance: check the neighbour graph")
    return float(mean_count), float(var_count)


def schilling_p(idx, tags, n1, n2, k):
    """Schilling test using the graph-conditioned normal approximation."""
    coincidences = int((tags[:, None] == tags[idx]).sum())
    mean_count, var_count = knn_conditional_moments(idx, n1, n2, k)
    if var_count == 0:
        return 1.0
    z = (coincidences - mean_count) / np.sqrt(var_count)
    return float(norm.sf(z))


def henze_p(idx, tags_obs, n_perm, seed):
    """Henze label-permutation test on a fixed graph."""
    rng = np.random.default_rng(seed)
    def stat(tags):
        return (tags[:, None] == tags[idx]).sum()
    T_obs = stat(tags_obs)
    T_perm = np.array([stat(rng.permutation(tags_obs)) for _ in range(n_perm)])
    return float((np.sum(T_perm >= T_obs) + 1) / (n_perm + 1))


def c2st_test(emb_tr, emb_te, n_sub=8000, seed=42):
    """MLP classifier two-sample test; return (held-out accuracy, p-value)."""
    rng = np.random.RandomState(seed)
    n_each = min(n_sub, len(emb_tr), len(emb_te))
    X_p = emb_tr[rng.permutation(len(emb_tr))[:n_each]]
    X_q = emb_te[rng.permutation(len(emb_te))[:n_each]]
    X = np.concatenate([X_p, X_q])
    if not np.isfinite(X).all():
        raise ValueError("c2st_test: the subsample contains NaN or inf")
    y = np.array([0] * n_each + [1] * n_each)
    shuf = rng.permutation(len(X))
    X, y = X[shuf], y[shuf]
    n_tr = len(X) // 2
    X_train, y_train = X[:n_tr], y[:n_tr]
    X_test, y_test = X[n_tr:], y[n_tr:]
    n_te = len(X_test)
    clf = MLPClassifier(hidden_layer_sizes=(20,), activation='relu',
                        solver='adam', max_iter=100, random_state=seed)
    clf.fit(X_train, y_train)
    acc = float((clf.predict(X_test) == y_test).mean())
    se = np.sqrt(1.0 / (4.0 * n_te))
    z = (acc - 0.5) / se
    return acc, float(1.0 - norm.cdf(z))


def mmdagg_test(emb_tr, emb_te, n_sub=1500, seed=42):
    """Author-provided NumPy MMDAgg; return 1 for rejection and 0 otherwise."""
    if mmdagg_fn is None:
        raise ImportError("mmdagg is required. Install with: pip install mmdagg")
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]].astype(np.float32)
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]].astype(np.float32)
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        raise ValueError("mmdagg_test: the subsample contains NaN or inf")
    return int(mmdagg_fn(X, Y, seed=seed))


def ed_cached(emb_tr, emb_te, n_sub, n_perm, seed=42):
    """Historical float32 cache and scaled energy statistic."""
    rng = np.random.RandomState(seed)
    X = emb_tr[rng.permutation(len(emb_tr))[:n_sub]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_sub]]
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    N = n + m
    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2 * pooled @ pooled.T, 0.0)
    Dm = np.sqrt(d2, out=d2).astype(np.float32)
    np.fill_diagonal(Dm, 0.0)
    tot = float(Dm.sum())

    def stat(z):
        S = Dm @ z
        xx = float(z @ S)
        sx = float(S.sum())
        xy = sx - xx
        yy = tot - xx - 2 * xy
        return (n * m / (n + m)) * (2 * xy / (n * m) - xx / (n * n) - yy / (m * m))

    z0 = np.zeros(N, dtype=np.float32)
    z0[:n] = 1.0
    obs = stat(z0)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(N, dtype=np.float32)
        z[rng.permutation(N)[:n]] = 1.0
        if stat(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def _mmd_float32(X, Y, n_perm, rng):
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    N = n + m
    sub = pooled[rng.choice(N, min(2000, N), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0)
    np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0

    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2*pooled@pooled.T, 0.0)
    Km = np.exp(-d2 / (2*sigma**2)).astype(np.float32)
    del d2
    np.fill_diagonal(Km, 0.0)
    tot = float(Km.sum())

    def stat(z):

        S = Km @ z
        Kxx = float(z @ S)
        sx = float(S.sum())
        Kyy = tot - 2*sx + Kxx
        Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)

    z0 = np.zeros(N, dtype=np.float32)
    z0[:n] = 1.0
    obs = stat(z0)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(N, dtype=np.float32)
        z[rng.permutation(N)[:n]] = 1.0
        if stat(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def mmd_test(emb_tr, emb_te, n_sub=1500, n_perm=1000, seed=42):
    """Historical MMD protocol with a float32 cached kernel."""
    rng = np.random.RandomState(seed)
    n_x = min(n_sub, len(emb_tr))
    n_y = min(n_sub, len(emb_te))
    X = emb_tr[rng.permutation(len(emb_tr))[:n_x]]
    Y = emb_te[rng.permutation(len(emb_te))[:n_y]]
    return _mmd_float32(X, Y, n_perm, rng)


def norm_perm_test(ntr, nte, n_perm=2000, seed=0):
    """Two-sided median permutation test on feature lengths."""
    rng = np.random.RandomState(seed)
    obs = np.median(nte) - np.median(ntr)
    pooled = np.concatenate([ntr, nte])
    na = len(ntr)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def mmd_fast(X, Y, n_perm, seed=42):
    """Historical float32-cached MMD on already selected points."""
    return _mmd_float32(X, Y, n_perm, np.random.RandomState(seed))


def mmd_cached(X, Y, n_perm=1000, seed=42):
    """Historical mask-summed float32 MMD on already selected points."""
    rng = np.random.RandomState(seed)
    pooled = np.concatenate([X, Y]).astype(np.float64)
    n, m = len(X), len(Y)
    sub = pooled[rng.choice(len(pooled), min(2000, len(pooled)), replace=False)]
    d2s = np.sum(sub**2, 1, keepdims=True) + np.sum(sub**2, 1, keepdims=True).T - 2*sub@sub.T
    d2s = np.maximum(d2s, 0.0)
    np.fill_diagonal(d2s, np.inf)
    vals = d2s[np.isfinite(d2s) & (d2s > 0)]
    sigma = max(float(np.sqrt(np.median(vals))), 1e-8) if len(vals) else 1.0

    sq = np.sum(pooled**2, 1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2*pooled@pooled.T, 0.0)
    Km = np.exp(-d2 / (2*sigma**2)).astype(np.float32)
    np.fill_diagonal(Km, 0.0)
    tot = Km.sum()
    rows = Km.sum(axis=1)

    def mmd2_from_mask(zx):

        sx = rows[zx].sum()
        Kxx = Km[np.ix_(zx, zx)].sum()
        Kyy = tot - 2*sx + Kxx
        Kxy = sx - Kxx
        return Kxx/(n*(n-1)) + Kyy/(m*(m-1)) - 2*Kxy/(n*m)

    base = np.zeros(n+m, dtype=bool)
    base[:n] = True
    obs = mmd2_from_mask(base)
    count = 0
    for _ in range(n_perm):
        z = np.zeros(n+m, dtype=bool)
        z[rng.permutation(n+m)[:n]] = True
        if mmd2_from_mask(z) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)
