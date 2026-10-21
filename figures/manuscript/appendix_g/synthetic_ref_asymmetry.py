"""
synthetic_ref_asymmetry.py — Reference set choice on S^{D-1}
======================================================================
For each alternative, run VRT
with sample A as the reference set and with sample B as the reference set. 500 repetitions each.
Reports rejection rate at alpha=0.05.

Self-contained.

Usage:
  python synthetic_ref_asymmetry.py
"""
import numpy as np, time
from scipy.special import betainc, betaln

D = 500
N = 5000
N_REP = 500
ALPHA = 0.05

# ── VRT ──
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

def vrt_test(emb_A, emb_B, D, k=60, M=2000, n_qtr=8000, n_perm=2000, seed=42):
    """Standard VRT: A is reference, B is test."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(emb_A))
    ref = emb_A[perm[:M]]
    q_A = emb_A[perm[M:M+min(n_qtr, len(emb_A)-M)]]
    c_A = compute_chat(q_A, ref, D, k)
    c_B = compute_chat(emb_B, ref, D, k)
    obs = np.median(c_B) - np.median(c_A)
    pooled = np.concatenate([c_A, c_B]); na = len(c_A)
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(len(pooled))
        if abs(np.median(pooled[p[na:]]) - np.median(pooled[p[:na]])) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)

# ── Samplers ──
def sample_uniform(n, D, rng):
    x = rng.standard_normal((n, D))
    return x / np.linalg.norm(x, axis=1, keepdims=True)

def sample_vmf(n, D, mu, kappa, rng):
    if kappa < 1e-10:
        return sample_uniform(n, D, rng)
    d = D
    b = (-2*kappa + np.sqrt(4*kappa**2 + (d-1)**2)) / (d-1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d-1) * np.log(1 - x0**2)
    samples = np.empty((n, D))
    idx = 0
    batch = max(n, 256)
    while idx < n:
        z = rng.beta((d-1)/2, (d-1)/2, size=batch)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = rng.uniform(size=batch)
        accept = kappa*w + (d-1)*np.log(1 - x0*w) - c >= np.log(u)
        w = w[accept]
        if len(w) == 0: continue
        need = min(len(w), n - idx)
        w = w[:need]
        v = rng.standard_normal((need, D-1))
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        sq = np.sqrt(np.maximum(1 - w**2, 0))[:, None]
        pts = np.concatenate([w[:, None], sq * v], axis=1)
        # Rotate to mu
        e1 = np.zeros(D); e1[0] = 1.0
        if not np.allclose(mu, e1):
            u_vec = mu - e1
            u_vec = u_vec / np.linalg.norm(u_vec)
            pts = pts - 2 * np.outer(pts @ u_vec, u_vec)
        samples[idx:idx+need] = pts
        idx += need
    return samples

# ── Alternatives ──
mu = np.zeros(D); mu[0] = 1.0
kappa_base = D * 0.5  # = 250

alternatives = [
    # (name, description, sampler_A, sampler_B)
    ('conc_0.3',
     'Concentration shift dk=0.3',
     lambda n, rng: sample_vmf(n, D, mu, kappa_base, rng),
     lambda n, rng: sample_vmf(n, D, mu, kappa_base + 0.3 * np.sqrt(D), rng)),

    ('mean_2deg',
     'Mean-direction shift 2 degrees',
     lambda n, rng: sample_vmf(n, D, mu, kappa_base, rng),
     lambda n, rng: sample_vmf(n, D, np.concatenate([[np.cos(np.radians(2))], [np.sin(np.radians(2))], np.zeros(D-2)]), kappa_base, rng)),

    ('mixture_0.05',
     'Mixture eps=0.05',
     lambda n, rng: sample_uniform(n, D, rng),
     lambda n, rng: _sample_mixture(n, D, 0.05, D*2.0, rng)),

    ('unif_vs_vmf',
     'Uniform vs vMF(250)',
     lambda n, rng: sample_uniform(n, D, rng),
     lambda n, rng: sample_vmf(n, D, mu, kappa_base, rng)),
]

def _sample_mixture(n, D, eps, kappa, rng):
    mu_mix = np.zeros(D); mu_mix[0] = 1.0
    n_cluster = rng.binomial(n, eps)
    n_uniform = n - n_cluster
    parts = []
    if n_uniform > 0: parts.append(sample_uniform(n_uniform, D, rng))
    if n_cluster > 0: parts.append(sample_vmf(n_cluster, D, mu_mix, kappa, rng))
    out = np.concatenate(parts, 0)
    rng.shuffle(out)
    return out

# ── Main ──
print(f"=== Synthetic reference asymmetry | D={D}, N={N}, {N_REP} reps ===")
print(f"    For each alternative, run VRT under two reference-set choices:")
print(f"    (1) A as reference: reference from A, query from both")
print(f"    (2) B as reference: reference from B, query from both\n")

results = []
for alt_name, desc, sampler_A, sampler_B in alternatives:
    print(f"\n{'='*70}")
    print(f"{alt_name}: {desc}")
    print(f"{'='*70}")

    rej_A_ref = 0  # A as reference
    rej_B_ref = 0  # B as reference

    t0 = time.time()
    for rep in range(N_REP):
        rng = np.random.RandomState(rep)
        A = sampler_A(N, rng).astype(np.float64)
        B = sampler_B(N, rng).astype(np.float64)

        # A as reference: A as reference
        _, p1 = vrt_test(A, B, D, seed=rep)
        if p1 < ALPHA: rej_A_ref += 1

        # B as reference: B as reference
        _, p2 = vrt_test(B, A, D, seed=rep)
        if p2 < ALPHA: rej_B_ref += 1

        if (rep + 1) % 50 == 0:
            dt = time.time() - t0
            print(f"  rep {rep+1:4d}/{N_REP}: "
                  f"A-ref={100*rej_A_ref/(rep+1):5.1f}%  "
                  f"B-ref={100*rej_B_ref/(rep+1):5.1f}%  "
                  f"({dt:.0f}s)", flush=True)

    dt = time.time() - t0
    print(f"\n  FINAL: A-as-ref = {rej_A_ref}/{N_REP} ({100*rej_A_ref/N_REP:.1f}%)")
    print(f"         B-as-ref = {rej_B_ref}/{N_REP} ({100*rej_B_ref/N_REP:.1f}%)")
    print(f"         Time: {dt:.0f}s")

    results.append((alt_name, rej_A_ref, rej_B_ref))

print(f"\n{'='*70}")
print("Summary")
print(f"{'='*70}")
print(f"{'Alternative':>15s}  {'A ref':>10s}  {'B ref':>10s}")
print('-' * 45)
for name, ra, rb in results:
    print(f"{name:>15s}  {ra:4d}/{N_REP} ({100*ra/N_REP:5.1f}%)  {rb:4d}/{N_REP} ({100*rb/N_REP:5.1f}%)")

print("\nDone.")
