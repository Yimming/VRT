"""cifar_split_pixels.py — do the official CIFAR test images differ from the training images at the pixel level?
For CIFAR-10 and CIFAR-100, five statistics of every image: mean R, G and B (on [0, 1]), contrast (standard deviation of
the grayscale image) and sharpness (mean absolute difference of neighbouring grayscale pixels). For each statistic: the
test mean minus the training mean, in units of the per-image standard deviation, its z-score and a two-sided p-value
against B class-stratified random splits of the 60,000 images with the official sizes (what a random split gives); and
one combined p-value for all five (Mahalanobis distance of the mean differences). A small p means the official test set
differs from a random split of the same images in that statistic, a difference that randomly initialized networks see.
Run on the login node (numpy only, about one to two minutes, 2 GB of memory):  python cifar_split_pixels.py [B]
"""
import os, sys, pickle, time
import numpy as np

DATA = "/home3/p318770/venvs/overfitting"
B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NAMES = ["mean R", "mean G", "mean B", "contrast", "sharpness"]


def load(ds):
    if ds == "cifar10":
        d = os.path.join(DATA, "cifar-10-batches-py")
        tr = [pickle.load(open(os.path.join(d, f"data_batch_{i}"), "rb"), encoding="bytes") for i in range(1, 6)]
        te = pickle.load(open(os.path.join(d, "test_batch"), "rb"), encoding="bytes")
        Xtr, ytr = np.concatenate([b[b"data"] for b in tr]), np.concatenate([b[b"labels"] for b in tr])
        Xte, yte = te[b"data"], np.array(te[b"labels"])
    else:
        d = os.path.join(DATA, "cifar-100-python")
        tr = pickle.load(open(os.path.join(d, "train"), "rb"), encoding="bytes")
        te = pickle.load(open(os.path.join(d, "test"), "rb"), encoding="bytes")
        Xtr, ytr = tr[b"data"], np.array(tr[b"fine_labels"])
        Xte, yte = te[b"data"], np.array(te[b"fine_labels"])
    return np.concatenate([Xtr, Xte]), np.concatenate([ytr, yte]), len(Xtr)


def image_stats(raw):
    out = []
    for s in range(0, len(raw), 5000):                           # rows: 1,024 R, 1,024 G, 1,024 B values, row-major 32 x 32
        X = raw[s:s + 5000].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        g = 0.299 * X[:, 0] + 0.587 * X[:, 1] + 0.114 * X[:, 2]
        sharp = 0.5 * (np.abs(np.diff(g, axis=1)).mean(axis=(1, 2)) + np.abs(np.diff(g, axis=2)).mean(axis=(1, 2)))
        out.append(np.column_stack([X.mean(axis=(2, 3)), g.std(axis=(1, 2)), sharp]))
    return np.concatenate(out).astype(np.float64)


for ds in ("cifar10", "cifar100"):
    t0 = time.time()
    raw, y, n_tr = load(ds)
    S = image_stats(raw)
    n = len(S); n_te = n - n_tr
    is_te = np.zeros(n, bool); is_te[n_tr:] = True
    total = S.sum(0)

    def diff(mask):                                              # test mean minus training mean, per statistic
        st = S[mask].sum(0)
        return st / n_te - (total - st) / n_tr

    d_obs = diff(is_te)
    classes = np.unique(y)
    members = [np.where(y == c)[0] for c in classes]
    k_te = [int(is_te[m].sum()) for m in members]                # official test images per class (1,000 or 100)
    rng = np.random.default_rng(2026)
    D = np.empty((B, S.shape[1]))
    for b in range(B):
        mask = np.zeros(n, bool)
        for m, k in zip(members, k_te):
            mask[rng.choice(m, k, replace=False)] = True
        D[b] = diff(mask)
    sd_img = S.std(0)
    Vinv = np.linalg.inv(np.cov(D.T))                            # covariance of the differences under random splits
    T_obs = d_obs @ Vinv @ d_obs
    T_rand = np.einsum("bi,ij,bj->b", D, Vinv, D)
    print(f"\n{ds.upper()}: official test ({n_te}) vs training images ({n_tr}); reference: {B} class-stratified "
          f"random splits of the {n} images  ({time.time() - t0:.0f} s)")
    print(f"{'statistic':10s} {'train':>8s} {'test':>8s} {'diff/sd':>8s} {'z':>6s} {'p':>7s}")
    for j, name in enumerate(NAMES):
        tr_mean = (total[j] - S[is_te, j].sum()) / n_tr
        z = d_obs[j] / D[:, j].std()
        p = (1 + np.sum(np.abs(D[:, j]) >= abs(d_obs[j]))) / (B + 1)
        print(f"{name:10s} {tr_mean:8.4f} {S[is_te, j].mean():8.4f} {d_obs[j] / sd_img[j]:+8.4f} {z:+6.2f} {p:7.4f}")
    print(f"{'all five':10s} {'':8s} {'':8s} {'':8s} {'':6s} {(1 + np.sum(T_rand >= T_obs)) / (B + 1):7.4f}"
          f"   (Mahalanobis, chi2 df 5 would give about the same)")
