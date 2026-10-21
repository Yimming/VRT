"""
ssl_nll.py — the SimCLR objective on training and test images at every checkpoint (camera-ready #27102, October 12).

The supervised models report train and test cross-entropy per epoch (Figure 2, Appendix F, the NLL paragraph of
Section 4.2). The SimCLR models have no labels in their objective; the analogue is the contrastive loss itself, the
NT-Xent / SupContrast "SimCLR" loss, which is the negative log-likelihood of picking the positive view among the batch.
This script evaluates that loss, with the training augmentations, on
  (a) a fixed random subset of 10,000 training images (seed 0), and
  (b) the 10,000 test images,
at every checkpoint ckpt_epoch_<e>.pth with lo <= e <= hi of a SimCLR run. Two augmented views per image are drawn ONCE
with a fixed seed and reused for every checkpoint (SupContrast's CIFAR pipeline: RandomResizedCrop(32, scale 0.2 to 1),
horizontal flip, colour jitter 0.4/0.4/0.4/0.1 with p = 0.8, grayscale p = 0.2, mean/std normalisation), so that only the
model changes along the trajectory. Batches of 512 images (1,024 views) in a fixed order, drop_last, temperature 0.5.
The loss is SupContrast's SupConLoss without labels (contrast_mode "all", scaled by temperature / base_temperature with
base_temperature 0.07), i.e. the quantity written to train_log.csv during training, imported from the training code when
available and otherwise reimplemented below with the same arithmetic.
Both BatchNorm modes are reported: eval mode (running statistics) and train mode (batch statistics, no gradient).

Output: <out_dir>/ssl_nll_<dataset>_c<lo>-<hi>.csv with columns
  epoch, tr_eval, te_eval, gap_eval, tr_train, te_train, gap_train      (gap = test minus train)
appended one row per epoch as they finish; epochs present in any ssl_nll_<dataset>_c*.csv of out_dir are skipped (resume).
Usage (GPU node, venv active, from the s50 code folder, which holds train_simclr_supcon.py and supcon_fill_vrt.py):
  python ssl_nll.py --ckpt_dir "$(ls -d /scratch/p318770/model_zoos/ssl_supcon/SimCLR_cifar10_* | head -1)" \
      --dataset cifar10 --lo 1 --hi 100 --out_dir /home3/p318770/venvs/overfitting/benchmark_final/s50/ssl_nll
"""
import argparse, csv, os, sys, time
import numpy as np
import torch
import torchvision
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from supcon_fill_vrt import SupConResNet, load_ckpt, MEAN_STD, DATA_FOLDER

try:
    from train_simclr_supcon import SupConLoss
except ImportError:
    try:
        from losses import SupConLoss
    except ImportError:
        SupConLoss = None

N_TRAIN_SUBSET = 10000
BATCH = 512
TEMP = 0.5
BASE_TEMP = 0.07
VIEW_SEED = 0


def supcon_simclr_loss(features, temperature=TEMP, base_temperature=BASE_TEMP):
    """SupContrast SupConLoss(temperature)(features) with no labels, contrast_mode 'all'; features [bsz, 2, d], normalized"""
    bsz, n_views = features.shape[0], features.shape[1]
    device = features.device
    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)            # [2 bsz, d]: view 1 of all, then view 2 of all
    logits = contrast @ contrast.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True)[0].detach()
    mask = torch.eye(bsz, device=device).repeat(n_views, n_views)         # same image
    logits_mask = 1.0 - torch.eye(bsz * n_views, device=device)           # not self
    mask = mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -(temperature / base_temperature) * mean_log_prob_pos
    return loss.view(n_views, bsz).mean()


class TwoCrop:
    def __init__(self, tf): self.tf = tf
    def __call__(self, x): return [self.tf(x), self.tf(x)]


def build_views(dataset, train, indices, device):
    """two augmented views per image, drawn once with a fixed seed; returns float tensors [N, 3, 32, 32] x 2 on the CPU"""
    mean, std = MEAN_STD[dataset]
    tf = T.Compose([T.RandomResizedCrop(size=32, scale=(0.2, 1.0)), T.RandomHorizontalFlip(),
                    T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8), T.RandomGrayscale(p=0.2),
                    T.ToTensor(), T.Normalize(mean, std)])
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    ds = cls(root=DATA_FOLDER, train=train, transform=TwoCrop(tf), download=False)
    torch.manual_seed(VIEW_SEED + (0 if train else 1))
    v1, v2 = [], []
    for i in indices:
        a, b = ds[int(i)][0]
        v1.append(a); v2.append(b)
    return torch.stack(v1), torch.stack(v2)


@torch.no_grad()
def mean_loss(model, v1, v2, device):
    n = (len(v1) // BATCH) * BATCH
    losses = []
    for s in range(0, n, BATCH):
        x = torch.cat([v1[s:s + BATCH], v2[s:s + BATCH]]).to(device, non_blocking=True)
        z = model(x)                                                       # normalized projection, [2 B, d]
        f = torch.stack(torch.split(z, BATCH), dim=1)                      # [B, 2, d]
        if SupConLoss is not None:
            losses.append(float(SupConLoss(temperature=TEMP)(f)))
        else:
            losses.append(float(supcon_simclr_loss(f)))
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--lo", type=int, default=1)
    ap.add_argument("--hi", type=int, default=1000)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    import glob
    done = set()
    for f in glob.glob(os.path.join(args.out_dir, f"ssl_nll_{args.dataset}_c*.csv")):   # every range file of this dataset
        with open(f) as fh:
            done |= {int(r["epoch"]) for r in csv.DictReader(fh)}
    out = os.path.join(args.out_dir, f"ssl_nll_{args.dataset}_c{args.lo}-{args.hi}.csv")
    if not os.path.exists(out):
        with open(out, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "tr_eval", "te_eval", "gap_eval", "tr_train", "te_train", "gap_train"])
    names = [n for n in os.listdir(args.ckpt_dir) if n.startswith("ckpt_epoch_") and n.endswith(".pth")]
    eps = sorted(int(n[len("ckpt_epoch_"):-4]) for n in names)
    todo = [e for e in eps if args.lo <= e <= args.hi and e not in done]
    print(f"{args.dataset}: {len(todo)} checkpoints to do in [{args.lo}, {args.hi}] ({len(done)} already in {out}); "
          f"loss = {'SupConLoss from training code' if SupConLoss is not None else 'reimplemented SupConLoss'}, "
          f"temperature {TEMP}, batch {BATCH}, device {device}", flush=True)
    if not todo:
        return
    rng = np.random.RandomState(0)
    tr_idx = rng.permutation(50000)[:N_TRAIN_SUBSET]
    t0 = time.time()
    tr1, tr2 = build_views(args.dataset, True, tr_idx, device)
    te1, te2 = build_views(args.dataset, False, np.arange(10000), device)
    print(f"views built in {time.time() - t0:.0f} s: train subset {len(tr1)}, test {len(te1)}", flush=True)
    model = SupConResNet(name=args.model).to(device)
    for e in todo:
        t1 = time.time()
        load_ckpt(model, os.path.join(args.ckpt_dir, f"ckpt_epoch_{e}.pth"), device)
        model.eval()
        tr_e, te_e = mean_loss(model, tr1, tr2, device), mean_loss(model, te1, te2, device)
        model.train()                                                      # batch statistics; running stats are discarded with the next load
        tr_t, te_t = mean_loss(model, tr1, tr2, device), mean_loss(model, te1, te2, device)
        with open(out, "a", newline="") as f:
            csv.writer(f).writerow([e, f"{tr_e:.6f}", f"{te_e:.6f}", f"{te_e - tr_e:.6f}", f"{tr_t:.6f}", f"{te_t:.6f}", f"{te_t - tr_t:.6f}"])
        print(f"  epoch {e:4d}: eval train {tr_e:.4f} test {te_e:.4f} gap {te_e - tr_e:+.4f} | train-mode BN train {tr_t:.4f} "
              f"test {te_t:.4f} gap {te_t - tr_t:+.4f}  ({time.time() - t1:.0f} s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
