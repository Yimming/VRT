"""
typeI_imagenet_verbatim.py — the supplementary script typeI_imagenet_500seeds.py (Appendix C random-initialization
null, VRT only) run VERBATIM on the full ImageNet copy in $TMPDIR, to compare with the 22/30 eval-mode rejections of
the s50 null pipeline (imagenet_randinit_extract.py + vrt_forward_budget.py) on the same images.
Changes to the published file, and nothing else: data root from the command line (default the full ImageNet copy in
$TMPDIR, as in imagenet_multiseed_vrt.py), vrt_test imported from the supplementary methods.py copied next to this file
(supp_methods.py), the number of seeds, the first seed and an output CSV from the command line, per-seed rows appended
to the CSV (resume by seed), and a --preproc switch between the two input pipelines that exist in the project:
  paper   : Resize(224), CenterCrop(224), ToTensor()                    (supplementary code, Table 1 feature pipeline,
            imagenet_extract.py; no mean/std normalization, as the checkpoints were trained)
  cluster : Resize(256), CenterCrop(224), ToTensor(), Normalize(ImageNet mean/std)
            (the version of typeI_imagenet_500seeds.py that actually ran on Habrok for the 18/500 of Appendix C)
  python typeI_imagenet_verbatim.py --preproc paper   --n_seeds 30 --out $BF/null/typeI_verbatim_paper.csv
  python typeI_imagenet_verbatim.py --preproc cluster --n_seeds 30 --out $BF/null/typeI_verbatim_cluster.csv
"""
import argparse, csv
import os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from supp_methods import vrt_test          # the supplementary src/methods.py, unchanged

ap = argparse.ArgumentParser()
ap.add_argument('--n_seeds', type=int, default=30)
ap.add_argument('--seed0', type=int, default=0, help='init seeds seed0 .. seed0+n_seeds-1 (the published run used 0..499)')
ap.add_argument('--preproc', choices=['paper', 'cluster'], default='paper')
ap.add_argument('--data_root', default=os.path.join(os.environ.get('TMPDIR', '/tmp'), 'ILSVRC', 'Data', 'CLS-LOC'),
                help='folder with train/ (class folders) and val/ (flat JPEGs); default the full ImageNet copy in $TMPDIR')
ap.add_argument('--out', required=True)
args = ap.parse_args()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SEEDS = args.n_seeds
D = 2048
ALPHA = 0.05
MAX_TR = 50000
MAX_TE = 10000

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels,out_channels//4,1,bias=False)
        self.bn1=nn.BatchNorm2d(out_channels//4)
        self.conv2=nn.Conv2d(out_channels//4,out_channels//4,3,padding=1,stride=2 if downsample else 1,bias=False)
        self.bn2=nn.BatchNorm2d(out_channels//4)
        self.conv3=nn.Conv2d(out_channels//4,out_channels,1,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        self.relu=nn.ReLU()
        self.downsample_fn=nn.AvgPool2d(1,stride=2) if downsample else nn.Identity()
        self.need_pad=(out_channels!=in_channels)
        self.pad_channels=out_channels-in_channels if self.need_pad else 0
    def forward(self,x):
        y=self.relu(self.bn1(self.conv1(x)))
        y=self.relu(self.bn2(self.conv2(y)))
        y=self.bn3(self.conv3(y))
        skip=self.downsample_fn(x)
        if self.need_pad:
            pad=torch.zeros(skip.shape[0],self.pad_channels,skip.shape[2],skip.shape[3],device=x.device)
            skip=torch.cat([skip,pad],dim=1)
        return self.relu(y+skip)

class DeepResNet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,64,7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU())
        self.max_pool=nn.MaxPool2d(3,stride=2,padding=1)
        widths=[3,4,6,3];channels=[256,512,1024,2048];in_ch=[64,256,512,1024]
        self.levels=nn.ModuleList()
        for i,(w,ic,oc) in enumerate(zip(widths,in_ch,channels)):
            blocks=[]
            for j in range(w):
                blocks.append(BottleNeckBlock(ic if j==0 else oc,oc,downsample=(i>0 and j==0)))
            self.levels.append(nn.Sequential(*blocks))
    def forward_features(self,x):
        x=self.stem(x);x=self.max_pool(x)
        for level in self.levels: x=level(x)
        return F.adaptive_avg_pool2d(x,1).flatten(1)

class FlatImageFolder(Dataset):
    def __init__(self,root,transform):
        self.paths=sorted(glob(os.path.join(root,'*.JPEG')))
        if len(self.paths)==0: self.paths=sorted(glob(os.path.join(root,'*','*.JPEG')))
        self.transform=transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.transform(Image.open(self.paths[i]).convert('RGB')),0

class SubsampledImageFolder(Dataset):
    def __init__(self,root,transform):
        self.transform=transform;self.paths=[]
        for cls_dir in sorted(os.listdir(root)):
            cls_path=os.path.join(root,cls_dir)
            if os.path.isdir(cls_path):
                for f in sorted(os.listdir(cls_path)):
                    if f.endswith('.JPEG'): self.paths.append(os.path.join(cls_path,f))
    def __len__(self): return len(self.paths)
    def __getitem__(self,i): return self.transform(Image.open(self.paths[i]).convert('RGB')),0

@torch.inference_mode()
def extract(model, loader, max_n, data_seed):
    torch.manual_seed(data_seed)
    model.eval(); out=[]; n=0
    for x,_ in loader:
        feat = F.normalize(model.forward_features(x.to(DEVICE)), dim=1).cpu()
        out.append(feat); n += feat.shape[0]
        if n >= max_n: break
    return torch.cat(out,0)[:max_n].numpy().astype(np.float64)

print(f"Device: {DEVICE}")
print(f"VRT Type-I: {N_SEEDS} random DeepResNet50 on ImageNet\n")

if args.preproc == 'paper':
    tfm = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor()])
else:
    tfm = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
tr_ds = SubsampledImageFolder(os.path.join(args.data_root, 'train'), tfm)
te_ds = FlatImageFolder(os.path.join(args.data_root, 'val'), tfm)
print(f"preproc {args.preproc}  data {args.data_root}  train images {len(tr_ds)}  val images {len(te_ds)}  (full ImageNet: 1281167 / 50000)", flush=True)
kw = dict(batch_size=256, num_workers=4, pin_memory=True)
tr_l = DataLoader(tr_ds, shuffle=True, **kw)
te_l = DataLoader(te_ds, shuffle=True, **kw)

print(f"{'seed':>5s}  {'dc':>10s}  {'p':>8s}")
print('-' * 30)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
done = set()
if os.path.exists(args.out):
    done = {int(r['seed']) for r in csv.DictReader(open(args.out))}
new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
fout = open(args.out, 'a')
if new:
    fout.write('seed,dc,p\n'); fout.flush()
reject = 0; n_done = 0
for seed in range(args.seed0, args.seed0 + N_SEEDS):
    if seed in done:
        continue
    torch.manual_seed(seed)
    model = DeepResNet50().to(DEVICE)
    e_tr = extract(model, tr_l, MAX_TR, 42)
    e_te = extract(model, te_l, MAX_TE, 123)

    dc, p = vrt_test(e_tr, e_te, D=D, seed=seed)
    sig = '*' if p < ALPHA else ' '
    print(f"{seed:5d}  {dc:10.6f}  {p:8.4f} {sig}", flush=True)
    fout.write(f"{seed},{dc:.6f},{p:.6f}\n"); fout.flush()
    n_done += 1
    if p < ALPHA:
        reject += 1

fout.close()
allp = [float(r['p']) for r in csv.DictReader(open(args.out))]
print(f"\n{'='*40}")
print(f"this run: VRT reject {reject}/{n_done};  all rows in {args.out}: {sum(p < ALPHA for p in allp)}/{len(allp)} = {sum(p < ALPHA for p in allp)/max(1,len(allp)):.4f}  (expected under a true null: {ALPHA})")
print("Done.")
