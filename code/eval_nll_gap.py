"""ep62 absolute NLL gap (paper metric, cf. 0.563) for the 4 seed reruns."""
import torch, torch.nn as nn, torchvision as tv
import torchvision.transforms as T

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
NORM = T.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
tf = T.Compose([T.ToTensor(), NORM])
root = '/home3/p318770/venvs/overfitting'
tr = tv.datasets.CIFAR100(root, train=True, transform=tf, download=False)
te = tv.datasets.CIFAR100(root, train=False, transform=tf, download=False)

def build():
    m = tv.models.resnet50(num_classes=100)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    return m

@torch.no_grad()
def mean_nll(m, ds):
    ld = torch.utils.data.DataLoader(ds, batch_size=1000, num_workers=4)
    tot, n = 0.0, 0
    for x, y in ld:
        x, y = x.to(DEV), y.to(DEV)
        tot += nn.functional.cross_entropy(m(x), y, reduction='sum').item()
        n += len(y)
    return tot / n

for s in (1, 2, 3, 4):
    ck = torch.load(f'/scratch/p318770/model_zoos/resnet50_cifar100_seed{s}/checkpoint_062.pt',
                    map_location=DEV)
    sd = ck.get('model', ck.get('state_dict', ck))
    m = build().to(DEV); m.load_state_dict(sd); m.eval()
    a, b = mean_nll(m, tr), mean_nll(m, te)
    print(f'seed{s}: train NLL={a:.4f} test NLL={b:.4f} gap={b-a:.4f}', flush=True)
