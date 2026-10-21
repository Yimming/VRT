"""Feature extraction contracts retained by the compact entry points."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models


def test_checkpoint_loading_accepts_wrapped_and_parallel_weights(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": {"module." + k: v for k, v in source.state_dict().items()}}, path)
    models.load_backbone(target, str(path), "r18_c10", "cpu")
    for key, value in source.state_dict().items():
        assert torch.equal(value, target.state_dict()[key])


def test_cifar_train_mode_restores_bn_before_each_split(monkeypatch):
    class SmallCifar:
        def __init__(self, root, train, **kwargs):
            self.value = 1.0 if train else 2.0

        def __len__(self):
            return 4

        def __getitem__(self, index):
            return torch.full((3, 2, 2), self.value), 0

    monkeypatch.setattr(models.torchvision.datasets, "CIFAR10", SmallCifar)
    model = torch.nn.BatchNorm2d(3)
    initial_means = []

    def forward(module, x):
        initial_means.append(module.running_mean.clone())
        return module(x).mean((2, 3))

    result = models.cifar_features(model, forward, "r18_c10", "unused", "train", workers=0)
    assert result["h_tr"].shape == result["h_te"].shape == (4, 3)
    assert result["h_tr"].dtype == np.float32
    assert len(initial_means) == 2
    assert all(torch.equal(value, torch.zeros(3)) for value in initial_means)


def test_imagenet_train_train_uses_one_seeded_extraction(monkeypatch):
    calls = []

    def extract_raw(model, loader, maximum, seed, mode, device):
        calls.append((loader, maximum, seed, mode))
        indices = np.arange(maximum)
        return indices[:, None].astype(np.float32), indices

    monkeypatch.setattr(models, "extract_raw", extract_raw)
    result = models.imagenet_features(None, "train", "test", "train", "trtr", "cpu")
    assert calls == [("train", 60000, 42, "train")]
    assert result["h_tr"].shape == (50000, 1)
    assert result["h_te"].shape == (10000, 1)
    assert not np.intersect1d(result["idx_tr"], result["idx_te"]).size


def test_simclr_and_supervised_cifar10_keep_distinct_normalization():
    assert models.stats_for("r18_c10") == models.C10_STATS
    assert models.stats_for("simclr_c10") == models.MEAN_STD["cifar10"]
    assert tuple(models.stats_for("r18_c10")[1]) != tuple(models.stats_for("simclr_c10")[1])


def test_contrastive_loss_pairs_views_and_drops_incomplete_batch():
    import train
    generator = torch.Generator().manual_seed(15)
    first = torch.nn.functional.normalize(torch.randn(5, 4, generator=generator), dim=1)
    second = torch.nn.functional.normalize(torch.randn(5, 4, generator=generator), dim=1)
    first[-1] = second[-1] = float("nan")
    expected = []
    for start in (0, 2):
        combined = torch.cat([first[start:start + 2], second[start:start + 2]])
        logits = combined @ combined.T / 0.5
        positive = logits[torch.arange(4), (torch.arange(4) + 2) % 4]
        logits.fill_diagonal_(float("-inf"))
        expected.append(float(-(0.5 / 0.07) * (positive - logits.logsumexp(dim=1)).mean()))
    actual = train.mean_contrastive_loss(torch.nn.Identity(), first, second, "cpu", batch_size=2)
    assert actual == pytest.approx(np.mean(expected), abs=2e-6)


def test_fixed_view_augmentation_repeats_for_each_checkpoint(monkeypatch):
    import train
    from PIL import Image

    class SmallCifar:
        def __init__(self, root, train, transform, **kwargs):
            self.transform = transform
            self.image = Image.fromarray(np.random.RandomState(99).randint(0, 256, (32, 32, 3), dtype=np.uint8))

        def __getitem__(self, index):
            return self.transform(self.image), 0

    monkeypatch.setattr(train.torchvision.datasets, "CIFAR10", SmallCifar)
    first = train.build_loss_views("cifar10", True, [0, 1], "unused")
    second = train.build_loss_views("cifar10", True, [0, 1], "unused")
    test = train.build_loss_views("cifar10", False, [0, 1], "unused")
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    assert any(not torch.equal(a, b) for a, b in zip(first, test))
