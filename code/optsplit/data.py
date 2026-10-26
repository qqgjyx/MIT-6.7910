"""Datasets with a deterministic batch order: the batch at `step` depends only on (data_seed, step)."""

import json

import numpy as np
import torch

from .config import data_dir


def _gen(*keys):
    seed = 0
    for k in keys:
        seed = (seed * 1_000_003 + int(k)) % 2 ** 62
    return torch.Generator().manual_seed(seed)


# separate RNG streams, so measurements never perturb the training order
TRAIN, FIXED, RANDOM = 0, 1, 2


class LMData:
    kind = 'lm'

    def __init__(self, cfg, device):
        d = data_dir(cfg.dataset)
        self.splits = {s: np.memmap(d / f'{s}.bin', dtype=np.uint16, mode='r') for s in ('train', 'val')}
        self.vocab_size = json.loads((d / 'meta.json').read_text())['vocab_size']
        self.T, self.batch_size, self.seed = cfg.model.block_size, cfg.train.batch_size, cfg.train.data_seed
        self.device = device

    def _sample(self, split, n, gen):
        arr = self.splits[split]
        ix = torch.randint(len(arr) - self.T - 1, (n,), generator=gen).tolist()
        xy = torch.from_numpy(np.stack([arr[i:i + self.T + 1] for i in ix]).astype(np.int64))
        x, y = xy[:, :-1], xy[:, 1:]
        if self.device.type == 'cuda':
            return (x.pin_memory().to(self.device, non_blocking=True),
                    y.pin_memory().to(self.device, non_blocking=True))
        return x.to(self.device), y.to(self.device)

    def get_batch(self, step, batch_size=None, seed=None):
        seed = self.seed if seed is None else seed
        return self._sample('train', batch_size or self.batch_size, _gen(TRAIN, seed, step))

    def fixed_batch(self, split, n, seed):
        return self._sample(split, n, _gen(FIXED, seed))

    def random_batches(self, n_batches, batch_size, seed):
        gen = _gen(RANDOM, seed)
        return [self._sample('train', batch_size, gen) for _ in range(n_batches)]


class ImageData:
    kind = 'image'
    num_classes = 10
    MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)

    def __init__(self, cfg, device):
        d = data_dir(cfg.dataset)
        load = lambda f: torch.from_numpy(np.load(d / f)).to(device)
        self.splits = {'train': (load('train_x.npy'), load('train_y.npy').long()),
                       'val': (load('test_x.npy'), load('test_y.npy').long())}
        self.batch_size, self.seed, self.device = cfg.train.batch_size, cfg.train.data_seed, device
        self.mean, self.std = self.MEAN.to(device), self.STD.to(device)
        self._perm = (None, None)

    def _take(self, split, idx):
        x, y = self.splits[split]
        idx = idx.to(self.device)
        x = x[idx].permute(0, 3, 1, 2).float().div_(255)
        return (x - self.mean) / self.std, y[idx]

    def get_batch(self, step, batch_size=None, seed=None):
        """Epochs of shuffled data without replacement (the last incomplete batch is dropped)."""
        seed = self.seed if seed is None else seed
        bs = batch_size or self.batch_size
        n = len(self.splits['train'][1])
        per_epoch = n // bs
        epoch, i = divmod(step, per_epoch)
        if self._perm[0] != (seed, epoch, n):
            self._perm = ((seed, epoch, n), torch.randperm(n, generator=_gen(TRAIN, seed, epoch)))
        return self._take('train', self._perm[1][i * bs:(i + 1) * bs])

    def fixed_batch(self, split, n, seed):
        n_split = len(self.splits[split][1])
        return self._take(split, torch.randperm(n_split, generator=_gen(FIXED, seed))[:n])

    def random_batches(self, n_batches, batch_size, seed):
        gen = _gen(RANDOM, seed)
        n = len(self.splits['train'][1])
        return [self._take('train', torch.randint(n, (batch_size,), generator=gen)) for _ in range(n_batches)]


def build_data(cfg, device):
    device = torch.device(device)
    return ImageData(cfg, device) if cfg.dataset == 'cifar10' else LMData(cfg, device)
