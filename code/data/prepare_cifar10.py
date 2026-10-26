"""CIFAR-10 as uint8 arrays (N, 32, 32, 3) and int64 labels."""

import shutil

import numpy as np
from torchvision.datasets import CIFAR10

from optsplit.config import data_dir

out = data_dir('cifar10')
out.mkdir(parents=True, exist_ok=True)
for split, train in (('train', True), ('test', False)):
    ds = CIFAR10(out / 'raw', train=train, download=True)
    np.save(out / f'{split}_x.npy', ds.data)
    np.save(out / f'{split}_y.npy', np.array(ds.targets, dtype=np.int64))
    print(split, ds.data.shape)
shutil.rmtree(out / 'raw')
