"""Character-level tiny Shakespeare, as in nanoGPT's data/shakespeare_char/prepare.py: 90/10 split,
uint16 ids in train.bin/val.bin, vocabulary in meta.json."""

import json

import numpy as np
import requests

from optsplit.config import data_dir

URL = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'

out = data_dir('shakespeare_char')
out.mkdir(parents=True, exist_ok=True)
text = requests.get(URL, timeout=60).text
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
ids = np.array([stoi[c] for c in text], dtype=np.uint16)
n = int(0.9 * len(ids))
ids[:n].tofile(out / 'train.bin')
ids[n:].tofile(out / 'val.bin')
(out / 'meta.json').write_text(json.dumps({'vocab_size': len(chars), 'itos': chars}))
print(f'{len(chars)} characters, {n} train and {len(ids) - n} val tokens in {out}')
