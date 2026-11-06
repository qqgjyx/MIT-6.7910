"""TinyStories (Eldan & Li, 2023), GPT-4 version, with a 4096-token byte-level BPE trained on it (as
llama2.c does), so that the embedding and head stay small next to the 10M/30M-parameter bodies and the
stored next-token distributions stay manageable. Stories are separated by an end-of-text token."""

import json

import numpy as np
import requests
from tokenizers import ByteLevelBPETokenizer

from optsplit.config import data_dir

URL = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-{}.txt'
VOCAB, EOT, TOKENIZER_CHARS = 4096, '<|endoftext|>', 200_000_000

out = data_dir('tinystories')
out.mkdir(parents=True, exist_ok=True)


def download(split):
    path = out / f'{split}.txt'
    if not path.exists():
        with requests.get(URL.format(split), stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(path.with_suffix('.part'), 'wb') as f:
                for chunk in r.iter_content(1 << 24):
                    f.write(chunk)
        path.with_suffix('.part').replace(path)
    return path


def stories(path):
    text = path.read_text(encoding='utf-8')
    return [s.strip() for s in text.split(EOT) if s.strip()]


train, val = stories(download('train')), stories(download('valid'))
tok = ByteLevelBPETokenizer()
sample, n_chars = [], 0
for s in train:
    sample.append(s)
    n_chars += len(s)
    if n_chars > TOKENIZER_CHARS:
        break
tok.train_from_iterator(sample, vocab_size=VOCAB, special_tokens=[EOT])
tok.save(str(out / 'tokenizer.json'))
eot = tok.token_to_id(EOT)

for split, docs in (('train', train), ('val', val)):
    with open(out / f'{split}.bin', 'wb') as f:
        n = 0
        for i in range(0, len(docs), 100_000):
            ids = [t for enc in tok.encode_batch(docs[i:i + 100_000]) for t in enc.ids + [eot]]
            np.array(ids, dtype=np.uint16).tofile(f)
            n += len(ids)
    print(f'{split}: {len(docs)} stories, {n} tokens')
(out / 'meta.json').write_text(json.dumps({'vocab_size': VOCAB, 'eot': eot}))
