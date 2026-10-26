"""Fixed held-out and training examples on which every run's predictions and losses are logged."""

import torch
import torch.nn.functional as F


class Probe:
    def __init__(self, cfg, data):
        p = cfg.probe
        self.kind = data.kind
        self.val = data.fixed_batch('val', p.n_seqs, seed=0)
        self.train = data.fixed_batch('train', p.train_seqs, seed=0)
        self.chunk = 32 if data.kind == 'lm' else 1000
        if data.kind == 'lm':
            T, k = cfg.model.block_size, p.positions_per_seq
            self.pos = torch.arange(T) if k == 0 else torch.linspace(0, T - 1, k).round().long()

    def targets(self):
        y = self.val[1]
        return (y[:, self.pos].reshape(-1) if self.kind == 'lm' else y).cpu()

    def _logits(self, model, x):
        return torch.cat([model(x[i:i + self.chunk]).float() for i in range(0, len(x), self.chunk)])

    def _loss_acc(self, logits, y):
        logits, y = logits.reshape(-1, logits.size(-1)), y.reshape(-1)
        return F.cross_entropy(logits, y).item(), (logits.argmax(-1) == y).float().mean().item()

    @torch.no_grad()
    def train_loss(self, model):
        return self._loss_acc(self._logits(model, self.train[0]), self.train[1])[0]

    @torch.no_grad()
    def evaluate(self, model):
        """-> (float16 log-probabilities on the stored validation positions, eval record)."""
        logits = self._logits(model, self.val[0])
        val_loss, val_acc = self._loss_acc(logits, self.val[1])
        if self.kind == 'lm':
            logits = logits[:, self.pos.to(logits.device)].reshape(-1, logits.size(-1))
        logp = F.log_softmax(logits, -1).half().cpu()
        return logp, dict(train_loss=self.train_loss(model), val_loss=val_loss, val_acc=val_acc)
