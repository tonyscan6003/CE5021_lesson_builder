"""The toy problem used for the learning-rate examples of Q&A 1.

A three-arm spiral in 2-D, classified by a small MLP. It trains on a CPU in a
few seconds, so every curve in the deck is produced by a real training run
with a real torch.optim optimiser and scheduler, using the same loop
structure as the Assignment 1 notebook (an inner loop over the batches of an
epoch, an outer loop over the epochs).

    train(make_opt, make_sched=None, step="epoch")

`make_opt(params)` returns an optimiser; `make_sched(opt)` returns a
scheduler or None; `step` says where scheduler.step() is called: "epoch"
(after the inner loop), "batch" (after every optimizer.step()), or
"plateau" (after the inner loop, with the validation loss).
"""
from __future__ import annotations

import numpy as np

N_PER_CLASS, N_VAL_PER_CLASS = 500, 150
BATCH = 32
EPOCHS = 40
SEED = 0


TURNS, NOISE, HIDDEN = 2.2, 0.2, 128


def spiral(n_per_class: int, seed: int, noise: float | None = None):
    """Three interleaved spiral arms, one per class."""
    rng = np.random.default_rng(seed)
    noise = NOISE if noise is None else noise
    xs, ys = [], []
    for k in range(3):
        r = np.linspace(0.05, 1.0, n_per_class)
        t = np.linspace(k * 2 * np.pi / 3, k * 2 * np.pi / 3 + TURNS * np.pi, n_per_class)
        t = t + rng.normal(0, noise, n_per_class)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(n_per_class, k))
    return np.concatenate(xs).astype(np.float32), np.concatenate(ys).astype(np.int64)


def data():
    """Training and validation sets (different noise draws of the spiral)."""
    return spiral(N_PER_CLASS, SEED), spiral(N_VAL_PER_CLASS, SEED + 1)


def model(seed=SEED):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, HIDDEN), nn.ReLU(),
                         nn.Linear(HIDDEN, 3))


def train(make_opt, make_sched=None, step="epoch", epochs=EPOCHS, seed=SEED):
    """One training run. Returns per-batch lr and loss, and per-epoch
    train / validation loss and accuracy, as numpy arrays."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    (xt, yt), (xv, yv) = data()
    g = torch.Generator().manual_seed(seed)
    train_dataloader = DataLoader(TensorDataset(torch.tensor(xt), torch.tensor(yt)),
                                  batch_size=BATCH, shuffle=True, generator=g)
    Xv, Yv = torch.tensor(xv), torch.tensor(yv)
    net = model(seed)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_opt(net.parameters())
    scheduler = make_sched(optimizer) if make_sched else None
    lr_b, loss_b, tr_loss, tr_acc, va_loss, va_acc = [], [], [], [], [], []
    for epoch in range(epochs):
        net.train()
        run_loss, run_acc, n = 0.0, 0.0, 0
        for inputs, labels in train_dataloader:
            lr_b.append(optimizer.param_groups[0]["lr"])
            outputs = net(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and step == "batch":
                scheduler.step()
            lv = loss.item()
            loss_b.append(lv if np.isfinite(lv) else np.nan)
            run_loss += lv * len(labels)
            run_acc += (outputs.argmax(1) == labels).float().sum().item()
            n += len(labels)
        net.eval()
        with torch.no_grad():
            vo = net(Xv)
            vloss = loss_fn(vo, Yv).item()
            vacc = (vo.argmax(1) == Yv).float().mean().item()
        tr_loss.append(run_loss / n); tr_acc.append(run_acc / n)
        va_loss.append(vloss); va_acc.append(vacc)
        if scheduler is not None and step == "epoch":
            scheduler.step()
        elif scheduler is not None and step == "plateau":
            scheduler.step(vloss)
    f = lambda v: np.array(v, dtype=np.float64)
    return dict(lr=f(lr_b), loss_b=f(loss_b), tr_loss=f(tr_loss), tr_acc=f(tr_acc),
                va_loss=f(va_loss), va_acc=f(va_acc), batches=len(train_dataloader))


def lr_range_test(lo=1e-4, hi=10.0, steps=None):
    """Smith's learning-rate range test: one pass over the data (repeated if
    needed) with the learning rate increased exponentially on every batch.
    Returns the lr and the smoothed loss of each batch."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    (xt, yt), _ = data()
    g = torch.Generator().manual_seed(SEED)
    dl = DataLoader(TensorDataset(torch.tensor(xt), torch.tensor(yt)), batch_size=BATCH,
                    shuffle=True, generator=g)
    steps = steps or 4 * len(dl)
    net = model()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(net.parameters(), lr=lo, momentum=0.9)
    gamma = (hi / lo) ** (1 / (steps - 1))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma)
    lrs, losses, avg = [], [], 0.0
    it = iter(dl)
    for s in range(steps):
        try:
            inputs, labels = next(it)
        except StopIteration:
            it = iter(dl); inputs, labels = next(it)
        lrs.append(opt.param_groups[0]["lr"])
        loss = loss_fn(net(inputs), labels)
        loss.backward(); opt.step(); opt.zero_grad(); sched.step()
        lv = loss.item()
        avg = 0.9 * avg + 0.1 * lv                     # smoothed, with bias correction
        losses.append(avg / (1 - 0.9 ** (s + 1)))
        if not np.isfinite(lv) or (s > 10 and avg > 4 * min(losses)):
            break
    return np.array(lrs), np.array(losses)
