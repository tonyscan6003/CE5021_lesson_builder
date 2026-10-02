"""A tiny fully connected network in NumPy, with backpropagation by hand.

Used where a lesson's claim is about training behaviour and can be shown by
actually training something small rather than by drawing the expected curve:
degradation and gradient flow in very deep plain vs residual networks (2.3),
and pre-training vs training from scratch (2.4). It runs in seconds on the
sklearn digits set and needs no deep learning framework.

The backward pass is checked against finite differences in `gradcheck()`.
"""
from __future__ import annotations

import numpy as np


def _act(name):
    if name == "relu":
        return (lambda z: np.maximum(z, 0.0)), (lambda z, a: (z > 0).astype(z.dtype))
    if name == "tanh":
        return np.tanh, (lambda z, a: 1.0 - a * a)
    raise ValueError(name)


class MLP:
    """in -> [hidden]*depth -> out, softmax cross-entropy.

    residual=True adds an identity shortcut around every hidden layer after
    the first (all hidden layers have the same width), i.e. h <- h + f(W h + b),
    the fully connected analogue of a ResNet basic block.
    """

    def __init__(self, n_in, width, depth, n_out, *, act="relu", residual=False,
                 init_scale=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.act_name, self.residual = act, residual
        self.f, self.df = _act(act)
        gain = np.sqrt(2.0) if act == "relu" else 1.0
        dims = [n_in] + [width] * depth + [n_out]
        self.W, self.b = [], []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
            s = gain * init_scale / np.sqrt(a)
            # residual branches start small so each block begins near identity
            if residual and 0 < i < depth:
                s *= 0.5 / np.sqrt(depth)
            self.W.append(rng.normal(0, s, (a, b)))
            self.b.append(np.zeros(b))

    # -- forward / backward --------------------------------------------------
    def forward(self, X):
        cache, h = [], X
        L = len(self.W)
        for i in range(L):
            z = h @ self.W[i] + self.b[i]
            if i == L - 1:
                cache.append((h, z, None))
                return z, cache
            a = self.f(z)
            out = h + a if (self.residual and 0 < i < L - 1 and h.shape == a.shape) else a
            cache.append((h, z, a))
            h = out

    def loss_grad(self, X, y):
        logits, cache = self.forward(X)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        g = p
        g[np.arange(n), y] -= 1.0
        g /= n
        gW, gb = [None] * len(self.W), [None] * len(self.W)
        L = len(self.W)
        for i in range(L - 1, -1, -1):
            h, z, a = cache[i]
            if i == L - 1:
                dz = g
            else:
                dz = g * self.df(z, a)
            gW[i], gb[i] = h.T @ dz, dz.sum(0)
            gh = dz @ self.W[i].T
            if self.residual and 0 < i < L - 1 and h.shape == a.shape:
                gh = gh + g          # the shortcut passes the gradient straight back
            g = gh
        return loss, gW, gb

    def predict(self, X):
        return self.forward(X)[0].argmax(1)

    def accuracy(self, X, y):
        return float((self.predict(X) == y).mean())


def train(net, X, y, Xte=None, yte=None, *, epochs=60, lr=0.05, momentum=0.9,
          batch=64, freeze=(), seed=0, record_every=1):
    """Mini-batch SGD with momentum. `freeze` lists layer indices not updated."""
    rng = np.random.default_rng(seed)
    vW = [np.zeros_like(w) for w in net.W]
    vb = [np.zeros_like(b) for b in net.b]
    hist = dict(epoch=[], train_loss=[], test_acc=[])
    n = len(y)
    for ep in range(1, epochs + 1):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            j = idx[s:s + batch]
            _, gW, gb = net.loss_grad(X[j], y[j])
            for i in range(len(net.W)):
                if i in freeze:
                    continue
                vW[i] = momentum * vW[i] - lr * gW[i]
                vb[i] = momentum * vb[i] - lr * gb[i]
                net.W[i] += vW[i]
                net.b[i] += vb[i]
        if ep % record_every == 0 or ep == epochs:
            hist["epoch"].append(ep)
            hist["train_loss"].append(net.loss_grad(X, y)[0])
            if Xte is not None:
                hist["test_acc"].append(net.accuracy(Xte, yte))
    return hist


def layer_grad_norms(net, X, y):
    """Norm of dL/dW for every layer, input side first."""
    _, gW, _ = net.loss_grad(X, y)
    return np.array([np.linalg.norm(g) for g in gW])


def gradcheck(seed=0, eps=1e-6):
    """Compare analytic gradients with central differences on a small net."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(7, 5))
    y = rng.integers(0, 3, 7)
    worst = 0.0
    for residual in (False, True):
        for act in ("tanh", "relu"):
            net = MLP(5, 6, 3, 3, act=act, residual=residual, seed=seed)
            _, gW, _ = net.loss_grad(X, y)
            for li in range(len(net.W)):
                for _ in range(4):
                    r, c = rng.integers(0, net.W[li].shape[0]), rng.integers(0, net.W[li].shape[1])
                    old = net.W[li][r, c]
                    net.W[li][r, c] = old + eps
                    lp = net.loss_grad(X, y)[0]
                    net.W[li][r, c] = old - eps
                    lm = net.loss_grad(X, y)[0]
                    net.W[li][r, c] = old
                    num = (lp - lm) / (2 * eps)
                    worst = max(worst, abs(num - gW[li][r, c]) / max(1e-8, abs(num) + abs(gW[li][r, c])))
    assert worst < 1e-5, f"backprop disagrees with finite differences (rel err {worst:.2e})"
    return worst


def digits(split_seed=0, test_frac=0.3):
    """sklearn digits, scaled to [0, 1], split into train and test."""
    from sklearn.datasets import load_digits
    d = load_digits()
    X, y = d.data / 16.0, d.target
    rng = np.random.default_rng(split_seed)
    idx = rng.permutation(len(y))
    nt = int(len(y) * test_frac)
    return X[idx[nt:]], y[idx[nt:]], X[idx[:nt]], y[idx[:nt]]
