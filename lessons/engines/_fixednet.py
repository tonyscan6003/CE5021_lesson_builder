"""A fixed, hand-built convolutional feature network with an exact input gradient.

Lesson 2.5's visualisation techniques — reconstruction (inversion), activation
maximisation, style transfer and deep dream — are all *optimisations over the
input image*, driven by the gradient of some function of a network's features
with respect to the pixels. To run those algorithms for real without a deep
learning framework, this module provides a small network whose gradient is
written out by hand:

    features at depth d  =  avgpool_{2^d}( ReLU( conv(image, Gabor bank) ) )

The first layer is a bank of oriented Gabor filters at two scales — the kind
of filter the first layer of a trained CNN ends up learning — and "depth" here
is repeated pooling, which is what makes deeper features coarser and lets
them see a larger part of the image. It is NOT a trained network, and the
lesson's captions say so: it shows how each algorithm works and how depth
changes what survives, not what a real CNN's deep layers have learned.

`gradcheck()` compares the analytic gradient with finite differences.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve
from scipy.optimize import minimize


def gabor_bank(size=9, n_orient=4, wavelengths=(4.0, 7.0), sigma_frac=0.45):
    """Zero-mean, unit-norm Gabor filters: n_orient orientations x each wavelength,
    in even (cosine) and odd (sine) phase."""
    r = size // 2
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(float)
    bank = []
    for lam in wavelengths:
        sig = sigma_frac * lam
        for k in range(n_orient):
            th = np.pi * k / n_orient
            xr = xx * np.cos(th) + yy * np.sin(th)
            env = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
            for phase in (0.0, np.pi / 2):
                g = env * np.cos(2 * np.pi * xr / lam + phase)
                g -= g.mean()
                bank.append(g / np.linalg.norm(g))
    return np.array(bank)


BANK = gabor_bank()


def _pool(a, p):
    if p == 1:
        return a
    F, H, W = a.shape
    return a[:, : H - H % p, : W - W % p].reshape(F, H // p, p, W // p, p).mean((2, 4))


def _unpool(g, p, shape):
    if p == 1:
        return g
    out = np.zeros(shape)
    up = np.repeat(np.repeat(g, p, 1), p, 2) / (p * p)
    out[:, : up.shape[1], : up.shape[2]] = up
    return out


def _correlate_all(x, bank):
    """Zero-padded 'same' correlation of every channel with every filter, at once.
    Correlation is convolution with the flipped kernel; FFT makes it one call."""
    # mode="same" crops every axis to the first argument's size, including the
    # un-convolved filter axis, so the image must already span all the filters
    xb = np.broadcast_to(x[:, None], (x.shape[0], len(bank)) + x.shape[1:])
    z = fftconvolve(xb, bank[None, :, ::-1, ::-1], mode="same", axes=(-2, -1))
    return z.reshape(-1, *x.shape[1:])


def features(img, depth, bank=BANK):
    """img: (H, W) or (C, H, W). Returns pooled responses (F*C, h, w) and a cache."""
    x = img[None] if img.ndim == 2 else img
    z = _correlate_all(x, bank)
    a = np.maximum(z, 0.0)
    return _pool(a, 2 ** depth), (img.shape, z, depth)


def backward(g, cache, bank=BANK):
    """Gradient w.r.t. the image of <g, features(img)>."""
    shape, z, depth = cache
    ga = _unpool(g, 2 ** depth, z.shape) * (z > 0)
    nb = len(bank)
    C = 1 if len(shape) == 2 else shape[0]
    ga = ga.reshape(C, nb, *ga.shape[1:])
    # the adjoint of zero-padded correlation is zero-padded convolution
    gx = fftconvolve(ga, bank[None], mode="same", axes=(-2, -1)).sum(1)
    return gx[0] if len(shape) == 2 else gx


def gram(phi):
    """Style statistics: channel-by-channel inner products of the feature maps."""
    F = phi.reshape(phi.shape[0], -1)
    return F @ F.T / F.shape[1]


def tv(x):
    """Total variation (anisotropic, squared) and its gradient — keeps images smooth."""
    dx = np.diff(x, axis=-1)
    dy = np.diff(x, axis=-2)
    val = (dx ** 2).sum() + (dy ** 2).sum()
    g = np.zeros_like(x)
    g[..., :, :-1] -= 2 * dx
    g[..., :, 1:] += 2 * dx
    g[..., :-1, :] -= 2 * dy
    g[..., 1:, :] += 2 * dy
    return val, g


def optimise(f_and_g, x0, iters=200, bounds=(0.0, 1.0)):
    """L-BFGS on the pixels. Unconstrained, then clipped to the valid range: with
    tens of thousands of pixels, L-BFGS-B's box constraints cost more than the
    network does, and the losses here keep pixels in range in practice."""
    shape = x0.shape

    def fun(v):
        val, g = f_and_g(v.reshape(shape))
        return val, g.ravel()

    res = minimize(fun, x0.ravel(), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=iters))
    return np.clip(res.x.reshape(shape), *bounds), res


# -- the four techniques --------------------------------------------------------

def reconstruct(target, depth, *, iters=250, tv_weight=0.002, seed=0):
    """Find an image whose depth-d features match the target's (inversion)."""
    phi0, _ = features(target, depth)
    scale = (phi0 ** 2).sum()

    def fg(x):
        phi, cache = features(x, depth)
        d = phi - phi0
        t, gt = tv(x)
        return ((d ** 2).sum() / scale + tv_weight * t,
                backward(2 * d / scale, cache) + tv_weight * gt)

    x0 = np.random.default_rng(seed).uniform(0.3, 0.7, target.shape)
    return optimise(fg, x0, iters)[0]


def activation_max(size, depth, filt, *, iters=200, l2=0.02, seed=0):
    """Image that maximises one unit — the centre of one feature map — at depth d."""
    x0 = 0.5 + np.random.default_rng(seed).normal(0, 0.02, (size, size))
    phi, _ = features(x0, depth)
    cy, cx = phi.shape[1] // 2, phi.shape[2] // 2

    def fg(x):
        phi, cache = features(x, depth)
        g = np.zeros_like(phi)
        g[filt, cy, cx] = -1.0                     # maximise => minimise the negative
        c = x - 0.5
        return -phi[filt, cy, cx] + l2 * (c ** 2).sum(), backward(g, cache) + 2 * l2 * c

    return optimise(fg, x0, iters)[0]


def style_transfer(content, style, *, content_depth=2, style_depths=(0, 1),
                   alpha=1.0, beta=4.0, iters=300, tv_weight=0.001, seed=0):
    """Gatys-style: match content features at one depth and Gram matrices at others."""
    pc, _ = features(content, content_depth)
    pcs = (pc ** 2).sum()
    gs = {d: gram(features(style, d)[0]) for d in style_depths}
    gss = {d: (g ** 2).sum() for d, g in gs.items()}

    def fg(x):
        total, grad = 0.0, np.zeros_like(x)
        phi, cache = features(x, content_depth)
        d = phi - pc
        total += alpha * (d ** 2).sum() / pcs
        grad += backward(alpha * 2 * d / pcs, cache)
        for dep, G0 in gs.items():
            phi, cache = features(x, dep)
            Fm = phi.reshape(phi.shape[0], -1)
            G = Fm @ Fm.T / Fm.shape[1]
            dG = G - G0
            total += beta * (dG ** 2).sum() / gss[dep]
            gF = (beta * 4.0 / gss[dep]) * (dG @ Fm) / Fm.shape[1]
            grad += backward(gF.reshape(phi.shape), cache)
        t, gt = tv(x)
        return total + tv_weight * t, grad + tv_weight * gt

    x0 = np.clip(content + np.random.default_rng(seed).normal(0, 0.1, content.shape), 0, 1)
    return optimise(fg, x0, iters)[0]


def deep_dream(img, depth, *, steps=25, step=0.04):
    """Gradient ascent on the energy of a layer's features, starting from a photo."""
    x = img.copy()
    for _ in range(steps):
        phi, cache = features(x, depth)
        g = backward(phi / max(np.abs(phi).max(), 1e-9), cache)
        x = np.clip(x + step * g / (np.abs(g).mean() + 1e-9) * 0.05, 0, 1)
    return x


def gradcheck(seed=0, eps=1e-5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, (2, 14, 14))
    worst = 0.0
    for depth in (0, 1, 2):
        phi, cache = features(x, depth)
        w = rng.normal(size=phi.shape)
        gx = backward(w, cache)
        for _ in range(6):
            c, i, j = rng.integers(0, 2), rng.integers(2, 12), rng.integers(2, 12)
            xp, xm = x.copy(), x.copy()
            xp[c, i, j] += eps
            xm[c, i, j] -= eps
            num = ((features(xp, depth)[0] - features(xm, depth)[0]) * w).sum() / (2 * eps)
            worst = max(worst, abs(num - gx[c, i, j]) / max(1e-8, abs(num) + abs(gx[c, i, j])))
    assert worst < 1e-4, f"fixednet gradient disagrees with finite differences ({worst:.2e})"
    return worst
