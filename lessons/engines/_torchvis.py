"""Feature visualisation with a pre-trained VGG-16 (PyTorch), for lesson 2.5.

All four techniques keep the network FIXED and optimise the INPUT image:

    reconstruct      inversion: match one layer's features of a target image
    activation_max   maximise one unit, or one whole channel
    style_transfer   content loss on a deep layer + Gram-matrix style loss
                     (with content weight 0 it is texture synthesis)
    deep_dream       amplify whatever a layer already sees in a photograph

The network is torchvision's VGG-16 with its ImageNet weights, the model the
lecture slides use for these visualisations. It runs on the GPU when one is
available and on the CPU otherwise.

Results are cached in data/_torch/<name>.npz together with a hash of the
settings that produced them (`cached`). A build reuses the stored images and
only recomputes when the settings — or VERSION, bumped when the algorithms
change — differ. Set CE5021_RECOMPUTE=1 to force a recompute.
"""
from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from ._common import DATA_DIR

CACHE = DATA_DIR / "_torch"
VERSION = 2

# ReLU outputs of VGG-16's convolution layers, as indices into model.features
LAYERS = {"conv1_1": 1, "conv1_2": 3, "conv2_1": 6, "conv2_2": 8, "conv3_1": 11,
          "conv3_3": 15, "conv4_1": 18, "conv4_2": 20, "conv4_3": 22,
          "conv5_1": 25, "conv5_3": 29}
# theoretical receptive field of one unit of each layer, in input pixels
RF = {"conv1_1": 3, "conv1_2": 5, "conv2_1": 10, "conv2_2": 14, "conv3_1": 24,
      "conv3_3": 40, "conv4_1": 60, "conv4_2": 76, "conv4_3": 92,
      "conv5_1": 132, "conv5_3": 196}


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------

def cached(name: str, config: dict, compute) -> dict:
    """Return compute()'s arrays, from data/_torch/<name>.npz when up to date."""
    key = hashlib.sha1(json.dumps([VERSION, config], sort_keys=True).encode()).hexdigest()[:16]
    path = CACHE / f"{name}.npz"
    if path.exists() and not os.environ.get("CE5021_RECOMPUTE"):
        with np.load(path, allow_pickle=False) as z:
            if str(z["_key"]) == key:
                return {k: z[k] for k in z.files if k != "_key"}
    t0 = time.time()
    out = compute()
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, _key=np.array(key), **out)
    print(f"      [torch] computed {name} on {device()} in {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------------------
# network
# ---------------------------------------------------------------------------

_NET = None


def device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def vgg():
    """VGG-16's convolutional part, ImageNet weights, frozen, on the device."""
    global _NET
    if _NET is None:
        import torch
        from torchvision import models
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        net = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        for p in net.parameters():
            p.requires_grad_(False)
        _NET = net.eval().to(device())
    return _NET


def _norm(x):
    import torch
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def features(x, names, pre_relu=False):
    """Activations of the named layers for an image batch x in [0, 1].

    With `pre_relu=True` the convolution output BEFORE its ReLU is returned:
    activation maximisation needs it, because a unit that starts inactive has
    a ReLU output of zero and so no gradient to climb."""
    want = {LAYERS[n] - (1 if pre_relu else 0): n for n in names}
    out, h = {}, _norm(x)
    for i, layer in enumerate(vgg()):
        h = layer(h)
        if i in want:
            out[want[i]] = h
            if len(out) == len(want):
                break
    return out


def to_tensor(img: np.ndarray):
    import torch
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None].contiguous()
    return x.to(device())


def to_uint8(x) -> np.ndarray:
    return (x.detach().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def tv(x):
    """Total variation: penalises pixel-to-pixel noise, favouring natural images."""
    return ((x[..., 1:, :] - x[..., :-1, :]) ** 2).mean() + \
        ((x[..., :, 1:] - x[..., :, :-1]) ** 2).mean()


def _lbfgs(param, closure, iters):
    import torch
    opt = torch.optim.LBFGS([param], max_iter=iters, history_size=50,
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12)

    def step():
        opt.zero_grad()
        loss = closure()
        loss.backward()
        return loss
    opt.step(step)


# ---------------------------------------------------------------------------
# the four techniques
# ---------------------------------------------------------------------------

def reconstruct(img: np.ndarray, layer: str, iters=400, tv_weight=2.0, seed=0) -> np.ndarray:
    """Inversion (Mahendran and Vedaldi): find an image whose `layer` features
    match those of `img`, starting from random noise. The pixels are
    parameterised through a sigmoid so they stay in range."""
    import torch
    torch.manual_seed(seed)
    target = features(to_tensor(img), [layer])[layer].detach()
    z = (0.5 * torch.randn_like(to_tensor(img))).requires_grad_(True)

    def closure():
        x = torch.sigmoid(z)
        f = features(x, [layer])[layer]
        return ((f - target) ** 2).sum() / (target ** 2).sum() + tv_weight * tv(x)
    _lbfgs(z, closure, iters)
    return to_uint8(torch.sigmoid(z))


# colour decorrelation (Olah et al., "Feature Visualization", Distill 2017):
# optimising in a decorrelated colour space and in Fourier space with a 1/f
# scaling gives far more natural images than optimising raw pixels.
_COLOR = np.array([[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]],
                  np.float32)
_COLOR /= np.max(np.linalg.norm(_COLOR, axis=0))


def _fourier_image(size, seed):
    import torch
    g = torch.Generator().manual_seed(seed)
    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.rfftfreq(size)[None, :]
    scale = 1.0 / np.maximum(np.sqrt(fx ** 2 + fy ** 2), 1.0 / size)
    scale = torch.tensor(scale * size / scale.max() * 0.4, dtype=torch.float32, device=device())
    spec = (0.01 * torch.randn(3, size, size // 2 + 1, 2, generator=g)).to(device())
    spec.requires_grad_(True)
    colour = torch.tensor(_COLOR, device=device())

    def image():
        s = torch.view_as_complex(spec) * scale
        x = torch.fft.irfft2(s, s=(size, size), norm="ortho") / 4.0
        x = torch.einsum("chw,dc->dhw", x, colour)
        return torch.sigmoid(x)[None]
    return spec, image


def _jitter(x, rng):
    """Random translation and scale before each step: patterns that survive
    small transformations are the ones the unit genuinely prefers."""
    import torch
    import torch.nn.functional as F
    dy, dx = rng.integers(-8, 9, 2)
    x = torch.roll(x, (int(dy), int(dx)), (2, 3))
    s = float(rng.choice([0.95, 0.975, 1.0, 1.025, 1.05]))
    if s != 1.0:
        n = x.shape[-1]
        x = F.interpolate(x, scale_factor=s, mode="bilinear", align_corners=False)
        m = x.shape[-1]
        x = x[..., (m - n) // 2:(m - n) // 2 + n, (m - n) // 2:(m - n) // 2 + n] if m >= n \
            else F.pad(x, ((n - m) // 2, n - m - (n - m) // 2, (n - m) // 2, n - m - (n - m) // 2),
                       mode="reflect")
    return x


def activation_max(layer: str, channel: int, unit: bool, size=224, steps=384,
                   lr=0.05, seed=0) -> np.ndarray:
    """Gradient ASCENT on one channel of `layer` (its pre-ReLU output).

    unit=False: the channel's mean over the whole map. The image is optimised
      in a decorrelated Fourier space with random jitter, which gives natural
      looking patterns that fill the image.
    unit=True: the centre unit only — one point of the feature map. Here the
      pixels themselves are optimised (with a small smoothness penalty): a
      Fourier coefficient touches every pixel, which would spread colour far
      outside the unit's receptive field, whereas pixels outside the receptive
      field receive no gradient at all and stay at the grey start.
    """
    import torch
    rng = np.random.default_rng(seed)
    if unit:
        torch.manual_seed(seed)
        z = (0.01 * torch.randn(1, 3, size, size, device=device())).requires_grad_(True)
        image = lambda: torch.sigmoid(z)
        param, j = z, 4
    else:
        param, image = _fourier_image(size, seed)
        j = None
    opt = torch.optim.Adam([param], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        x = image()
        xs = torch.roll(x, tuple(int(v) for v in rng.integers(-j, j + 1, 2)), (2, 3)) \
            if unit else _jitter(x, rng)
        f = features(xs, [layer], pre_relu=True)[layer][0, channel]
        if unit:
            loss = -f[f.shape[0] // 2, f.shape[1] // 2] + 20.0 * tv(x)
        else:
            loss = -f.mean()
        loss.backward()
        opt.step()
    return to_uint8(image())


def gram(f):
    """Gram matrix of a (1, C, H, W) activation: inner products of every pair
    of feature maps, summed over positions and normalised."""
    _, c, h, w = f.shape
    f = f.view(c, h * w)
    return f @ f.t() / (c * h * w)


STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
CONTENT_LAYER = "conv4_2"


def style_transfer(content: np.ndarray | None, style: np.ndarray, size=(256, 320),
                   content_weight=1.0, style_weight=1e5, iters=500, tv_weight=1.0,
                   seed=0) -> np.ndarray:
    """Gatys et al.: start from random noise and minimise a content loss on a
    deep layer plus a style loss on the Gram matrices of five layers. With
    `content=None` only the style term remains: texture synthesis."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    s_img = to_tensor(style)
    gs = {k: gram(v).detach() for k, v in features(s_img, STYLE_LAYERS).items()}
    if content is not None:
        c_img = to_tensor(content)
        size = tuple(c_img.shape[-2:])
        ct = features(c_img, [CONTENT_LAYER])[CONTENT_LAYER].detach()
    z = (0.5 * torch.randn(1, 3, *size, device=device())).requires_grad_(True)
    names = STYLE_LAYERS + ([CONTENT_LAYER] if content is not None else [])

    def closure():
        x = torch.sigmoid(z)
        f = features(x, names)
        loss = style_weight * sum(F.mse_loss(gram(f[k]), gs[k]) for k in STYLE_LAYERS) / 5
        if content is not None:
            loss = loss + content_weight * F.mse_loss(f[CONTENT_LAYER], ct)
        return loss + tv_weight * tv(x)
    _lbfgs(z, closure, iters)
    return to_uint8(torch.sigmoid(z))


def deep_dream(img: np.ndarray, layer: str, octaves=4, octave_scale=1.4, steps=24,
               lr=0.012, seed=0) -> np.ndarray:
    """Gradient ascent on the energy of `layer`'s activations, starting from
    the photograph itself, over several scales (octaves) so that features of
    every size are amplified; the detail added at a coarse scale is carried
    up to the next."""
    import torch
    import torch.nn.functional as F
    rng = np.random.default_rng(seed)
    base = to_tensor(img)
    h0, w0 = base.shape[-2:]
    shapes = [(int(h0 / octave_scale ** i), int(w0 / octave_scale ** i))
              for i in reversed(range(octaves))]
    detail = torch.zeros_like(F.interpolate(base, size=shapes[0], mode="bilinear"))
    for shp in shapes:
        b = F.interpolate(base, size=shp, mode="bilinear", align_corners=False)
        detail = F.interpolate(detail, size=shp, mode="bilinear", align_corners=False)
        x = (b + detail).detach().requires_grad_(True)
        for _ in range(steps):
            dy, dx = (int(v) for v in rng.integers(-16, 17, 2))
            xs = torch.roll(x, (dy, dx), (2, 3))
            loss = (features(xs, [layer])[layer] ** 2).mean()
            g, = torch.autograd.grad(loss, x)
            with torch.no_grad():
                x += lr * g / (g.abs().mean() + 1e-8)
                x.clamp_(0, 1)
        detail = (x - b).detach()
    return to_uint8(x)
