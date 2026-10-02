"""Architecture cost calculator: shapes, parameters and multiply-accumulates.

Every parameter count and operation count quoted for a published network in
these lessons comes from here, computed from the layer specification with the
same arithmetic the students are taught:

    output size     o = floor((i + 2p - k) / s) + 1                (lesson 1.1)
    conv params     k * k * (C_in / groups) * C_out  (+ C_out bias)  (lesson 1.3)
    conv MACs       params_without_bias * o_h * o_w                  (lesson 1.3)
    dense params    n_in * n_out (+ n_out bias)                      (lesson 1.2)
    batch norm      2 * C learnable (gamma, beta); running stats are not parameters

`verify()` checks the totals against the counts published for torchvision's
reference implementations, so a wrong formula fails the build instead of
quietly printing a wrong number on a page. MACs are conv and dense
multiply-accumulates only, the convention torchvision's GFLOPS column uses.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Layer:
    name: str
    kind: str            # input conv dwconv fc bn pool gap flatten concat add se
    out: tuple           # (C, H, W) for maps, (N,) for vectors
    params: int = 0
    macs: int = 0
    stage: str = ""      # used to aggregate per stage / block


@dataclass
class Net:
    name: str
    layers: list = field(default_factory=list)

    # -- shape bookkeeping is done by the builders; Net just records -----------
    def add(self, *a, **k):
        self.layers.append(Layer(*a, **k))
        return self.layers[-1].out

    @property
    def params(self):
        return sum(l.params for l in self.layers)

    @property
    def macs(self):
        return sum(l.macs for l in self.layers)

    def by_kind(self, *kinds):
        return [l for l in self.layers if l.kind in kinds]

    def stage_totals(self):
        out = {}
        for l in self.layers:
            p, m = out.get(l.stage, (0, 0))
            out[l.stage] = (p + l.params, m + l.macs)
        return out


def osize(i, k, s=1, p=0):
    """Lesson 1.1's general output-size relationship."""
    return (i + 2 * p - k) // s + 1


def conv(net, shape, cout, k, s=1, p=0, *, bias=False, groups=1, bn=False,
         name="conv", stage=""):
    cin, h, w = shape
    oh, ow = osize(h, k, s, p), osize(w, k, s, p)
    wts = k * k * (cin // groups) * cout
    kind = "dwconv" if groups == cin and groups > 1 else "conv"
    out = net.add(name, kind, (cout, oh, ow), wts + (cout if bias else 0),
                  wts * oh * ow, stage)
    if bn:
        net.add(name + " bn", "bn", out, 2 * cout, 0, stage)
    return out


def pool(net, shape, k, s, p=0, name="pool", stage=""):
    c, h, w = shape
    return net.add(name, "pool", (c, osize(h, k, s, p), osize(w, k, s, p)), 0, 0, stage)


def gap(net, shape, name="global avg pool", stage=""):
    return net.add(name, "gap", (shape[0],), 0, 0, stage)


def flatten(net, shape, name="flatten", stage=""):
    n = 1
    for d in shape:
        n *= d
    return net.add(name, "flatten", (n,), 0, 0, stage)


def fc(net, shape, nout, *, bias=True, name="fc", stage=""):
    nin = shape[0]
    return net.add(name, "fc", (nout,), nin * nout + (nout if bias else 0),
                   nin * nout, stage)


# ---------------------------------------------------------------------------
# Section 2.2: AlexNet, VGG-16, GoogLeNet / Inception
# ---------------------------------------------------------------------------

def alexnet(grouped=False):
    """AlexNet as the lesson draws it: 5 conv, 3 pool, 3 dense.

    Input is 227 x 227: the paper says 224, but 11x11 stride 4 with no padding
    only gives the paper's 55 x 55 from 227 — a well-known slip, and a nice
    exercise in lesson 1.1's formula. `grouped=True` reproduces the original
    two-GPU split, where conv2, conv4 and conv5 see only half the channels.
    """
    g = 2 if grouped else 1
    n = Net("AlexNet" + (" (2-GPU, grouped)" if grouped else ""))
    x = n.add("input 227×227×3", "input", (3, 227, 227))
    x = conv(n, x, 96, 11, 4, bias=True, name="conv1 11×11/4", stage="conv1")
    x = pool(n, x, 3, 2, name="max pool 3×3/2", stage="conv1")
    x = conv(n, x, 256, 5, 1, 2, bias=True, groups=g, name="conv2 5×5", stage="conv2")
    x = pool(n, x, 3, 2, name="max pool 3×3/2", stage="conv2")
    x = conv(n, x, 384, 3, 1, 1, bias=True, name="conv3 3×3", stage="conv3")
    x = conv(n, x, 384, 3, 1, 1, bias=True, groups=g, name="conv4 3×3", stage="conv4")
    x = conv(n, x, 256, 3, 1, 1, bias=True, groups=g, name="conv5 3×3", stage="conv5")
    x = pool(n, x, 3, 2, name="max pool 3×3/2", stage="conv5")
    x = flatten(n, x, stage="fc6")
    x = fc(n, x, 4096, name="fc6", stage="fc6")
    x = fc(n, x, 4096, name="fc7", stage="fc7")
    fc(n, x, 1000, name="fc8 + softmax", stage="fc8")
    return n


VGG16_CFG = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
             512, 512, 512, "M", 512, 512, 512, "M"]


def vgg16():
    n = Net("VGG-16")
    x = n.add("input 224×224×3", "input", (3, 224, 224))
    block, i = 1, 1
    for v in VGG16_CFG:
        if v == "M":
            x = pool(n, x, 2, 2, name="max pool 2×2/2", stage=f"block{block}")
            block, i = block + 1, 1
        else:
            x = conv(n, x, v, 3, 1, 1, bias=True, name=f"conv{block}_{i} 3×3",
                     stage=f"block{block}")
            i += 1
    x = flatten(n, x, stage="fc")
    x = fc(n, x, 4096, name="fc1", stage="fc")
    x = fc(n, x, 4096, name="fc2", stage="fc")
    fc(n, x, 1000, name="fc3 + softmax", stage="fc")
    return n


# (name, C_in, #1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool proj) — Szegedy
# et al. 2015, Table 1.
INCEPTION_CFG = [
    ("3a", 192, 64, 96, 128, 16, 32, 32),
    ("3b", 256, 128, 128, 192, 32, 96, 64),
    "M",
    ("4a", 480, 192, 96, 208, 16, 48, 64),
    ("4b", 512, 160, 112, 224, 24, 64, 64),
    ("4c", 512, 128, 128, 256, 24, 64, 64),
    ("4d", 512, 112, 144, 288, 32, 64, 64),
    ("4e", 528, 256, 160, 320, 32, 128, 128),
    "M",
    ("5a", 832, 256, 160, 320, 32, 128, 128),
    ("5b", 832, 384, 192, 384, 48, 128, 128),
]


def inception_module(net, shape, cfg, *, reduce=True, stage=None):
    """One inception module. With reduce=False it is the 'naive' version: no
    1x1 reductions before the 3x3 and 5x5, and the pool path is passed through
    unprojected — which is what makes the naive module's output grow."""
    name, cin, c1, r3, c3, r5, c5, pp = cfg
    stage = stage or f"inception {name}"
    assert shape[0] == cin, (name, shape, cin)
    b1 = conv(net, shape, c1, 1, bias=True, name=f"{name} 1×1", stage=stage)
    if reduce:
        t = conv(net, shape, r3, 1, bias=True, name=f"{name} 3×3 reduce", stage=stage)
        b3 = conv(net, t, c3, 3, 1, 1, bias=True, name=f"{name} 3×3", stage=stage)
        t = conv(net, shape, r5, 1, bias=True, name=f"{name} 5×5 reduce", stage=stage)
        b5 = conv(net, t, c5, 5, 1, 2, bias=True, name=f"{name} 5×5", stage=stage)
        t = pool(net, shape, 3, 1, 1, name=f"{name} max pool", stage=stage)
        bp = conv(net, t, pp, 1, bias=True, name=f"{name} pool proj", stage=stage)
    else:
        b3 = conv(net, shape, c3, 3, 1, 1, bias=True, name=f"{name} 3×3", stage=stage)
        b5 = conv(net, shape, c5, 5, 1, 2, bias=True, name=f"{name} 5×5", stage=stage)
        bp = pool(net, shape, 3, 1, 1, name=f"{name} max pool", stage=stage)
    cout = b1[0] + b3[0] + b5[0] + bp[0]
    return net.add(f"{name} concat", "concat", (cout,) + shape[1:], 0, 0, stage)


def googlenet():
    n = Net("GoogLeNet")
    x = n.add("input 224×224×3", "input", (3, 224, 224))
    x = conv(n, x, 64, 7, 2, 3, bias=True, name="conv 7×7/2", stage="stem")
    x = pool(n, x, 3, 2, 1, name="max pool 3×3/2", stage="stem")
    x = conv(n, x, 64, 1, bias=True, name="conv 1×1", stage="stem")
    x = conv(n, x, 192, 3, 1, 1, bias=True, name="conv 3×3", stage="stem")
    x = pool(n, x, 3, 2, 1, name="max pool 3×3/2", stage="stem")
    for cfg in INCEPTION_CFG:
        if cfg == "M":
            x = pool(n, x, 3, 2, 1, name="max pool 3×3/2", stage="pool")
        else:
            x = inception_module(n, x, cfg)
    x = gap(n, x, stage="head")
    fc(n, x, 1000, name="fc + softmax", stage="head")
    return n


# ---------------------------------------------------------------------------
# Section 2.3: ResNet, SE-ResNet, DenseNet
# ---------------------------------------------------------------------------

RESNET_CFG = {18: ("basic", [2, 2, 2, 2]), 34: ("basic", [3, 4, 6, 3]),
              50: ("bottleneck", [3, 4, 6, 3]), 101: ("bottleneck", [3, 4, 23, 3]),
              152: ("bottleneck", [3, 8, 36, 3])}


def _se(net, shape, r, stage, name):
    """Squeeze-excite: global average pool, FC C->C/r, ReLU, FC C/r->C, sigmoid,
    rescale. The rescale is elementwise and costs no parameters."""
    c, h, w = shape
    p = c * (c // r) + c // r + (c // r) * c + c
    net.add(f"{name} SE (r={r})", "se", shape, p, 2 * c * (c // r), stage)


def resnet(depth, se_ratio=None):
    kind, blocks = RESNET_CFG[depth]
    exp = 1 if kind == "basic" else 4
    n = Net(f"{'SE-' if se_ratio else ''}ResNet-{depth}")
    x = n.add("input 224×224×3", "input", (3, 224, 224))
    x = conv(n, x, 64, 7, 2, 3, bn=True, name="conv1 7×7/2", stage="stem")
    x = pool(n, x, 3, 2, 1, name="max pool 3×3/2", stage="stem")
    planes = 64
    for si, nb in enumerate(blocks):
        stage = f"conv{si + 2}_x"
        for b in range(nb):
            s = 2 if (b == 0 and si > 0) else 1
            inp = x
            tag = f"{stage}.{b}"
            if kind == "basic":
                x = conv(n, inp, planes, 3, s, 1, bn=True, name=f"{tag} 3×3", stage=stage)
                x = conv(n, x, planes, 3, 1, 1, bn=True, name=f"{tag} 3×3", stage=stage)
            else:
                x = conv(n, inp, planes, 1, bn=True, name=f"{tag} 1×1 reduce", stage=stage)
                x = conv(n, x, planes, 3, s, 1, bn=True, name=f"{tag} 3×3", stage=stage)
                x = conv(n, x, planes * exp, 1, bn=True, name=f"{tag} 1×1 expand",
                         stage=stage)
            if se_ratio:
                _se(n, x, se_ratio, stage, tag)
            if s != 1 or inp[0] != planes * exp:      # projection shortcut
                conv(n, inp, planes * exp, 1, s, bn=True, name=f"{tag} projection",
                     stage=stage)
            n.add(f"{tag} add", "add", x, 0, 0, stage)
        planes *= 2
    x = gap(n, x, stage="head")
    fc(n, x, 1000, name="fc + softmax", stage="head")
    return n


def densenet121(growth=32, blocks=(6, 12, 24, 16), bn_size=4, compression=0.5):
    n = Net("DenseNet-121")
    x = n.add("input 224×224×3", "input", (3, 224, 224))
    x = conv(n, x, 2 * growth, 7, 2, 3, bn=True, name="conv 7×7/2", stage="stem")
    x = pool(n, x, 3, 2, 1, name="max pool 3×3/2", stage="stem")
    c = x[0]
    for bi, nl in enumerate(blocks):
        stage = f"dense block {bi + 1}"
        for li in range(nl):
            # composite function BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3), pre-activation
            n.add(f"{stage}.{li} bn", "bn", (c,) + x[1:], 2 * c, 0, stage)
            t = conv(n, (c,) + x[1:], bn_size * growth, 1, bn=True,
                     name=f"{stage}.{li} 1×1", stage=stage)
            conv(n, t, growth, 3, 1, 1, name=f"{stage}.{li} 3×3", stage=stage)
            c += growth
            n.add(f"{stage}.{li} concat", "concat", (c,) + x[1:], 0, 0, stage)
        x = (c,) + x[1:]
        if bi < len(blocks) - 1:
            stage = f"transition {bi + 1}"
            n.add(f"{stage} bn", "bn", x, 2 * c, 0, stage)
            c = int(c * compression)
            x = conv(n, x, c, 1, name=f"{stage} 1×1", stage=stage)
            x = pool(n, x, 2, 2, name=f"{stage} avg pool", stage=stage)
    n.add("final bn", "bn", x, 2 * c, 0, "head")
    x = gap(n, x, stage="head")
    fc(n, x, 1000, name="fc + softmax", stage="head")
    return n


# ---------------------------------------------------------------------------
# Section 2.4 / 3.4: MobileNetV2 (depthwise separable, inverted residuals)
# ---------------------------------------------------------------------------

MOBILENET_V2_CFG = [(1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
                    (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]   # t, c, n, s


def mobilenet_v2(num_classes=1000):
    n = Net("MobileNetV2")
    x = n.add("input 224×224×3", "input", (3, 224, 224))
    x = conv(n, x, 32, 3, 2, 1, bn=True, name="conv 3×3/2", stage="features")
    for t, c, reps, s in MOBILENET_V2_CFG:
        for i in range(reps):
            stride = s if i == 0 else 1
            cin, hid = x[0], x[0] * t
            y = x
            if t != 1:
                y = conv(n, y, hid, 1, bn=True, name="expand 1×1", stage="features")
            y = conv(n, y, hid, 3, stride, 1, groups=hid, bn=True,
                     name="depthwise 3×3", stage="features")
            y = conv(n, y, c, 1, bn=True, name="project 1×1", stage="features")
            x = y
    x = conv(n, x, 1280, 1, bn=True, name="conv 1×1", stage="features")
    x = gap(n, x, stage="features")
    fc(n, x, num_classes, name="classifier", stage="classifier")
    return n


# ---------------------------------------------------------------------------
# Verification against published reference counts
# ---------------------------------------------------------------------------

# torchvision.models documentation: "Params" exactly, "GFLOPS" to 2 d.p.
REFERENCE = {
    "VGG-16":       (vgg16,                    138_357_544, 15.47),
    "ResNet-18":    (lambda: resnet(18),        11_689_512,  1.81),
    "ResNet-34":    (lambda: resnet(34),        21_797_672,  3.66),
    "ResNet-50":    (lambda: resnet(50),        25_557_032,  4.09),
    "ResNet-101":   (lambda: resnet(101),       44_549_160,  7.80),
    "ResNet-152":   (lambda: resnet(152),       60_192_808, 11.51),
    "DenseNet-121": (densenet121,                7_978_856,  2.83),
    "MobileNetV2":  (mobilenet_v2,               3_504_872,  0.30),
    # Not in torchvision, checked against the original papers instead:
    # Krizhevsky et al. 2012 (two-GPU, grouped) and Hu et al. 2018 (r = 16).
    "AlexNet (2-GPU)": (lambda: alexnet(grouped=True), 60_965_224, 0.724),
    "SE-ResNet-50":    (lambda: resnet(50, se_ratio=16), 28_088_024, 4.09),
}


def verify():
    """Parameters must match exactly; MACs to within 1.5%."""
    for name, (fn, p_ref, g_ref) in REFERENCE.items():
        net = fn()
        assert net.params == p_ref, f"{name}: {net.params:,} params, expected {p_ref:,}"
        g = net.macs / 1e9
        assert abs(g - g_ref) / g_ref < 0.015, f"{name}: {g:.3f} GMACs, expected {g_ref}"
    return True


if __name__ == "__main__":
    for name, (fn, p_ref, g_ref) in REFERENCE.items():
        net = fn()
        print(f"{name:13s} params {net.params:>12,} (ref {p_ref:>12,})  "
              f"GMACs {net.macs / 1e9:6.3f} (ref {g_ref})")
    verify()
    print("all reference counts reproduced")
