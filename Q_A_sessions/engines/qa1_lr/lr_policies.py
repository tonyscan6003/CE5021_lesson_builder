"""Q&A 1 — learning-rate policies.

Every curve is a real training run of the toy problem in toy.py (three-arm
spiral, small MLP, SGD with momentum 0.9, batch 32, 47 batches per epoch,
40 epochs), or the learning rate produced by a real torch.optim scheduler.
None of it uses the CIFAR-10 network or the settings of Assignment 1.

    toy_problem         the training set: three spiral arms, one per class
    fixed_lr            too small / suitable / too large fixed learning rates
    reduce_lr           an over-large rate rescued by StepLR or ReduceLROnPlateau
    warmup              linear warm-up then cosine decay, against cosine alone, over 8 seeds
    cyclic_lr           Smith's range test; triangular CyclicLR and OneCycleLR
    scheduler_catalogue the lr of each torch.optim.lr_scheduler class over a run
    step_placement      scheduler.step() per epoch vs per batch, with 391 batches per epoch

Engine contract as in lessons/engines: FIGURES, build(ctx) -> list[Figure], demo().
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from lessons.engines._common import BuildContext, Figure, C, SERIES, save_mpl
from . import toy

FIGURES = ["toy_problem", "fixed_lr", "reduce_lr", "warmup", "cyclic_lr", "scheduler_catalogue",
           "step_placement"]

FS = 13
B = None                                   # batches per epoch, set on first run


def _style():
    plt.rcParams.update({"font.size": FS, "axes.titlesize": FS, "axes.labelsize": FS,
                         "legend.fontsize": FS - 1, "xtick.labelsize": FS - 1,
                         "ytick.labelsize": FS - 1, "axes.spines.top": False,
                         "axes.spines.right": False, "lines.linewidth": 2.2})


def _sgd(lr):
    import torch
    return lambda p: torch.optim.SGD(p, lr=lr, momentum=0.9)


def _epochs(r):
    return np.arange(1, len(r["va_acc"]) + 1)


def pct(v):
    return f"{100 * v:.1f} %" if v >= 0.995 else f"{100 * v:.0f} %"


def _save(fig, ctx, fid):
    return save_mpl(fig, ctx, fid, dpi=200)


# ---------------------------------------------------------------------------

def fig_toy_problem(ctx: BuildContext) -> Figure:
    (x, y), (xv, _) = toy.data()
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    for k, col in enumerate((C["blue"], C["orange"], C["green"])):
        ax.scatter(*x[y == k].T, s=7, color=col, alpha=0.85, lw=0)
    ax.set_aspect("equal"); ax.axis("off")
    png = _save(fig, ctx, "toy_problem")
    return Figure("toy_problem",
                  f"The toy problem: {len(x)} training and {len(xv)} validation points on "
                  "three spiral arms, classified by an MLP with two hidden layers of "
                  f"{toy.HIDDEN} units.", png=png)


LR_SMALL, LR_GOOD, LR_LARGE = 0.003, 0.03, 0.3


def fig_fixed_lr(ctx: BuildContext) -> Figure:
    runs = {lr: toy.train(_sgd(lr)) for lr in (LR_SMALL, LR_GOOD, LR_LARGE)}
    acc = {lr: r["va_acc"][-1] for lr, r in runs.items()}
    assert acc[LR_GOOD] > 0.95 and acc[LR_SMALL] < 0.8 and acc[LR_LARGE] < 0.6, acc
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    labels = {LR_SMALL: "too small", LR_GOOD: "suitable", LR_LARGE: "too large"}
    for (lr, r), col in zip(runs.items(), (C["sky"], C["green"], C["red"])):
        e = _epochs(r)
        axes[0].plot(e, r["tr_loss"], color=col, label=f"lr = {lr} ({labels[lr]})")
        axes[1].plot(e, 100 * r["va_acc"], color=col)
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("training loss")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_ylim(0, 102)
    fig.tight_layout()
    png = _save(fig, ctx, "fixed_lr")
    return Figure("fixed_lr",
                  f"SGD with momentum 0.9 at three fixed learning rates. Final validation "
                  f"accuracy: {pct(acc[LR_SMALL])} (lr {LR_SMALL}), {pct(acc[LR_GOOD])} "
                  f"(lr {LR_GOOD}), {pct(acc[LR_LARGE])} (lr {LR_LARGE}; chance is 33 %).",
                  png=png)


LR_HIGH = 0.25


def fig_reduce_lr(ctx: BuildContext) -> Figure:
    import torch
    S = torch.optim.lr_scheduler
    runs = {"fixed": toy.train(_sgd(LR_HIGH)),
            "StepLR (×0.1 at epoch 20)": toy.train(_sgd(LR_HIGH), lambda o: S.StepLR(o, 20, 0.1)),
            "ReduceLROnPlateau (×0.5, patience 2)": toy.train(
                _sgd(LR_HIGH), lambda o: S.ReduceLROnPlateau(o, factor=0.5, patience=2),
                step="plateau")}
    acc = {k: r["va_acc"][-1] for k, r in runs.items()}
    keys = list(runs)
    assert acc[keys[0]] < 0.6 and acc[keys[1]] > 0.9 and acc[keys[2]] > 0.95, acc
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for (k, r), col in zip(runs.items(), (C["red"], C["blue"], C["green"])):
        nb = r["batches"]
        lr_ep = r["lr"][::nb]
        e = _epochs(r)
        axes[0].step(e, lr_ep, where="post", color=col, label=k)
        axes[1].plot(e, 100 * r["va_acc"], color=col)
    axes[0].set_yscale("log"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("learning rate")
    axes[1].legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="lower right")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation accuracy (%)")
    axes[1].set_ylim(0, 102)
    fig.tight_layout()
    png = _save(fig, ctx, "reduce_lr")
    return Figure("reduce_lr",
                  f"A learning rate of {LR_HIGH} is too large for the network to settle: "
                  f"with it fixed the validation accuracy ends at {pct(acc[keys[0]])}. "
                  f"Reducing it at epoch 20 gives {pct(acc[keys[1]])}, and halving it "
                  f"whenever the validation loss stops improving gives {pct(acc[keys[2]])}.",
                  png=png)


PEAK, WARM, N_SEEDS = 0.4, 5, 8


def _warm_cos(o, nb, epochs=toy.EPOCHS):
    import torch
    S = torch.optim.lr_scheduler
    return S.SequentialLR(o, [S.LinearLR(o, 0.01, 1.0, WARM * nb),
                              S.CosineAnnealingLR(o, (epochs - WARM) * nb)], [WARM * nb])


def fig_warmup(ctx: BuildContext) -> Figure:
    import torch
    S = torch.optim.lr_scheduler
    nb = toy.train(_sgd(0.01), epochs=1)["batches"]
    mk = {"warm-up 5 epochs, then cosine": lambda o: _warm_cos(o, nb),
          "cosine only": lambda o: S.CosineAnnealingLR(o, toy.EPOCHS * nb)}
    runs = {k: [toy.train(_sgd(PEAK), m, step="batch", seed=s) for s in range(N_SEEDS)]
            for k, m in mk.items()}
    final = {k: np.array([r["va_acc"][-1] for r in rs]) for k, rs in runs.items()}
    stuck = {k: int((v < 0.4).sum()) for k, v in final.items()}
    kw, kc = list(mk)
    assert stuck[kc] >= N_SEEDS // 2 and stuck[kw] == 0, stuck
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw=dict(width_ratios=[1.1, 1.3, 0.8]))
    cols = {kw: C["green"], kc: C["red"]}
    ep10 = 10 * nb
    for k in mk:
        r = runs[k][0]
        it = np.arange(len(r["lr"])) / nb
        axes[0].plot(it, r["lr"], color=cols[k], label=k.replace(", then", ",\nthen"))
        sm = np.convolve(r["loss_b"][:ep10], np.ones(9) / 9, mode="valid")
        axes[1].plot(np.arange(len(sm)) / nb, sm, color=cols[k])
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("learning rate")
    axes[0].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.05, 0.0), fontsize=FS - 3)
    axes[0].set_title(f"(a) the two schedules, peak {PEAK}")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("training loss (per batch)")
    axes[1].set_title("(b) the first 10 epochs, one run")
    for j, k in enumerate(mk):
        x = np.full(N_SEEDS, j) + np.linspace(-0.12, 0.12, N_SEEDS)
        axes[2].scatter(x, 100 * final[k], color=cols[k], s=46, zorder=3)
    axes[2].axhline(100 / 3, color=C["grey"], lw=1, ls="--")
    axes[2].text(1.45, 100 / 3 + 1.5, "chance", color=C["grey"], fontsize=FS - 2, ha="right")
    axes[2].set_xticks([0, 1], ["warm-up", "no\nwarm-up"]); axes[2].set_xlim(-0.5, 1.5)
    axes[2].set_ylim(0, 102); axes[2].set_ylabel("final validation accuracy (%)")
    axes[2].set_title(f"(c) {N_SEEDS} runs each")
    fig.tight_layout()
    png = _save(fig, ctx, "warmup")
    return Figure("warmup",
                  f"A peak learning rate of {PEAK} with and without a {WARM}-epoch linear "
                  f"warm-up. Without it, {stuck[kc]} of {N_SEEDS} runs stall at chance from the "
                  f"first epochs; with warm-up, none do "
                  f"(median {100 * np.median(final[kw]):.0f} %). The peak is still too high "
                  "for a good result: warm-up protects the start of training, it does not "
                  "fix an unsuitable learning rate.", png=png)


CLR_HI = 0.3                  # max_lr: below the lr at which the range-test loss is lowest
CLR_LO = CLR_HI / 4           # base_lr: Smith's rule of thumb, a third to a quarter of max_lr
SHORT, SHORT_SEEDS = 10, 3    # a short training budget (epochs), and the runs per policy


def fig_cyclic_lr(ctx: BuildContext) -> Figure:
    import torch
    S = torch.optim.lr_scheduler
    lrs, losses = toy.lr_range_test()
    lr_min = float(lrs[int(np.argmin(losses))])
    assert CLR_HI < lr_min, (CLR_HI, lr_min)
    nb = toy.train(_sgd(0.01), epochs=1)["batches"]
    policies = {
        f"fixed {LR_GOOD}": (lambda: _sgd(LR_GOOD), None),
        "fixed 0.1": (lambda: _sgd(0.1), None),
        f"CyclicLR {CLR_LO:.3g}–{CLR_HI}": (
            lambda: _sgd(CLR_LO),
            lambda o: S.CyclicLR(o, CLR_LO, CLR_HI, step_size_up=SHORT // 2 * nb)),
        f"OneCycleLR max {CLR_HI}": (
            lambda: _sgd(CLR_HI),
            lambda o: S.OneCycleLR(o, CLR_HI, epochs=SHORT, steps_per_epoch=nb)),
    }
    runs = {k: [toy.train(mo(), ms, step="batch", epochs=SHORT, seed=sd) for sd in range(SHORT_SEEDS)]
            for k, (mo, ms) in policies.items()}
    final = {k: np.array([r["va_acc"][-1] for r in rs]) for k, rs in runs.items()}
    k0, k1, k2, k3 = list(policies)
    # 0.03 is too slow for the budget; 0.1 is erratic (good for some seeds, poor for others)
    assert final[k0].max() < 0.9 and final[k1].min() < 0.9, final
    assert final[k2].min() > 0.97 and final[k3].min() > 0.97, final
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2),
                             gridspec_kw=dict(width_ratios=[1, 1.1, 1]))
    ax = axes[0]
    ax.plot(lrs, losses, color=C["ink"])
    ax.set_xscale("log"); ax.set_xlabel("learning rate (increased every batch)")
    ax.set_ylabel("training loss (smoothed)")
    for v, t in ((CLR_LO, "base_lr"), (CLR_HI, "max_lr")):
        ax.axvline(v, color=C["orange"], ls="--", lw=1.4)
    top = ax.get_ylim()[1]
    ax.text(CLR_HI * 1.12, top, "max_lr", color=C["orange"], fontsize=FS - 2, va="top")
    ax.text(CLR_LO / 1.12, top, "base_lr", color=C["orange"], fontsize=FS - 2, va="top",
            ha="right")
    ax.set_title("(a) range test: one short run")
    cols = {k0: C["light"], k1: C["grey"], k2: C["blue"], k3: C["purple"]}
    for k in (k2, k3):
        r = runs[k][0]
        axes[1].plot(np.arange(len(r["lr"])) / nb, r["lr"], color=cols[k], label=k)
    axes[1].axhline(LR_GOOD, color=cols[k0], lw=2, label=k0)
    axes[1].axhline(0.1, color=cols[k1], lw=2, label=k1)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("learning rate")
    axes[1].legend(loc="upper right", fontsize=FS - 3, framealpha=0.95, edgecolor="none")
    axes[1].set_ylim(0, 0.42)
    axes[1].set_title(f"(b) the policies, {SHORT} epochs")
    for j, k in enumerate(policies):
        x = np.full(SHORT_SEEDS, j) + np.linspace(-0.1, 0.1, SHORT_SEEDS)
        axes[2].scatter(x, 100 * final[k], color=cols[k] if k != k0 else C["grey"], s=50,
                        zorder=3, marker="o" if j > 1 else "s")
    axes[2].set_xticks(range(4), ["fixed\n0.03", "fixed\n0.1", "cyclic", "one-\ncycle"])
    axes[2].set_xlim(-0.5, 3.5); axes[2].set_ylim(70, 101)
    axes[2].set_ylabel("validation accuracy (%)")
    axes[2].set_title(f"(c) after {SHORT} epochs, {SHORT_SEEDS} runs each")
    fig.tight_layout()
    png = _save(fig, ctx, "cyclic_lr")
    rng = lambda v: f"{100 * v.min():.1f}–{100 * v.max():.1f} %"
    return Figure("cyclic_lr",
                  f"The range-test loss is lowest at lr ≈ {lr_min:.2f}; max_lr = {CLR_HI} is "
                  f"taken below it and base_lr = max_lr / 4. After {SHORT} epochs the fixed "
                  f"rates reach {rng(final[k0])} (0.03) and {rng(final[k1])} (0.1); one "
                  f"triangular cycle reaches {rng(final[k2])} and the one-cycle policy "
                  f"{rng(final[k3])}.", png=png)


def _lr_trace(make, epochs, nb, per="epoch", metric=None):
    """The lr seen at every batch for a scheduler stepped per epoch or per batch."""
    import torch
    p = [torch.nn.Parameter(torch.zeros(1))]
    o = torch.optim.SGD(p, lr=0.1, momentum=0.9)
    s = make(o)
    out = []
    for e in range(epochs):
        for b in range(nb):
            out.append(o.param_groups[0]["lr"])
            o.step()
            if per == "batch":
                s.step()
        if per == "epoch":
            s.step(metric[e]) if metric is not None else s.step()
    return np.array(out)


def fig_scheduler_catalogue(ctx: BuildContext) -> Figure:
    import torch
    S = torch.optim.lr_scheduler
    E, nb = 30, 50
    vloss = toy.train(_sgd(LR_HIGH), epochs=E)["va_loss"]     # a real validation-loss curve
    cat = [
        ("StepLR(step_size=10, gamma=0.1)", "epoch", lambda o: S.StepLR(o, 10, 0.1), None),
        ("MultiStepLR([10, 20], gamma=0.3)", "epoch", lambda o: S.MultiStepLR(o, [10, 20], 0.3), None),
        ("ExponentialLR(gamma=0.9)", "epoch", lambda o: S.ExponentialLR(o, 0.9), None),
        ("CosineAnnealingLR(T_max=30)", "epoch", lambda o: S.CosineAnnealingLR(o, E), None),
        ("ReduceLROnPlateau(factor=0.5,\npatience=2)  .step(val_loss)", "epoch",
         lambda o: S.ReduceLROnPlateau(o, factor=0.5, patience=2), vloss),
        ("SequentialLR: LinearLR warm-up\nthen CosineAnnealingLR", "batch",
         lambda o: S.SequentialLR(o, [S.LinearLR(o, 0.01, 1.0, 3 * nb),
                                      S.CosineAnnealingLR(o, (E - 3) * nb)], [3 * nb]), None),
        ("CyclicLR(base_lr=0.01, max_lr=0.1,\nstep_size_up=250)", "batch",
         lambda o: S.CyclicLR(o, 0.01, 0.1, step_size_up=5 * nb), None),
        ("OneCycleLR(max_lr=0.1, epochs=30,\nsteps_per_epoch=50)", "batch",
         lambda o: S.OneCycleLR(o, 0.1, epochs=E, steps_per_epoch=nb), None),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 4.6), sharex=True)
    for ax, (name, per, make, metric) in zip(axes.ravel(), cat):
        tr = _lr_trace(make, E, nb, per, metric)
        col = C["blue"] if per == "epoch" else C["purple"]
        ax.plot(np.arange(len(tr)) / nb, tr, color=col, lw=2)
        ax.set_title(name, fontsize=FS - 2)
        ax.text(0.97, 0.97, f"step() per {per}", transform=ax.transAxes, ha="right", va="top",
                fontsize=FS - 2, color=col, weight="bold",
                bbox=dict(fc="white", ec="none", alpha=0.9, pad=1.5))
        ax.set_ylim(0, 0.13); ax.tick_params(labelsize=FS - 2)
    for ax in axes[1]:
        ax.set_xlabel("epoch", fontsize=FS - 2)
    for ax in axes[:, 0]:
        ax.set_ylabel("learning rate", fontsize=FS - 2)
    fig.tight_layout()
    png = _save(fig, ctx, "scheduler_catalogue")
    return Figure("scheduler_catalogue",
                  "The learning rate produced by eight torch.optim.lr_scheduler set-ups over "
                  f"{E} epochs of {nb} batches (initial lr 0.1). ReduceLROnPlateau is driven "
                  "by a real validation-loss curve of the toy problem.", png=png)


def fig_step_placement(ctx: BuildContext) -> Figure:
    import torch
    S = torch.optim.lr_scheduler
    E, nb = 20, 391                     # CIFAR-10: 50,000 images / batch 128 = 391 batches
    assert -(-50000 // 128) == nb
    cases = [
        ("StepLR(step_size=5, gamma=0.5): step() per epoch", "epoch",
         lambda o: S.StepLR(o, 5, 0.5), C["blue"]),
        ("OneCycleLR(max_lr=0.1): step() per batch", "batch",
         lambda o: S.OneCycleLR(o, 0.1, epochs=E, steps_per_epoch=nb), C["purple"]),
        ("mistake: that StepLR, step() per batch", "batch",
         lambda o: S.StepLR(o, 5, 0.5), C["red"]),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 5.2), sharex=True)
    traces = []
    for ax, (name, per, make, col) in zip(axes, cases):
        tr = _lr_trace(make, E, nb, per)
        traces.append(tr)
        ax.plot(np.arange(len(tr)) / nb, tr, color=col, lw=2.2)
        ax.set_title(name, fontsize=FS - 2, loc="left")
        ax.set_ylim(0, 0.11); ax.set_ylabel("lr", fontsize=FS - 1)
    axes[-1].set_xlabel("epoch")
    # the mistake: the lr has fallen below 1e-6 within the first epoch
    below = int(np.argmax(traces[2] < 1e-6))
    assert 0 < below < nb, below
    axes[2].annotate(f"below 10⁻⁶ after {below} batches", xy=(below / nb, 0.0),
                     xytext=(2.5, 0.06), fontsize=FS - 2, color=C["red"],
                     arrowprops=dict(arrowstyle="->", color=C["red"]))
    fig.tight_layout()
    png = _save(fig, ctx, "step_placement")
    return Figure("step_placement",
                  f"The learning rate over {E} epochs of {nb} batches, the batch count of "
                  "CIFAR-10 at a batch size of 128. A per-epoch scheduler stepped every "
                  f"batch decays {nb} times too fast: here the lr is below 10⁻⁶ after "
                  f"{below} batches.", png=png)


def build(ctx: BuildContext) -> list[Figure]:
    _style()
    return [fig_toy_problem(ctx), fig_fixed_lr(ctx), fig_reduce_lr(ctx), fig_warmup(ctx), fig_cyclic_lr(ctx),
            fig_scheduler_catalogue(ctx), fig_step_placement(ctx)]


def demo():
    from pathlib import Path
    import tempfile
    d = Path(tempfile.mkdtemp())
    for f in build(BuildContext(d, d / "figures")):
        print(f.id, f.png)
