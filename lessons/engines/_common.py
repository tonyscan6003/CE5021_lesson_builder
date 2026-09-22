"""Shared helpers for lesson figure engines.

Every engine module exposes:

    FIGURES: list[str]                      figure ids it produces
    build(ctx: BuildContext) -> list[Figure]
    demo() -> None                          Streamlit page (optional)

Static figures are matplotlib PNGs. Interactive figures are Plotly divs
(plotly.js is loaded once per page from the CDN by the site template) or
small self-contained HTML widgets. No browser is needed at build time.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import plotly.io as pio  # noqa: E402

LESSONS_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = LESSONS_DIR / "data"

# Colour-blind safe series palette (Okabe-Ito) plus neutral greys.
C = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#6c757d",
    "light": "#d9dde3",
    "ink": "#1f2933",
}
SERIES = [C["blue"], C["orange"], C["green"], C["red"], C["purple"], C["sky"]]


def setup_mpl() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 120,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "image.cmap": "gray",
        "legend.frameon": False,
        "figure.constrained_layout.use": True,
    })


setup_mpl()


@dataclass
class Figure:
    id: str
    caption: str
    png: str | None = None      # path relative to the site root
    html: str | None = None     # interactive Plotly div or widget markup
    kind: str = "static"        # static | plotly | widget
    sources: list[str] | None = None   # credit keys of images used (auto-filled)

    def __post_init__(self):
        # Every image loaded through load_image() since the previous Figure was
        # created belongs to this figure. Engines may also pass sources explicitly.
        global _PENDING_SOURCES
        if self.sources is None:
            self.sources = sorted(_PENDING_SOURCES)
        _PENDING_SOURCES = set()


_PENDING_SOURCES: set[str] = set()
_CREDITS: dict | None = None


def credits() -> dict:
    """Credits for every source image, from data/credits.yaml (see build/fetch_images.py)."""
    global _CREDITS
    if _CREDITS is None:
        import yaml
        path = DATA_DIR / "credits.yaml"
        _CREDITS = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    return _CREDITS


@dataclass
class BuildContext:
    site_dir: Path
    fig_dir: Path

    def rel(self, path: Path | str) -> str:
        return Path(path).relative_to(self.site_dir).as_posix()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def save_mpl(fig, ctx: BuildContext, fid: str, dpi: int = 120) -> str:
    """Save a matplotlib figure as figures/<fid>.png and return the site-relative path."""
    ctx.fig_dir.mkdir(parents=True, exist_ok=True)
    path = ctx.fig_dir / f"{fid}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return ctx.rel(path)


def plotly_html(fig, height: int = 480) -> str:
    """Return a Plotly figure as an embeddable div (plotly.js loaded by the page)."""
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="system-ui, sans-serif", size=13),
        height=height,
    )
    return pio.to_html(
        fig, include_plotlyjs=False, full_html=False,
        config={"displaylogo": False, "responsive": True},
        default_width="100%", default_height=f"{height}px",
    )


def slider_frames_layout(labels, prefix: str, extra_updatemenus=None) -> dict:
    """Standard slider spec for a frame-animated Plotly figure."""
    steps = [
        dict(method="animate", label=str(lab),
             args=[[str(lab)], dict(mode="immediate",
                                    frame=dict(duration=0, redraw=True),
                                    transition=dict(duration=0))])
        for lab in labels
    ]
    layout = dict(sliders=[dict(steps=steps, currentvalue=dict(prefix=prefix),
                                pad=dict(t=30), x=0.05, len=0.9)])
    if extra_updatemenus:
        layout["updatemenus"] = extra_updatemenus
    return layout


def to_data_uri(img: np.ndarray, fmt: str = "png", quality: int = 85) -> str:
    """Encode a uint8 grey or RGB image as a data URI (for go.Image(source=...)).

    Use fmt="jpg" for photographic content to keep embedded pages small;
    keep "png" for block/pixel-art images where exact values matter.
    """
    img = to_uint8(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if fmt == "jpg":
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", img)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode()


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def to_uint8(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return x
    x = np.asarray(x, dtype=np.float64)
    if x.max() <= 1.0 and x.min() >= 0.0:
        x = x * 255.0
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def norm01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def load_image(key: str, gray: bool = False, max_side: int | None = None) -> np.ndarray:
    """Load a credited source image by key (see data/credits.yaml) as RGB or grey uint8.

    Loading by key is deliberate: an image without a credit cannot be used,
    and the key is recorded so the figure caption can print the attribution.
    """
    cred = credits().get(key)
    if cred is None:
        raise KeyError(f"no credit entry for image {key!r}; add it to data/sources.yaml "
                       "and run build/fetch_images.py")
    _PENDING_SOURCES.add(key)
    path = DATA_DIR / cred["file"]
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if cred.get("trim_border"):
        img = trim_dark_border(img)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    if gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def use_source(key: str) -> None:
    """Record that the figure being built uses the credited source `key`
    (for datasets the engine loads itself, e.g. Olivetti faces)."""
    if key not in credits():
        raise KeyError(f"no credit entry for source {key!r}; add it to data/sources.yaml")
    _PENDING_SOURCES.add(key)


def skimage_image(name: str, gray: bool = False) -> np.ndarray:
    """Fetch a scikit-image sample image; only ones with a credit entry are allowed.

    The credit key is "skimage_<name>" and must exist in data/credits.yaml
    (declared with title/author/licence in data/sources.yaml).
    """
    key = f"skimage_{name}"
    if key not in credits():
        raise KeyError(f"scikit-image sample {name!r} has no credit entry ({key}); "
                       "only images of documented provenance may be used")
    _PENDING_SOURCES.add(key)
    from skimage import data
    img = getattr(data, name)()
    img = to_uint8(img)
    if gray and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def trim_dark_border(img: np.ndarray, thresh: float = 12.0) -> np.ndarray:
    """Remove near-black letterbox rows/columns around a photo."""
    g = img if img.ndim == 2 else img.mean(axis=2)
    rows = np.where(g.mean(axis=1) > thresh)[0]
    cols = np.where(g.mean(axis=0) > thresh)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def upscale_nearest(img: np.ndarray, factor: int) -> np.ndarray:
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=cv2.INTER_NEAREST)


def sync_matrix(ax, M, fmt="{:g}", cmap="Blues", highlight=None, title=None,
                fontsize=10, vmin=None, vmax=None, text_color=None):
    """Draw a small matrix as a coloured grid with the values printed in each cell."""
    M = np.asarray(M, dtype=float)
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    r, c = M.shape
    span = (np.nanmax(M) - np.nanmin(M)) or 1.0
    for i in range(r):
        for j in range(c):
            v = M[i, j]
            if np.isnan(v):
                continue
            lum = (v - np.nanmin(M)) / span
            col = text_color or ("white" if lum > 0.6 else C["ink"])
            ax.text(j, i, fmt.format(v), ha="center", va="center",
                    fontsize=fontsize, color=col)
    if highlight is not None:
        (i0, j0, h, w) = highlight
        ax.add_patch(plt.Rectangle((j0 - 0.5, i0 - 0.5), w, h, fill=False,
                                   lw=2.5, ec=C["red"]))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)
