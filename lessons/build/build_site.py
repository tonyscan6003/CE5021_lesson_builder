#!/usr/bin/env python3
"""Build the CE6021 lesson site.

    python lessons/build/build_site.py            # build every lesson in content/
    python lessons/build/build_site.py --only 01_02

Pipeline per lesson:
    1. read content/<id>_*.md (YAML front matter + markdown body)
    2. run the figure engines named in the front matter -> site/figures/*.png (+ html)
    3. expand directives  {{fig ID}}  {{example TYPE seed=N}}  {{selfcheck ID}}
    4. render markdown (maths protected, KaTeX in the browser) into the page template
Also writes site/index.html from content/_course.yaml.
"""
from __future__ import annotations

import argparse
import html
import importlib
import re
import shutil
import sys
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader
from markdown_it import MarkdownIt

LESSONS = Path(__file__).resolve().parents[1]
REPO = LESSONS.parent
sys.path.insert(0, str(REPO))

from lessons.engines._common import BuildContext, Figure, credits  # noqa: E402
from lessons.engines._vendor import worked_example  # noqa: E402

CONTENT = LESSONS / "content"
SITE = LESSONS / "site"          # rebound by --out
FIG_DIR = SITE / "figures"       # rebound by --out
BUILD = LESSONS / "build"

md = MarkdownIt("commonmark", {"html": True, "typographer": True}).enable(["table", "strikethrough"])
env = Environment(loader=FileSystemLoader(BUILD), autoescape=False)


# ---------------------------------------------------------------------------
# Markdown with protected maths
# ---------------------------------------------------------------------------

_MATH_RE = re.compile(r"(\$\$.*?\$\$|(?<!\\)\$(?!\s)(?:[^$\n\\]|\\.)+?(?<!\s)\$)", re.S)


def render_markdown(text: str) -> str:
    """Render markdown; $...$ / $$...$$ spans are hidden from the parser and
    re-inserted verbatim (HTML-escaped) for KaTeX auto-render in the browser."""
    store: list[str] = []

    def stash(m):
        store.append(m.group(0))
        return f"\x02MATH{len(store) - 1}\x03"

    protected = _MATH_RE.sub(stash, text)
    out = md.render(protected)

    def restore(m):
        return html.escape(store[int(m.group(1))], quote=False)

    return re.sub(r"\x02MATH(\d+)\x03", restore, out)


# ---------------------------------------------------------------------------
# Front matter and directives
# ---------------------------------------------------------------------------

def split_front_matter(text: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
    if not m:
        raise ValueError("content file needs YAML front matter")
    return yaml.safe_load(m.group(1)), m.group(2)


def run_engines(engine_names: list[str], ctx: BuildContext) -> dict[str, Figure]:
    figs: dict[str, Figure] = {}
    for name in engine_names:
        mod = importlib.import_module(f"lessons.engines.{name}")
        print(f"    engine {name}: ", end="", flush=True)
        for fig in mod.build(ctx):
            figs[fig.id] = fig
            print(fig.id, end=" ", flush=True)
        print()
    return figs


def credit_line(key: str) -> str:
    c = credits()[key]
    title = html.escape(c["title"])
    if c.get("source_url"):
        title = f'<a href="{c["source_url"]}">{title}</a>'
    lic = html.escape(c["licence"])
    if c.get("licence_url"):
        lic = f'<a href="{c["licence_url"]}">{lic}</a>'
    return f'{title}, {html.escape(c["author"])}, {lic}'


def figure_html(fig: Figure, number: int) -> str:
    cap = f"<b>Figure {number}.</b> {fig.caption}"
    if fig.sources:
        for key in fig.sources:
            if key not in credits():
                raise KeyError(f"figure {fig.id}: unknown image credit key {key!r}")
        cap += ' <span class="credit">Image: ' + "; ".join(credit_line(k) for k in fig.sources) + ".</span>"
    if fig.kind == "widget":
        return (f'<figure class="lesson-fig" id="fig-{fig.id}">'
                f'<img src="{fig.png}" alt="{html.escape(fig.caption)}">'
                f'<figcaption>{cap}</figcaption>{fig.html}</figure>')
    toggle = ""
    inter = ""
    if fig.html:
        toggle = (f' <button class="fig-toggle" data-target="fig-{fig.id}" type="button">'
                  f'Interactive version</button>')
        inter = f'<div class="fig-interactive" hidden>{fig.html}</div>'
    return (f'<figure class="lesson-fig" id="fig-{fig.id}">'
            f'<img src="{fig.png}" alt="{html.escape(fig.caption)}">{inter}'
            f'<figcaption>{cap}{toggle}</figcaption></figure>')


def example_html(type_name: str, seed: int, number: int, extra: dict) -> str:
    mdtext = worked_example(type_name, seed, fig_dir=FIG_DIR, fig_prefix="figures", **extra)
    title_m = re.search(r"^## Question:\s*(.+)$", mdtext, re.M)
    title = title_m.group(1).strip() if title_m else type_name
    mdtext = re.sub(r"^## Question:.*$", "", mdtext, count=1, flags=re.M)
    q, _, sol = mdtext.partition("### Solution")
    q = q.rstrip().rstrip("-").rstrip()
    body = (f'<div class="example-q">{render_markdown(q)}</div>'
            f'<details class="example-sol"><summary>Show the solution</summary>'
            f'{render_markdown(sol)}</details>')
    return (f'<section class="worked-example" id="ex-{type_name}-{seed}">'
            f'<h4>Worked example {number}: {html.escape(title)}</h4>{body}</section>')


def selfcheck_html(check_id: str) -> str:
    path = CONTENT / "_selfcheck" / f"{check_id}.yaml"
    if not path.exists():
        return ""
    items = yaml.safe_load(path.read_text(encoding="utf-8"))
    parts = ['<section class="selfcheck"><h2 id="self-check">Check your understanding</h2>']
    for qi, it in enumerate(items):
        parts.append(f'<div class="sc-q" data-answer="{it["answer"]}"><p class="sc-text">'
                     f'{qi + 1}. {html.escape(it["q"])}</p><ul class="sc-options">')
        for oi, opt in enumerate(it["options"]):
            parts.append(f'<li><label><input type="radio" name="sc{check_id}_{qi}" value="{oi}"> '
                         f'{html.escape(str(opt))}</label></li>')
        parts.append(f'</ul><p class="sc-feedback" hidden></p>'
                     f'<p class="sc-explain" hidden>{html.escape(it.get("explain", ""))}</p></div>')
    parts.append('<p><button type="button" class="sc-check">Check answers</button> '
                 '<span class="sc-score"></span></p></section>')
    return "".join(parts)


# Directives are expanded AFTER the page markdown has been rendered, so the HTML
# they produce is never fed back through render_markdown. Feeding it back would
# escape the maths inside a worked example a second time, turning the "&" column
# separators of a LaTeX matrix into "&amp;" and printing "amp;" in every column.
# Markdown wraps a directive on its own line in a paragraph, so match that too.
_DIRECTIVE_RE = re.compile(
    r"^[ \t]*<p>\{\{\s*(fig|example|selfcheck)\s+([^}]+?)\s*\}\}</p>[ \t]*$", re.M)


def expand_directives(body: str, figs: dict[str, Figure]) -> tuple[str, bool]:
    counters = {"fig": 0, "example": 0}
    uses_plotly = False

    def repl(m):
        nonlocal uses_plotly
        kind, arg = m.group(1), m.group(2)
        if kind == "fig":
            fid = arg.strip()
            if fid not in figs:
                raise KeyError(f"figure {fid!r} not produced by the lesson's engines")
            counters["fig"] += 1
            if figs[fid].kind == "plotly":
                uses_plotly = True
            return figure_html(figs[fid], counters["fig"])
        if kind == "example":
            toks = arg.split()
            type_name, extra = toks[0], {}
            seed = 0
            for t in toks[1:]:
                k, v = t.split("=")
                if k == "seed":
                    seed = int(v)
                else:
                    extra[k] = int(v)
            counters["example"] += 1
            return example_html(type_name, seed, counters["example"], extra)
        return selfcheck_html(arg.strip())

    return _DIRECTIVE_RE.sub(repl, body), uses_plotly


def add_heading_ids(page_html: str):
    """Give every <h2> an id and return (html, [(id, text), ...]) for the page TOC."""
    items = []

    def repl(m):
        attrs, text = m.group(1) or "", m.group(2)
        if 'id="' in attrs:
            hid = re.search(r'id="([^"]+)"', attrs).group(1)
        else:
            plain = re.sub(r"<[^>]+>", "", text)
            hid = re.sub(r"[^a-z0-9]+", "-", plain.lower()).strip("-")
        items.append((hid, re.sub(r"<[^>]+>", "", text)))
        return f'<h2 id="{hid}">{text}</h2>'

    return re.sub(r"<h2([^>]*)>(.*?)</h2>", repl, page_html), items


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def load_course():
    course = yaml.safe_load((CONTENT / "_course.yaml").read_text(encoding="utf-8"))
    order = []
    for sec in course["sections"]:
        for les in sec["lessons"]:
            les["section"] = sec["number"]
            les["file"] = f"{les['id']}.html"
            order.append(les)
    return course, order


def content_files() -> dict[str, Path]:
    return {p.stem[:5]: p for p in sorted(CONTENT.glob("[0-9][0-9]_[0-9][0-9]_*.md"))}


def build_lesson(path: Path, course, order, built_ids: set[str]) -> None:
    meta, body = split_front_matter(path.read_text(encoding="utf-8"))
    lid = meta["id"]
    print(f"  {lid} {meta['title']}")
    ctx = BuildContext(site_dir=SITE, fig_dir=FIG_DIR)
    figs = run_engines(meta.get("engines", []), ctx)
    page_html = render_markdown(body)
    page_html, uses_plotly = expand_directives(page_html, figs)
    page_html, toc = add_heading_ids(page_html)
    leftover = re.search(r"\{\{\s*(fig|example|selfcheck)\b[^}]*\}\}", page_html)
    if leftover:
        raise ValueError(f"{lid}: unexpanded directive {leftover.group(0)!r}")
    used = sorted({k for f in figs.values() for k in (f.sources or [])})
    image_credits = [credit_line(k) for k in used]

    idx = [o["id"] for o in order].index(lid)
    prev_ = next((o for o in reversed(order[:idx]) if o["id"] in built_ids), None)
    next_ = next((o for o in order[idx + 1:] if o["id"] in built_ids), None)
    section = next(s for s in course["sections"] if s["number"] == meta["section"])

    tmpl = env.get_template("template.html.j2")
    out = tmpl.render(course=course, meta=meta, section=section, body=page_html, toc=toc,
                      prev=prev_, next=next_, uses_plotly=uses_plotly, built=built_ids,
                      practice=meta.get("practice", []), lesson_number=lid.replace("_", "."),
                      image_credits=image_credits)
    (SITE / f"{lid}.html").write_text(out, encoding="utf-8")


def build_index(course, order, built_ids: set[str], summaries: dict[str, str]) -> None:
    tmpl = env.get_template("index.html.j2")
    out = tmpl.render(course=course, built=built_ids, summaries=summaries)
    (SITE / "index.html").write_text(out, encoding="utf-8")


def read_released(path: Path) -> list[str]:
    """Lesson ids listed in a release manifest (one per line, # comments allowed)."""
    ids = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def main() -> None:
    global SITE, FIG_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="lesson ids to build, e.g. 01_02")
    ap.add_argument("--clean", action="store_true", help="delete the output directory first")
    ap.add_argument("--out", help="output directory (default lessons/site)")
    ap.add_argument("--released", nargs="*",
                    help="publish only these lesson ids: no page is written for any "
                         "other lesson, previous/next navigation skips them, and the "
                         "index greys them out")
    ap.add_argument("--released-file",
                    help="a file of released lesson ids, one per line (# comments allowed)")
    args = ap.parse_args()

    if args.out:
        SITE = Path(args.out).resolve()
        FIG_DIR = SITE / "figures"

    if args.clean and SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(BUILD / "style.css", SITE / "style.css")
    shutil.copy(BUILD / "site.js", SITE / "site.js")

    course, order = load_course()
    files = content_files()

    released = None
    if args.released_file:
        released = read_released(Path(args.released_file))
    if args.released:
        released = list(args.released)
    if released is not None:
        unknown = [r for r in released if r not in files]
        if unknown:
            raise SystemExit(f"released lesson(s) with no content file: {', '.join(unknown)}")
        built_ids = set(released)
    else:
        built_ids = set(files)

    # Only released lessons are ever rendered, so an unreleased page and its
    # figures never reach the output directory.
    targets = [lid for lid in (args.only or list(files)) if lid in built_ids]

    summaries = {lid: split_front_matter(p.read_text(encoding="utf-8"))[0].get("summary", "")
                 for lid, p in files.items()}

    if released is not None:
        held = [lid for lid in files if lid not in built_ids]
        print(f"Releasing {len(built_ids)} of {len(files)} lessons; "
              f"{len(held)} held back: {', '.join(sorted(held)) or 'none'}")
    print("Building lessons:")
    for lid in targets:
        build_lesson(files[lid], course, order, built_ids)
    build_index(course, order, built_ids, summaries)
    print(f"Done -> {SITE}")


if __name__ == "__main__":
    main()
