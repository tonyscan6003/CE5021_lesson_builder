# lessons/

Home for everything new in this repository: the CE6021 HTML lessons, the
Python engines that draw their figures, and the interactive examples that are
published to a public student repo.

See `COURSE_PLAN.md` for the full lesson map, engine list and build order.

## Layout

| Path | Contents |
|---|---|
| `content/` | One markdown file per lesson (YAML front matter + body with `{{fig …}}`, `{{example …}}`, `{{selfcheck …}}` directives); `_course.yaml` lists every lesson; `_selfcheck/` holds the quiz questions |
| `engines/` | Figure engines, one package per section. Each module has `FIGURES`, `build(ctx)` and `demo()` |
| `engines/_common.py` | Shared helpers: image loading, palette, matplotlib/Plotly export |
| `engines/_vendor/` | Question generators copied from the question-bank repo, used for worked examples |
| `build/` | `build_site.py`, page templates, `style.css`, `site.js` |
| `examples/app.py` | Streamlit app exposing every engine's `demo()` |
| `data/` | Image subset the engines need |
| `site/` | Generated HTML output (rebuilt by the build script) |
| `reference/legacy_md/` | Old slide decks and narration scripts converted from PDF (input only) |

## Build the site

```bash
pip install -r .devcontainer/requirements.txt   # already installed in the devcontainer
python lessons/build/build_site.py              # all lessons in content/
python lessons/build/build_site.py --only 01_02
python lessons/build/build_site.py --clean      # wipe site/ first
```

## Publish the student site

```bash
lessons/publish/publish_site.sh --dry-run   # build a preview, push nothing
lessons/publish/publish_site.sh --create    # first publication
lessons/publish/publish_site.sh --update    # release more lessons
```

`lessons/publish/released.txt` chooses which lessons go out. See
`COURSE_PLAN.md`, "Publishing the student site", for what is and is not
protected by this.

Open `lessons/site/index.html` in a browser. Maths is rendered by KaTeX and
interactive figures by plotly.js, both loaded from a CDN, so the pages need an
internet connection but no server.

## Run the interactive examples

```bash
streamlit run lessons/examples/app.py
```

## Add a lesson

1. Write `engines/<section>/<topic>.py` with `FIGURES`, `build(ctx) -> list[Figure]`
   and `demo()`; register it in `engines/__init__.py`.
2. Write `content/<id>_<slug>.md` with front matter (`id`, `title`, `section`,
   `engines`, `practice`, `summary`) and the directives.
3. Add `content/_selfcheck/<id>.yaml` (list of `q`, `options`, `answer`, `explain`).
4. Run the build script; check the PNGs in `site/figures/`.

Rules for demos: never reproduce a lab pipeline or use lab data; show
behaviour with library calls on different images (see `COURSE_PLAN.md`,
"Relationship to the labs").

## Regenerating the reference markdown

The slide decks and narration scripts under `reference/legacy_md/` were
converted from the original lecture PDFs. Those PDFs are not kept in this
repository. To convert a fresh set:

```bash
python lessons/build/convert_pdfs.py \
    --input-dir path/to/pdfs \
    --output-dir lessons/reference/legacy_md --force
```

Uses PyMuPDF text-layer extraction, no OCR.
