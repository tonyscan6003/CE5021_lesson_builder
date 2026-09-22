# lessons/

Home for everything new in this repository: the CE5021 HTML lessons, the
Python engines that draw their figures, and the interactive examples that are
published to a public student repo.

See `COURSE_PLAN.md` for the full lesson map, engine list and build order.

> **Conversion in progress.** Lessons 1.1–1.3 are built; the other 14 are not
> written yet and show as "coming soon" on the index. The CE6021 course is
> preserved in this repository's first commit.

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
| `reference/pdf_for_conversion/` | The source lecture PDFs (gitignored, input only) |
| `reference/legacy_md/` | Those slide decks and narration scripts converted to markdown (input only) |

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

The slide decks and narration scripts under `reference/legacy_md/` are
converted from the lecture PDFs in `reference/pdf_for_conversion/`. The PDFs
are gitignored (53 MB, input only); the markdown is tracked.

```bash
python lessons/build/convert_pdfs.py \
    --input-dir lessons/reference/pdf_for_conversion \
    --output-dir lessons/reference/legacy_md --force
```

Uses PyMuPDF text-layer extraction, no OCR.

## Re-vendoring the question generators

The worked examples come from the question-bank repo. Clone it beside
`lessons/` (the clone is gitignored) and copy the generators across:

```bash
git clone https://github.com/tonyscan6003/autoquiz.git autoquiz
for g in $(python -c 'from lessons.engines._vendor import TYPES; print(*TYPES)'); do
    sed 's/from \.\.utils import/from ._utils import/' \
        autoquiz/question_tool/generators/$g.py > lessons/engines/_vendor/$g.py
done
```

That import redirect is the only change made to the files. `_vendor/__init__.py`
splits the list into `DL_TYPES` (deep learning, used by CE5021) and `CV_TYPES`
(classical machine vision, kept from CE6021).
