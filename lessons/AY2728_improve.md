# CE5021 improvements for AY 2027/28

These are recommendations for developing the CE5021 *Deep Learning for Computer Vision* content before the 2027/28 academic year. Each recommendation is marked as either **raised** (from the module lead's notes) or **additional** (suggested during the AY 2026/27 conversion work). The current state of the course is recorded in [COURSE_PLAN.md](COURSE_PLAN.md).

## Summary

| # | Recommendation | Origin | Priority | Effort | Depends on |
|---|---|---|---|---|---|
| 1 | A defined visual theme shared by the HTML pages and every figure engine | raised | high | medium | — |
| 2 | Modernise the content while keeping the focus on convolutional fundamentals | raised | high | medium | — |
| 3 | Prepare the repository for the follow-on transformer / generative module, and merge the Q&A engines | raised | high | medium | 1 |
| 4 | Integrate the Prince textbook, *Understanding Deep Learning*, as the reading spine | raised | medium | low | — |
| 5 | Companion Jupyter notebooks built on the existing engines | raised | medium | high | 1 |
| 6 | Rebuild the slide decks from the same source as the HTML | raised | medium | high | 1 |
| 7 | A narration pipeline that keeps the lecturer's own voice and speaking style | raised | medium | high | 6 |
| 8 | Quality, accessibility and maintenance work | additional | medium | low–medium | — |

Items 1 and 3 should be settled first. Notebooks, slides and a second module all reuse the figure engines, so a theme or structure fixed later would have to be applied several times.

---

## 1. Visual theme and consistency (raised)

The HTML pages and figures were converted lesson by lesson from slides of different ages. Each figure is correct, but there are now several visual families: matplotlib charts (about 180 figures), slide-style block diagrams (`_blocks.py`), feature-map volumes (`_volumes.py`), numeric grids (`sync_matrix` and the new `_grid_matrix`), and about 20 Plotly figures. These families differ in font size, line weight, colour use and aspect ratio.

**Recommendations**

1. **One source of design tokens.** The page palette is defined in `build/style.css` (`--ink`, `--accent`, …) and the figure palette in `engines/_common.py` (`C`). Both are already based on Okabe–Ito. Move fonts, colours, line widths and font sizes into a single theme file, read by the CSS build and by a shared matplotlib style (`.mplstyle`) and Plotly template, so that a change is made once.
2. **Fixed roles for colour.** Write down what each colour means and apply it in every lesson, for example:
   - one fill each for input, convolution, pooling, fully connected and output;
   - one colour each for ground truth, prediction and anchor.

   The recent matrix figure in 3.5 showed that an even, neutral fill works better than colour where colour carries no meaning.
3. **A figure-size budget.** The page column is 860 px. Define three or four standard figure widths and aspect ratios (full width, half width, tall diagram), and a minimum rendered font size, so that text in a tall diagram such as the FPN figure is not shrunk below that of a chart. The engines should take a size class rather than ad hoc `figsize` values.
4. **SVG output for diagrams.** Block and volume diagrams would be sharper as SVG, and could then follow a dark mode. Photographs and heat maps would stay as PNG.
5. **A contact sheet.** Add a build option that renders every figure in the course onto one page, to review consistency across lessons at a glance. It can also be kept as a visual regression check when the theme changes.
6. **Page furniture.** Treat learning outcomes, key equations, "in practice" notes, worked examples and self-checks as styled components, used the same way in every lesson.
7. **A written style guide.** Add a short figure and caption style guide to `COURSE_PLAN.md`, beside the prose-style section.

## 2. Modernisation of the content (raised)

The module is to stay focused on the fundamentals of convolution-based models, with transformers and generative models in the follow-on module. Within that scope, several developments since the source slides were written belong in a CNN module, and some topics should be signposted forward.

**Additions within scope**

| Lesson area | Suggested update |
|---|---|
| Training (1.5, 1.6) | AdamW and decoupled weight decay; learning-rate warm-up with cosine schedules; label smoothing; mixed-precision training |
| Normalisation (1.6, 2.3) | Group and layer normalisation beside batch normalisation, and when small batches make batch normalisation unreliable |
| Architectures (2.2, 2.3) | EfficientNet (compound scaling) and **ConvNeXt** (2022), the modern convolutional network that adopts design choices from transformers; it is a natural bridge to the follow-on module |
| Transfer learning (2.4) | Current practice with pretrained weights from `torchvision` and `timm`; augmentation policies such as RandAugment, mixup and CutMix |
| Visualisation (2.5) | Grad-CAM and related attribution methods, which are now the usual explanation tools |
| Detection (3.3, 3.4) | Anchor-free detectors (FCOS, CenterNet); the current YOLO family as the practical single-stage detector; COCO mAP averaged over IoU 0.5–0.95 |
| Segmentation (3.5, 3.6) | DeepLabV3+; panoptic segmentation as the union of semantic and instance segmentation |

**Signposts to the follow-on module**

These should be short forward references, not new lessons:
- **Vision transformer:** after ConvNeXt.
- **DETR:** after non-maximum suppression, as the detector that removes NMS.
- **Segment Anything:** after Mask R-CNN.

**Approach**

- Review each lesson against a recent survey or textbook chapter (see item 4) instead of rewriting from scratch. Most of the fundamentals have not changed.
- Keep the honesty rule used in the conversion: new figures are computed from real models, and their captions state what was measured.

## 3. A repository for two modules (raised)

The follow-on transformer and generative module will be added to this repository, with all content shared. The current structure assumes one course (`content/_course.yaml`, one `site/`, section packages `s1_`–`s3_` under `engines/`).

**Recommendations**

1. **One directory per module.** Give each module its own content directory and course file (for example `content/ce5021/` and `content/<next module>/`), and keep the engines shared. The build would take a `--course` option and could produce a combined index.
2. **Separate shared from module-specific engines.** Keep `_common`, `_blocks`, `_volumes`, `_torchvis` and the theme in a shared layer. Put the lesson engines in module packages.
3. **Shared notation and glossary.** Agree on one set of symbols for both modules: for example, the IoU, the box parameterisation $t_x, t_y, t_w, t_h$, the feature-map shapes $H \times W \times C$, and the loss terms. A glossary page shared by both modules would help students moving between them, and following the textbook's notation (item 4) would make this easier.
4. **Cross-links.** Allow a directive such as `{{see ce50xx:04_02}}` so the signposts in item 2 link to the lessons they point to.
5. **Repository size.** Cached model results (`data/_torch/*.npz`) and source images are committed. A second module with transformer and diffusion models will add more, so decide early whether to keep using plain Git, move to Git LFS, or regenerate the caches in CI.
6. **README.** `lessons/README.md` still says that only Lessons 1.1–1.3 are built, and should be updated.
7. **Merge the Q&A engines.** The Q&A session decks (`Q_A_sessions/`, from AY 2026/27) have their own content and engine packages, with the same engine contract as the lessons and the shared helpers of `lessons/engines/_common.py`. When the module is revised, engines that support both a lesson and a Q&A session should be moved into the shared layer, so that one engine serves the lesson page, the Q&A deck and any notebook. The learning-rate engine of Q&A 1 (`engines/qa1_lr`), for example, overlaps Lesson 1.5's schedule figure. The two content trees could also share one build with an output option for HTML or slides.

## 4. Integration of the Prince textbook (raised)

S. J. D. Prince, *Understanding Deep Learning* (MIT Press, 2023), is freely available online with accompanying Python notebooks (udlbook.github.io). It covers both this module and the follow-on module, so it can serve as the reading spine for both.

**Recommendations**

1. **Readings for each lesson.** Add a "Further reading" block to each lesson with the chapters and sections that match it. For CE5021 these are mainly:
   - the chapters on fitting, gradients and initialisation, performance and regularisation (Sections 1.5–1.6);
   - the chapters on convolutional and residual networks (Sections 1.3–2.3).

   The transformer and generative-model chapters map onto the follow-on module.
2. **Notation.** Where the lesson notation differs from the book, either adopt the book's notation or note the difference once in the glossary.
3. **Licensing.** Check the licence terms of the book's figures and notebooks before reusing any of them. The lessons should continue to use figures generated from the engines, with the book cited for derivations and wider treatment.
4. **Exercises.** The book's problems could supplement the self-checks and worked examples, as optional exercises.

## 5. Companion Jupyter notebooks (raised)

Notebooks would let students run the models and step through the operations that the HTML figures show.

**Recommendations**

1. **Build the notebooks on the engines.** Import `lessons.engines` so a notebook reproduces exactly the figure on the page and then lets the student change it: for example, the kernel, the stride, the dilation or the input image. This avoids a second implementation that can drift from the lesson.
2. **Two kinds of notebook:**
   - **Mechanism notebooks**, run on a CPU: convolution stepping, receptive fields, the convolution matrix and transpose convolution, IoU and NMS, and the anchor encoding. These can use `ipywidgets` sliders.
   - **Model notebooks**, which need a GPU: transfer learning, visualisation, running Faster R-CNN, RetinaNet and DeepLab on the student's own images, and fine-tuning on a small dataset. These should be ready to open in Colab, with pinned package versions.
3. **The Streamlit app.** `examples/app.py` already exposes every engine's `demo()`. Decide which role each plays: the app for quick interactive exploration, the notebooks for code students read and modify.
4. **Keep the notebooks working.** Run them in CI (for example with `nbmake` or `papermill`) on every build, strip their outputs before committing (`nbstripout`), and link each one from its lesson page with an "Open in Colab" button.
5. **Exercises.** Short tasks with automatic checks can follow the worked-example approach already used from the question bank.

## 6. Rebuilt slide decks (raised)

The original slides were the source for the HTML. The figures are now generated by the engines, so the direction can be reversed: the slides can be generated from the same source as the HTML.

**Recommendations**

1. **One source for both.** Mark slide content inside each lesson's markdown, for example a `slides:` block per section with the bullet points and the figure to show. The HTML and the deck are then built together and cannot diverge.
2. **Tooling.** Generate the decks with a tool that suits a Python build:
   - **Reveal.js or Quarto:** HTML decks themed with the same tokens as item 1.
   - **`python-pptx`:** if PowerPoint files are required by the institution or the recording workflow.

   A pilot on one lesson would settle the choice. The Q&A decks (`Q_A_sessions/build/build_decks.py`) already build `.pptx` files with `python-pptx` from YAML content and engine figures, with speaker notes, and can serve as that pilot.
3. **Slide variants of the figures.** Slides are 16:9 and are viewed from further away. The engines should accept a `variant="slide"` option that changes the aspect ratio and enlarges the text without changing the content. This depends on the size classes in item 1.
4. **Speaker notes.** The narration scripts of the original lectures are already converted in `reference/legacy_md/`. They can be updated to match the new content and attached to the slides as speaker notes, which then also feed the narration pipeline (item 7).

## 7. Narration pipeline (raised)

The main difficulty with rebuilt slides is the narration: the recordings should keep the lecturer's own voice and speaking style, and it should be possible to update a single slide without re-recording a whole lecture.

**Options**

| Option | Description | Strengths | Concerns |
|---|---|---|---|
| A. Assisted recording | The script for each slide is shown as a teleprompter; one audio clip is recorded per slide; the pipeline normalises the audio, trims silence and assembles the video | The real voice; the simplest to approve | Recording time; a changed slide must be re-recorded |
| B. Synthetic voice of the lecturer | A text-to-speech model trained on the lecturer's own recordings reads the updated scripts | A change to a slide is a change to its script only | Institutional policy on synthetic voices; students should be told; quality must be checked for technical terms |
| C. Hybrid | Option A for the main recordings, option B only for small corrections between recordings | Keeps most of the real voice | Two processes to maintain |

**Recommendations, whichever option is chosen**

1. **Store narration per slide.** Keep one audio file per slide, keyed to the slide's identifier, so that editing a slide only touches its own clip. Assemble the video with `ffmpeg` from the slide images and the clips.
2. **Preserve the speaking style in the scripts.** New and updated scripts should be drafted with the existing narration scripts in `reference/legacy_md/` as the style reference. Every script should be read and approved by the lecturer before it is recorded or synthesised.
3. **Pronunciation lexicon.** Keep a list of technical terms and how they are said (for example RoIAlign, atrous, ResNet, softmax) for option B, or as a checklist for option A.
4. **Accessibility.** Produce captions and transcripts directly from the approved scripts, rather than from speech recognition.
5. **Pilot.** Run the whole pipeline on one lesson, for example 3.6, before committing to the full course.

## 8. Quality, accessibility and maintenance (additional)

1. **Accessibility audit.** Check the pages against WCAG 2.2 AA:
   - **Images:** every figure needs a short alt text alongside its caption.
   - **Colour:** contrast must be sufficient, and no meaning should be carried by colour alone.
   - **Navigation:** the pages must be usable with a keyboard.
   - **Equations:** they must render correctly with screen readers.
2. **Automated checks.** The build already finds missing images, leftover directives and unrendered Markdown in captions. Extend this with:
   - a link checker;
   - a lint for the prose style (passive voice, no second person);
   - the rule that a worked-example type appears in only one lesson.

   Example types used in two lessons should be reviewed; `conv_param1` is in both 1.3 and 2.2.
3. **Numbering of the sources.** The source PDFs use the old numbering (for example `lesson_1_7_…` for the regularisation lesson, now 1.6). A mapping table in `COURSE_PLAN.md` would avoid confusion when the sources are consulted.
4. **Reproducibility.**
   - **Package versions:** pin `torch`, `torchvision` and the model-weight versions.
   - **Cache:** record in each cached result the versions it was computed with.
   - **GPU:** make the GPU visible in the dev container, so that recomputing the caches is quick.
5. **Student feedback.** Collect feedback on the pages during AY 2026/27, for example:
   - which self-check questions are most often answered wrongly;
   - which figures students find unclear.

   Use it to choose which lessons to revise first.
6. **Assessment alignment.** Check that the learning outcomes of each lesson are covered by the self-checks, the worked examples and the module assessment, and update the outcomes for any new content from item 2.

---

## Proposed timeline

AY 2027/28 is assumed to begin in September 2027.

| Period | Work |
|---|---|
| Nov 2026 – Jan 2027 | Item 1 (theme, size classes, contact sheet) and item 3 (two-module structure); collect student feedback (item 8.5) |
| Feb – Apr 2027 | Item 2 (modernisation) and item 4 (textbook readings); first mechanism notebooks (item 5) |
| May – Jun 2027 | Slide generation pilot (item 6) and narration pilot on one lesson (item 7); model notebooks |
| Jul – Aug 2027 | Roll out slides and narration across the module; accessibility audit; final build and publication |

## Decisions needed

- **Slide format:** HTML decks (Reveal.js or Quarto) or PowerPoint files.
- **Narration:** option A, B or C, and whether the institution permits a synthetic version of the lecturer's voice.
- **Notebooks and the app:** whether the notebooks replace the Streamlit app or sit alongside it.
- **Notation:** whether to adopt the textbook's notation across both modules.
- **Repository storage:** how cached results and images are stored as the second module is added.
