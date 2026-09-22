# CE6021 Machine Vision — Lesson Builder Course Plan

Status: **all 30 lessons built, site publishing ready** (18 Sep 2026).
`lessons/site/` holds the index plus 30 lesson pages and 198 figures; every
figure has been inspected by eye and every Streamlit demo runs headlessly.
`lessons/publish/publish_site.sh` publishes the student-facing site a few
lessons at a time from `lessons/publish/released.txt`.

Deliberately out of scope for now: the practice pages
(`practice_section{1,2,3}.html`), because students get the practice PDFs through
Brightspace. `about_examples.html` and `publish_examples.sh` for the public
interactive-examples repo are not written yet. The folder cleanup in section 6
below still needs your go-ahead.

### Publishing the student site

The student site is **generated output only**. The lesson markdown, the figure
engines, the build system and this repository's history are never copied into
the published repo, so the course cannot be rebuilt or re-themed from what is
published.

| Step | Command |
|---|---|
| Preview locally, push nothing | `lessons/publish/publish_site.sh --dry-run` |
| First publication | `lessons/publish/publish_site.sh --create` |
| Release more lessons later | edit `released.txt`, then `--update` |
| Remove the remote repo | `lessons/publish/publish_site.sh --delete` |

`released.txt` lists the lesson ids to publish. Unreleased lessons get no page
and no figures in the output, are skipped by the previous/next navigation, and
appear greyed out as "coming soon" on the index and in the section sidebar.
The build is driven by `build_site.py --released-file` and `--out`, so the
released subset is rendered fresh rather than filtered after the fact.

**The published site is always publicly readable.** Anyone with the URL can
open it, with no sign-in. A private Pages site, where viewers must
authenticate, requires GitHub Enterprise Cloud and an organization-owned
repository, so it is not available here.

The repository is created private by default, which only affects github.com:
no fork button, not clonable, not in GitHub search. That needs GitHub Pro, Team
or Enterprise, since GitHub Free requires a Pages repository to be public; the
script checks the account plan and stops with an explanation rather than
half-failing. The published site also carries a reserved-rights `LICENSE`, a
`robots.txt` that excludes it from search engines, and a `build-info.txt`
provenance stamp.

None of this prevents a third party from saving the rendered pages, which is
not technically possible for any public site. The substantive protections are
that only rendered output is published, so the course cannot be rebuilt or
extended from it, and that the licence reserves all rights. If genuine access
control is needed, the site is a self-contained folder of relative-path HTML
and can be hosted inside Brightspace instead, where students already
authenticate. KaTeX and Plotly are loaded from CDNs and would need to be
vendored locally first if the LMS blocks external scripts.

## 1. Purpose of this repository

This repo builds the **student-facing HTML lessons** for CE6021 and the
**Python figure engines** behind them. The assignment repo and the question
bank repo stay where they are; this repo only *borrows* code from them.

Two outputs are published from here:

| Output | Where it goes | Audience |
|---|---|---|
| `lessons/site/` — static HTML lessons with embedded figures | GitHub Pages (private course repo or Brightspace upload) | Students, read-only |
| `lessons/examples/` + `lessons/engines/` — runnable interactive examples | Public `CE6021_examples` repo (Codespaces / devcontainer, same pattern as `scripts/publish_lab.sh`) | Students, run and tweak |

The same engine module produces both: a `build()` function that writes the
static figures for the HTML, and a `demo()` function that runs the same
computation as a Streamlit page with sliders.

## 2. Source material (now removed from this repository)

These were the inputs used to write the lessons. They came from the
assignment repository, the question-bank repository and the original
slide PDFs, and were deleted from this repository in the 18 Sep 2026
tidy-up. The parts that are still needed were copied into `lessons/`
first: the question generators into `engines/_vendor/`, the images into
`data/`, the converted slide text into `reference/legacy_md/`, and the
PDF converter into `build/convert_pdfs.py`. Nothing under `lessons/`
imports from outside it.

| Source | Reused for |
|---|---|
| `reference/legacy_md/lesson_*_script.md` (30 narrated scripts) | Lesson prose: the scripts are slide-by-slide narration and become the body text of each lesson after light editing. |
| `reference/legacy_md/lesson_*_<topic>.md` (30 slide decks) | Section/figure structure. Figures in the decks are screenshots; every one becomes a generated figure. |
| `question_tool/generators/*.py` (57 engines) | Numerical worked examples. The `generate_params()` / `compute()` pairs are reused directly; only the `render()` markdown layer is replaced. Six already draw matplotlib figures (`bi_lin`, `hough`, `hausdorff`, `iou`, `sift_bin_lin`, plus `_morph_utils` grids). |
| `question_bank/papers/practice_*.pdf`, `examples_*.pdf` | Linked from each section's "Practice" page as-is. |
| `conversions/Grasple_*.md` (classical topics only) | Per-lesson "check your understanding" MCQs. The deep-learning Grasple sets (CNNs, ViT, generative, DL object detection, transpose conv) belong to the DL module and are out of scope. |
| `Assignments/lab0X_sol/app_lab0X_sol.py` | The Streamlit apps are the model for the `demo()` pattern only. Lab solution code is **not** reused as demo content (see "Relationship to the labs" below). |
| `Assignments/helpers/` | `dataloader`, `plotting.save_html`, Olivetti downloader. |
| `Data/` | Test images that are not tied to a lab task (Sphinx, Sacré-Cœur, sunflowers). |

### Relationship to the labs

The labs ask students to implement an algorithm from scratch and evaluate it
on a specific image. Lesson demos must not hand them that work. Rules:

1. **No demo reproduces a lab pipeline.** Excluded outright: the MRI k-space
   notch filter (Lab 02), fluid-level Hough detection (Lab 03), watershed on
   the coins image (Lab 04), Harris on the checkerboard (Lab 05), LoG blob
   detection on the sunflowers (Lab 06), the HOG ship detector on the Kaggle
   satellite set (Lab 07), and the Eigenface enrol/authenticate/FAR-FRR
   engine (Lab 08). The chirp aliasing demo of Lab 00B and the efficient
   separable convolution of Lab 01 are also not repeated.
2. **Where a lesson must show an algorithm that a lab implements, the demo
   uses a library call** (`cv2`, `skimage`, `sklearn`) on a **different
   image**, and shows the algorithm's *behaviour* (parameter effects,
   intermediate outputs) rather than an implementation students could copy.
3. **Lab data files are not copied into `lessons/data/`**: no MRI k-space
   file, bottle image, coins, checkerboard, sunflowers, SF Bay scene, or
   shipsnet. Lesson images come from the slides' own examples, skimage's
   bundled data (`astronaut`, `camera`, `coffee`, `brick`, `cell`, `text`),
   or synthetic patterns generated in the engine.
4. From-scratch code in engines is limited to small didactic pieces that
   the lessons already spell out on the slides (a 3×3 convolution stepper,
   a 9×9 morphology grid, an integral-image table), never a lab class.

### Image attribution

Every photograph or third-party image that appears in a figure is credited,
on the page and in the figure caption. Rules:

1. **Source images come from Wikimedia Commons** (or are generated
   synthetically by the engine). Each is listed in `data/sources.yaml` by its
   Commons file title; `build/fetch_images.py` downloads a scaled copy and
   records title, author, licence and source URL in `data/credits.yaml`
   straight from the Commons API, so credits are never typed by hand.
2. **Attributions from the old slides are carried forward**: where a slide
   image had a Wikimedia credit, the same file is fetched and used again.
3. **Slide images without an attribution are replaced** by a Commons image
   with one (or by a synthetic figure). Images of unknown provenance,
   including scikit-image samples whose origin is not documented, are not
   used.
4. **Enforcement in code**: engines load images only by credit key
   (`load_image("lion")`), which fails if the key has no credit; every
   `Figure` automatically records the keys loaded while it was built; the
   site builder appends "Image: title, author, licence" to the caption and an
   "Image credits" list at the foot of each page.
5. Two images from the lab data set (Sphinx, Sacré-Cœur) are themselves
   Commons files and are fetched by key like everything else.

## 3. Target directory layout

```
lessons/
  COURSE_PLAN.md            this file
  README.md                 how to build and publish
                            (dependencies live in .devcontainer/requirements.txt)
  reference/legacy_md/      converted old slides + scripts (read-only input)
  data/                     credited Wikimedia Commons images the engines use
  engines/                  one package per section, one module per figure group
    __init__.py
    _common.py              image loading, colour palette, fig -> png/html export
    _vendor/                compute() functions vendored from the question bank
    s1_images/  s2_edges/  s3_features/  s4_detection/  s5_faces/  s6_bof/
  content/                  lesson sources, one markdown file per lesson
    01_01_digital_images.md ...
    _selfcheck/             MCQ yaml per lesson (from Grasple exports)
  build/
    build_site.py           run engines -> render markdown -> write site/
    template.html.j2        page shell (nav, KaTeX, figure + self-check widgets)
    style.css
  examples/
    app.py                  Streamlit multipage app: one page per engine demo()
  site/                     GENERATED (gitignored except when publishing)
    index.html, 01_01_digital_images.html, ..., figures/, practice/
  publish/
    publish_examples.sh     push engines/ examples/ data/ .devcontainer to public repo
    publish_site.sh         push site/ to gh-pages
```

### Engine contract

```python
# lessons/engines/s1_images/convolution.py
FIGURES = ["conv1d_slide", "conv2d_flip_slide", "padding_modes"]

def build(out_dir: Path, ctx: BuildContext) -> list[Figure]:
    """Write static assets; return Figure(id, caption, png_path, html_path|None)."""

def demo() -> None:
    """Streamlit page with sliders; imports the same helper functions as build()."""
```

Figures are produced with Plotly where interaction helps (sliders, hover on
matrices, 3D surfaces) and exported twice: PNG via kaleido for the page body,
and a self-contained HTML div for an "interactive" toggle. Matplotlib is used
where a plain static image is enough. Both routes go through
`_common.export()` so page code never touches a plotting library.

### Content file directives

Markdown with front matter and two custom fences:

```
---
title: Image Processing with Convolution
section: 1
lesson: 2
engines: [s1_images.convolution]
practice: [conv1d, conv2d, conv_arith1, conv_arith2, conv_arith3, conv_arith4]
---
{{fig conv2d_flip_slide}}          -> figure + caption + interactive toggle
{{example conv2d seed=7}}           -> worked numerical example from vendored generator
{{selfcheck 01_02}}                 -> MCQ block from content/_selfcheck/01_02.yaml
```

## 4. Lesson map

Thirty lessons keep the numbering of the existing videos so students, the
practice papers and the exam configs (`ce6021_section1..3.json`) line up.
Each row lists the HTML file, the engines to write, what each engine reuses,
and the question-bank generators that become the lesson's worked examples.

### Section 1 — Digital images and filtering

| # | HTML | Engines (figures / interactive) | Reuse | Worked examples |
|---|---|---|---|---|
| 1.1 | `01_01_digital_images.html` | `bayer_demosaic` (CFA pattern, channel toggle); `sample_quantise` (pixel-grid and bit-depth sliders on a photo); `rgb_colour_space` (3D RGB cube, additive mixing) | Data images | — |
| 1.2 | `01_02_convolution.html` | `conv1d_slide` (kernel flip and shift table, step through n); `conv2d_flip_slide` (pad, flip, slide with running sum); `padding_modes` (valid/same/full output-size calculator) | `conv1d`, `conv2d`, `conv_arith1-6` compute | conv1d, conv2d, conv_arith1–6 |
| 1.3 | `01_03_sampling_interpolation.html` | `aliasing_1d` (sinusoid above fs/2, mirrored spectrum); `image_aliasing` (skimage `brick` and a striped pattern downsampled by nearest-neighbour vs `cv2.resize` with INTER_AREA, moiré visible); `interp_kernels` (box/triangle/sinc and their frequency responses); `bilinear_interp` (four-pixel area diagram); `interp_compare` (nearest/bilinear/bicubic/Lanczos on an edge crop, Gibbs ringing) | `bi_lin` plot | bi_lin |
| 1.4 | `01_04_linear_nonlinear_filters.html` | `gaussian_kernel` (1D/2D sampled Gaussian, 6σ+1 rule, blur ladder); `separable_kernel` (outer product of two 1D kernels and the 3×3 two-pass worked example from the slide, via `scipy.signal.convolve2d` only, no timing or implementation); `filter_properties` (superposition, shift invariance, commutativity side-by-side); `image_noise` (Gaussian and salt-and-pepper PDFs and samples); `median_bilateral` (`cv2.GaussianBlur` vs `cv2.medianBlur` vs `cv2.bilateralFilter` on noisy image) | `sep_conv`, `med_filt` compute | sep_conv, med_filt |
| 1.5 | `01_05_morphology.html` | `morph_ops` (erosion/dilation/opening/closing on 9×9 grids with fit/hit/miss highlight); `distance_transform` (L1 vs L2 on a small binary grid and on two overlapping synthetic discs, `scipy.ndimage.distance_transform_edt`); `connected_components` (row-scan labelling stepper with equivalence table); `grey_morphology` | `_morph_utils`, `morph_opr1-4` | morph_opr1–4 |
| 1.6 | `01_06_fourier_1d.html` | `sinusoid_params` (A, ω, φ sliders); `complex_phasor` (rotating e^{jωt}, positive/negative frequency); `fourier_series_square` (partial sums, Gibbs overshoot); `dft_16` (16-sample DFT, mirror bins, leakage, Hann window); `conv_theorem_1d` (brick-wall LP in spectrum vs kernel convolution) | — | Grasple Fourier MCQs |
| 1.7 | `01_07_fourier_2d.html` | `dft_basis_2d` (corrugation grid over k, l); `spectrum_2d` (fftshift, log magnitude, windowed crosshatch); `magnitude_phase_swap`; `common_dfts` (square/Gaussian/circle and transforms, 3D); `freq_filter` (ideal circle LP/HP vs Gaussian, ringing); `periodic_noise` (diagonal sinusoidal interference added to a photo, spectrum peaks shown, notch applied in the spatial-image DFT, not k-space) | — | — |

### Section 2 — Edges, corners and segmentation

| # | HTML | Engines | Reuse | Worked examples |
|---|---|---|---|---|
| 2.1 | `02_01_edge_detection.html` | `edge_profile` (step edge, first and second derivative); `prewitt_sobel` (kernels, Gx, Gy, magnitude, direction, offset vs magnitude display); `gaussian_derivative` (kernels and multi-σ edge maps); `laplacian_zero_crossing` | `grad_dir` compute | grad_dir |
| 2.2 | `02_02_canny_hough.html` | `canny_stages` (blur → Sobel → NMS → hysteresis, each stage toggled); `nms_patch` (3×3 gradient-direction neighbour check); `hough_lines` (points ↔ sinusoids in (ρ,θ), accumulator heat map, `cv2.HoughLines` on a building or the kitchen example, threshold slider) | `canny_nms`, `hys_thresh`, `hough` plot | canny_nms, hys_thresh, hough |
| 2.3 | `02_03_harris_corners.html` | `shift_energy` (E(x,y) surface for flat/edge/corner patches); `second_moment_ellipse` (eigenvalue ellipses drawn on image points); `harris_response` (`skimage.feature.corner_harris` R map on the "moo" pattern and a building photo, k and σ sliders, rotation invariance shown by rotating the image) | `harris` compute | harris |
| 2.4 | `02_04_region_segmentation.html` | `threshold_hist` (histogram with draggable threshold); `otsu_curve` (between-class variance vs threshold); `image_surface` (grey image as 3D relief); `watershed_flood` (flood-level slider on a 1D profile and on skimage `cell`/`coffee`, `skimage.segmentation.watershed`, markers on/off to show over-segmentation); `graph_merge` (skimage Felzenszwalb, k slider) | `otsu_thresh` compute | otsu_thresh |
| 2.5 | `02_05_clustering_segmentation.html` | `kmeans_2d` (iteration stepper); `kmeans_colour` (k slider on image); `parzen_density` (histogram vs KDE, bandwidth); `mean_shift_path` (window trajectory to mode); `mean_shift_segmentation` (sklearn, Luv and Luv+xy) | `kmeans_iter` compute | kmeans_iter |

### Section 3 — Keypoints, scale space and descriptors

| # | HTML | Engines | Reuse | Worked examples |
|---|---|---|---|---|
| 3.1 | `03_01_keypoints_log.html` | `log_kernels` (σ family, 2D and 3D); `log_blob_response` (synthetic dots of three radii convolved with one LoG kernel via `scipy.ndimage.gaussian_laplace`, normalised square response heat map); `harris_not_scale_invariant` (balloon example, `skimage.feature.corner_harris`) | — | Grasple feature-detection MCQs |
| 3.2 | `03_02_scale_space.html` | `gaussian_scale_space` (σ slider); `characteristic_scale` (normalised Laplacian vs σ at a chosen point); `scale_invariance` (same feature at two zooms); `maxima_26` (3D neighbourhood diagram); `harris_laplacian`; `dog_vs_log` (approximation error vs ρ) — scale-space extrema shown with `skimage.feature.blob_log` on skimage `hubble_deep_field`, no from-scratch 26-neighbour search | — | — |
| 3.3 | `03_03_descriptors_matching.html` | `gradient_cell_histogram` (8×8 arrows → 4 histograms → 32-vector); `characteristic_orientation` (rotated patch normalised); `bf_matching` (OpenCV SIFT, ratio test, scaled image) | — | — |
| 3.4 | `03_04_sift.html` | `sift_pyramid` (octaves/scales diagram + real images); `dog_extrema`; `quadratic_localisation` (parabola fit on grid); `orientation_histogram`; `descriptor_bilinear` (4 nearest histograms); `sift_end_to_end` | `sift_orient`, `sift_bin_lin` | sift_bin_lin, sift_orient |
| 3.5 | `03_05_image_pyramids.html` | `gaussian_pyramid` (successive blur + subsample); `sigma_composition` (√(σ1²+σ2²) check); `crowley_pyramid` | — | — |

### Section 4 — Object detection and classification

| # | HTML | Engines | Reuse | Worked examples |
|---|---|---|---|---|
| 4.1 | `04_01_intro_object_detection.html` | `sliding_window` (window animation, pyramid levels); `iou_boxes` (two draggable boxes); `det_metrics` (TP/FP/FN counters → precision, recall) | `iou` plot, `objdet_pre`, `con_mat` compute | iou, objdet_pre |
| 4.2 | `04_02_machine_learning.html` | `boundary_complexity` (under/over/optimal fit on 2D data); `bias_variance_curve`; `train_val_curves`; `kfold_diagram`; `raw_pixel_features` (sklearn digits) | — | — |
| 4.3 | `04_03_classifiers.html` | `knn_boundary` (k slider); `svm_margin` (C slider, support vectors); `rbf_lift` (2D → 3D kernel trick); `naive_bayes_dot` (pedestrian vs road-sign numbers) | `con_mat` | con_mat |
| 4.4 | `04_04_template_matching.html` | `ncc_heatmap` (Euclidean vs correlation vs NCC); `chamfer` (edges → DT → correlation minimum); `hausdorff_shapes` (directed distances step-through) | `hausdorff` plot | hausdorff |
| 4.5 | `04_05_viola_jones.html` | `haar_types` (scaled/shifted features in 24×24 window); `integral_image` (column then row accumulation, 4-corner sum); `adaboost_2d` (weak learners combining); `cascade_diagram`; `opencv_face` | `int_img`, `haar_feature` compute | int_img, haar_feature |
| 4.6 | `04_06_hog.html` | `hog_cell_vote` (bilinear bin voting on one cell); `hog_descriptor_length` (cells/blocks/strides calculator); `hog_visual` (`skimage.feature.hog` with `visualize=True` on the astronaut image); `hog_pedestrians` (OpenCV pretrained `HOGDescriptor_getDefaultPeopleDetector`, `detectMultiScale` with pyramid scale slider, matching the slide's pyramid-scale comparison) | `hog_bilin`, `hog_calc` | hog_calc, hog_bilin |

### Section 5 — Facial recognition

| # | HTML | Engines | Reuse | Worked examples |
|---|---|---|---|---|
| 5.1 | `05_01_intro_face_recognition.html` | `olivetti_gallery` (identities × poses); `fr_pipeline` (SVG diagram); `identification_vs_verification` | `download_olivetti_dataset` | — |
| 5.2 | `05_02_pca.html` | `curse_of_dimensionality` (sparsity 1D→3D, error vs features); `covariance_shapes` (four distributions and matrices); `pca_2d` (eigenvectors overlay, transform, 1D projection); `pca_patch_compression` (k slider) | `pca_project` compute | pca_project, Grasple eigenvector MCQs |
| 5.3 | `05_03_eigenfaces.html` | `eigenfaces_gallery` (mean face, top-k); `face_reconstruction` (k slider, `sklearn.decomposition.PCA`); `weight_space` (two faces of one identity and one other plotted on the first two eigenface axes) — no enrol/authenticate pipeline and no FAR/FRR or ROC, which are Lab 08 | `download_olivetti_dataset` helper only | — |
| 5.4 | `05_04_lda.html` | `pca_vs_lda_line` (2-class 2D); `projection_objective` (line angle slider → transformed means, scatter, J(v)); `lda_solution` (Sw⁻¹(μ1−μ2) arrow) | — | — |
| 5.5 | `05_05_fisherfaces.html` | `multiclass_lda` (3 classes); `fisherfaces_pipeline` (PCA → LDA on Olivetti, k slider, condition number readout); `eigen_vs_fisher` (sklearn PCA vs PCA+LDA nearest-neighbour accuracy on Olivetti with synthetic illumination gradients added) | — | — |

### Section 6 — Bag of features

| # | HTML | Engines | Reuse | Worked examples |
|---|---|---|---|---|
| 6.1 | `06_01_bag_of_features.html` | `bag_of_words`, `tf_idf` (toy documents → weighted vectors); `bof_algorithm`; `keypoint_vs_dense` (two views of one building); `descriptor_clustering` (real SIFT descriptors → k-means); `visual_vocabulary` (codeword patch gallery + document-frequency curve); `bof_histogram` (three images → term histograms) | — | — |
| 6.2 | `06_02_bof_classification_retrieval.html` | `bof_classification` (pipeline + measured leave-one-out SVM accuracy); `spatial_pyramid` (grid levels, weights, vector length); `retrieval_task` (query → ranked results); `inverted_index` (word → image table with a query lookup); `vocab_tree` (hierarchical k-means + search cost); `bof_summary` | — | — |

Note: the plan originally called for `Data/scenes` in 6.2. That set does not
exist in this repo, so both Section 6 lessons build their vocabulary, classifier
and retrieval database from the 36 credited Wikimedia Commons images already in
`lessons/data/`, with queries made by cropping, rotating, rescaling and
brightening database images. All numbers quoted in those figures are measured at
build time, and the figures say plainly that the dataset is too small to be a
benchmark.

### Site-level pages

| HTML | Content |
|---|---|
| `index.html` | Course map: six sections, lesson list, links to labs repo and examples repo |
| `practice_section{1,2,3}.html` | Links to `practice_*.pdf` / `examples_*.pdf`, list of generator types per section |
| `about_examples.html` | How to open the examples repo in Codespaces and run `streamlit run examples/app.py` |

Engine count: 96 figure engines across 30 lessons. Sections 1–2 are the
priority because they carry most of the exam question types.

## 5. Build order

1. **Scaffold** `engines/_common.py`, `build/build_site.py`, template, `examples/app.py`, `requirements.txt`. Prove the round trip with one lesson (1.2 convolution) end to end: engine → PNG + interactive HTML → page → Streamlit page.
2. **Vendor** the compute functions from `question_tool/generators` into `engines/_vendor/` (keep the original filenames; strip `render()`). Copy the needed images into `lessons/data/`.
3. **Sections 1–2** (12 lessons) — prose from scripts, all engines, self-checks from Grasple exports.
4. **Sections 3–4** (11 lessons).
5. **Sections 5–6** (7 lessons).
6. **Publish scripts** and the public examples repo; gh-pages for `site/`.

## 6. Cleanup of old folders (done, 18 Sep 2026)

The repository now contains only `.devcontainer/`, `.gitignore` and `lessons/`.
Everything below was removed after checking that nothing under `lessons/`
imports or reads from it.

| Removed | Was tracked? | Note |
|---|---|---|
| `old_lesson_repo/` (60 PDFs) | no | converted text kept in `lessons/reference/legacy_md/` |
| `pdfs/`, `conversions/` | no | Grasple MCQs migrated to `content/_selfcheck/`; deep-learning sets dropped |
| `question_tool/`, `question_bank/`, `app.py` | no | compute functions already vendored into `engines/_vendor/` |
| `grader_tool/` | no | its `convert_pdfs` command extracted to `lessons/build/convert_pdfs.py`; student data lives in a separate repository |
| `Assignments/`, `scripts/` | yes | recoverable from git history; no helper was ever imported by `lessons/` |
| `Data/` | yes | recoverable from git history; `lessons/data/` holds credited Commons images fetched independently |
| `.devcontainer/` | yes | kept; `requirements.txt` replaced by the lesson build dependencies |

`Data/` was 667 MB and remains in the git history, so the `.git` directory is
still large. Shrinking it would need a history rewrite, which has not been done.
