# CE5021 Deep Learning for Computer Vision — Lesson Builder Course Plan

Status: **Sections 1 and 2 complete (11 lessons); Section 3's 6 to go** (2 Oct 2026).
`lessons/site/` holds the index, 11 lesson pages and 77 figures. Section 3's
lessons appear greyed out as "coming soon" on the index. A full clean build
takes about 1 min 45 s, most of it lesson 2.5's image optimisations.

This repository was inherited from the CE6021 Machine Vision lesson builder.
The build system, figure-engine contract, page templates, publishing scripts
and image-credit machinery are reused unchanged; the *course content* is
replaced. The CE6021 state is preserved in the repository's first commit and
can be recovered from there at any time.

### What has already been done

| Step | State |
|---|---|
| 35 CE5021 lecture PDFs converted to markdown in `reference/legacy_md/` | done |
| Question-bank repo cloned to `autoquiz/` (gitignored) | done |
| 22 deep-learning generators vendored into `engines/_vendor/` | done, all 22 render |
| CE6021 baseline captured as the initial commit | done |
| Lesson map and engine list (this document) | done |
| 27 self-check MCQs recovered from the Grasple exports | done |
| CE6021 content and engines retired; `s1_convnets/` created | done |
| **Section 1, all six lessons** — prose, 39 figures, 18 worked examples, 36 self-check questions | done |
| **Section 2, all five lessons** — prose, 38 figures, 6 worked examples, 32 self-check questions | done |
| Shared infrastructure: `_arch.py`, `_tinynet.py`, `_fixednet.py`, `_volumes.py`, `build/selfcheck_tools.py` | done (section 10) |
| Section 3 (6 lessons) | **not started** |

### Decisions taken

1. **CE6021 content is replaced, not kept alongside.** `content/`, `engines/s1_*`
   … `s6_*` and `content/_selfcheck/` are rewritten for CE5021. The baseline
   commit is the archive.
2. **No deep-learning framework is added.** Figures are NumPy + matplotlib +
   Plotly, as now. Everything CE5021 needs to *show* — convolution arithmetic,
   receptive fields, transpose and dilated convolution, batch norm, dropout,
   LR schedules, loss curves, PR/AP curves, IoU, anchor boxes, NMS — is exactly
   computable without one. Published architectures (AlexNet, VGG, Inception,
   ResNet, DenseNet, U-Net, SSD, FPN, Mask R-CNN) become schematic block
   diagrams whose parameter counts, FLOPs, memory and output sizes are
   **computed at build time from the layer specification**, not typed in.
   See "Why no PyTorch" below for what this costs.
3. **Lesson 1.6 does not exist** in the source material and 1.7 is renumbered
   to 1.6, so Section 1 is a contiguous 1.1–1.6.
4. **Lesson 3.7 (Pose Estimation) is dropped** — no longer on the syllabus. Its
   converted script remains in `reference/legacy_md/lesson_3_7_script.md` and is
   simply unused. There is no slide deck for it in any case.

Seventeen lessons: 6 + 5 + 6.

---

## 1. Purpose of this repository

This repo builds the **student-facing HTML lessons** for CE5021 and the
**Python figure engines** behind them. The question-bank repo
(`github.com/tonyscan6003/autoquiz`) stays where it is; this repo only
*borrows* code from it.

Two outputs are published from here:

| Output | Where it goes | Audience |
|---|---|---|
| `lessons/site/` — static HTML lessons with embedded figures | GitHub Pages (private course repo or Brightspace upload) | Students, read-only |
| `lessons/examples/` + `lessons/engines/` — runnable interactive examples | Public `CE5021_examples` repo (Codespaces / devcontainer) | Students, run and tweak |

The same engine module produces both: a `build()` function that writes the
static figures for the HTML, and a `demo()` function that runs the same
computation as a Streamlit page with sliders.

## 2. Source material

### Lecture PDFs → `reference/legacy_md/`

The 35 PDFs in `reference/pdf_for_conversion/` (17 slide decks, 18 narration
scripts) were converted with:

```bash
python lessons/build/convert_pdfs.py \
    --input-dir lessons/reference/pdf_for_conversion \
    --output-dir lessons/reference/legacy_md --force
```

Both the PDFs and their `:Zone.Identifier` sidecars are gitignored; the
converted markdown is tracked. As in CE6021, the **narration scripts become the
lesson body text after light editing**, and the **slide decks give the section
and figure structure**. Every screenshot in a deck becomes a generated figure.

Note the source files keep the original 1.7 / 3.7 numbering. The mapping to
lesson ids is in the lesson map below.

### Question bank → `engines/_vendor/`

`autoquiz/` is a temporary working copy, gitignored, recloneable with:

```bash
git clone https://github.com/tonyscan6003/autoquiz.git autoquiz
```

Twenty-two deep-learning generators are vendored, verbatim except for the
`..utils` → `._utils` import redirect (the re-vendoring one-liner is in
`engines/_vendor/__init__.py`). None of them draws a figure, so they are pure
markdown producers — simpler than the CE6021 set, six of which plotted.

`engines/_vendor/__init__.py` now splits `TYPES` into `DL_TYPES` (the 22 below)
and `CV_TYPES` (the 33 classical ones kept from CE6021, of which CE5021 uses
the `conv_arith*`, `conv1d`, `conv2d`, `iou`, `con_mat` and `objdet_pre`
families).

| Generator | Worked example | Lesson |
|---|---|---|
| `single_neu` | Single neuron activation | 1.2 |
| `softmax_act` | Softmax activation | 1.2 |
| `fc_param` | Fully connected layer parameters | 1.2 |
| `conv_param1` | Conv layer parameters, no padding, stride > 1 | 1.3 |
| `conv_param2` | Conv layer parameters, padding, stride 1 | 1.3 |
| `mult_kernel_conv` | Multi-channel convolution | 1.3 |
| `bottleneck_conv` | Bottleneck (1×1) convolution | 1.3 |
| `max_pool` | Max and average pooling | 1.1 |
| `rfield` | Receptive field calculation | 1.4 |
| `cross_ent` | Cross-entropy loss | 1.5 |
| `lr_schedule` | Step-decay learning rate | 1.5 |
| `lr_schedule_cos` | Cosine annealing schedule | 1.5 |
| `dropout` | Inverted dropout | 1.6 |
| `con_mat` | Confusion matrix | 2.1 |
| `global_avg_pool` | Global average pooling | 2.2 |
| `inception_block` | Inception block parameter count | 2.2 |
| `bat_norm1` | Batch normalisation | 2.3 |
| `bat_norm2` | Batch norm global stats and inference | 2.3 |
| `iou` | Intersection over union | 3.1 |
| `objdet_pre` | Object detection precision/recall | 3.1 |
| `depthwise_sep_conv` | Depthwise separable convolution | 3.4 |
| `mat_conv` | Matrix method convolution | 3.5 |
| `transpose_conv1` | Transpose convolution, stride 1 | 3.5 |
| `transpose_conv2` | Transpose convolution, stride 2 | 3.5 |
| `dilated_conv` | Dilated convolution | 3.6 |

Practice and examples PDFs for these sit in `autoquiz/question_bank/papers/`
as `practice_week8_9.pdf`, `practice_week10_11.pdf`, `practice_week12_13.pdf`
and the matching `examples_*.pdf`, driven by the `week_08_09`, `week_10_11` and
`week_12_13` configs. Students get these through Brightspace, so the practice
pages stay out of scope here, as in CE6021.

### Self-check MCQs — written fresh

The Grasple exports are in `reference/Grasple_*/`. All 18 sets were surveyed.
They hold two kinds of question: **numeric/matrix entry**, which the autoquiz
generators already cover as worked examples and which yields nothing here, and
**true MCQs**, which is what `_selfcheck/*.yaml` needs.

The deep-learning MCQ yield is 27 questions covering 8 lessons:

| Set | Questions | Lessons |
|---|---|---|
| `Grasple_convolutional_neural_networks` | 3 | 1.5, 1.6 |
| `Grasple_image_cls_deep_cnn` | 20 | 2.1, 2.2, 2.3, 2.4, and the bottleneck one in 1.3 |
| `Grasple_object_detection` | 4 | 3.2, 3.4 (sliding window, image pyramid, NMS) |

Three caveats, each recorded in the affected file: some source questions marked
**two** options 100% correct, which the single-answer widget cannot express, so
those are rewritten to combine both into one option; one question lost its
options in PDF conversion and they are reconstructed from its parallel
question; and option order is shuffled deterministically per file so the right
answer is not always first.

Some lessons have no MCQ source and are authored from the narration scripts;
others have too few and are topped up the same way. (`Grasple_segmentation`
turns out to be classical watershed segmentation, not semantic segmentation, so
it does not help 3.5; `Grasple_obj_det_deep_learning` and
`Grasple_transpose_dilated_conv` are entirely numeric.) **Sections 1 and 2 are
now complete at 6 questions per lesson (8 for 2.2).** Section 3 remains: 3.1,
3.3, 3.5 and 3.6 have no source, and 3.2 (3) and 3.4 (1) need topping up. Each
file's header records which questions are from Grasple and which authored.
Always write these files through `build/selfcheck_tools.py` (section 10).

### Images

`data/` already holds ~50 credited Wikimedia Commons photographs with
`sources.yaml` / `credits.yaml` and `build/fetch_images.py`. Most are still
useful — CE5021 needs photographs to convolve, to classify, to draw bounding
boxes on and to segment, and the existing set (animals, street scenes, crowds,
vehicles, buildings) covers all four. The attribution rules from CE6021 are
unchanged and still enforced in code: engines load by credit key, `Figure`
records the keys used, the builder prints the credit under the caption.

Additions expected: a multi-object street or market scene for detection
figures, and one scene with clear stuff/thing regions for segmentation. Add
them to `sources.yaml` and run `build/fetch_images.py`.

### Relationship to the labs

CE5021's labs are framework based (building and training networks). The CE6021
rule still applies in spirit and is simpler here: **lesson figures never train a
network, so no demo can hand a student their lab.** Engines compute and draw;
they do not fit. Where a lesson must show a training curve, the curve is either
(a) the analytic behaviour being described (a bias–variance U, a cosine LR
schedule, a focal-loss reweighting), or (b) a small synthetic 2D optimisation
run in NumPy, clearly labelled as a toy.

---

## 3. Directory layout

```
lessons/
  COURSE_PLAN.md            this file
  README.md                 how to build and publish
  reference/
    pdf_for_conversion/     the 35 source PDFs (gitignored, input only)
    legacy_md/              converted slides + scripts (tracked, read-only input)
  data/                     credited Wikimedia Commons images the engines use
  engines/
    __init__.py
    _common.py              image loading, colour palette, fig -> png/html export
    _vendor/                compute()/render() vendored from autoquiz
    s1_convnets/  s2_classification/  s3_detection/
  content/
    01_01_convolution.md ...
    _course.yaml            section and lesson index
    _selfcheck/             MCQ yaml per lesson
  build/
    build_site.py  convert_pdfs.py  fetch_images.py
    template.html.j2  index.html.j2  style.css  site.js
  examples/app.py           Streamlit multipage app, one page per engine demo()
  site/                     GENERATED (gitignored except when publishing)
  publish/
    publish_site.sh  released.txt
autoquiz/                   GITIGNORED working copy of the question-bank repo
```

Three engine packages rather than six, matching the three course sections.

### Engine contract (unchanged)

```python
# lessons/engines/s1_convnets/convolution.py
FIGURES = ["conv2d_slide", "padding_modes", "stride_grid", "pooling_ops"]

def build(ctx: BuildContext) -> list[Figure]:
    """Write static assets; return Figure(id, caption, png, html|None, kind)."""

def demo() -> None:
    """Streamlit page with sliders; imports the same helpers as build()."""
```

Plotly where interaction helps (sliders over stride/dilation/σ, hover on
feature-map tensors, 3D volumes), matplotlib where a static image is enough;
both via `_common`, so page code never touches a plotting library.

### Content file directives (unchanged)

```
---
id: "01_03"
title: Advanced convolution
section: 1
engines: [s1_convnets.advanced_convolution]
practice: [conv_param1, conv_param2, mult_kernel_conv, bottleneck_conv]
summary: ...
---
{{fig multi_channel_conv}}      -> figure + caption + interactive toggle
{{example conv_param1 seed=5}}  -> worked example from the vendored generator
{{selfcheck 01_03}}             -> MCQ block from content/_selfcheck/01_03.yaml
```

---

## 4. Lesson map

Seventeen lessons. Each row gives the lesson id and page, the source files in
`reference/legacy_md/`, the figure engines to write, and the worked examples.

### Section 1 — Convolution and neural network fundamentals

Package `engines/s1_convnets/` — **built**. Module names as written:
`convolution` (1.1), `fc_layers` (1.2), `advanced_convolution` (1.3),
`cnns` (1.4), `training` (1.5), `regularisation` (1.6).

| # | Page / source | Engines (figures) | Worked examples |
|---|---|---|---|
| 1.1 | `01_01.html` — 2D convolution<br>`lesson_1_1_convolution`, `lesson_1_1_script` | `conv2d_flip_slide` (flip, pad, slide, running sum on a small matrix); `padding_modes` (valid/same/full calculator, output-size formula); `stride_grid` (stride 1/2/3 over one input, output size and the floor in the formula); `pooling_ops` (max vs average pool, window and stride sliders, on a small grid and a photo crop); `corr_vs_conv` (symmetric kernel ⇒ identical, asymmetric ⇒ not) | `conv2d`, `conv_arith1`, `conv_arith3`, `conv_arith4`, `max_pool` |
| 1.2 | `01_02.html` — Fully connected layers and classifiers<br>`lesson_1_2_FC_layers`, `lesson_1_2_script` | `single_neuron` (weights, bias, sum, activation, with numbers flowing through); `activation_functions` (sigmoid, tanh, ReLU, leaky ReLU and their derivatives; saturation regions shaded); `fc_network` (one-hidden-layer graph, weight count annotated per layer); `binary_classifier` (2D toy data, sigmoid output surface, decision boundary); `softmax_output` (logits → probabilities bar chart, temperature slider, sum-to-one); `fc_param_count` (layer table → parameter total) | `single_neu`, `softmax_act`, `fc_param` |
| 1.3 | `01_03.html` — Advanced convolution<br>`lesson_1_3_advanced_convolution`, `lesson_1_3_script` | `multi_channel_conv` (RGB input, 3-channel kernel, per-channel 2D convolutions summed to one feature map — the slide 4 worked example drawn with its numbers); `multi_kernel_conv` (C-channel input, N kernels, N parallel feature maps); `conv_volumes` (3D volume diagram, input/kernel/output shapes linked to the size formula); `conv_cost` (parameters, MACs and activation memory vs C, N, k for one layer); `bottleneck_1x1` (1×1 projection before/after a 3×3, parameter saving computed) | `mult_kernel_conv`, `conv_param1`, `conv_param2`, `bottleneck_conv` |
| 1.4 | `01_04.html` — Convolutional neural networks<br>`lesson_1_4_cnns`, `lesson_1_4_script` | `fc_on_images` (16-pixel image → vector → dense layer; parameter explosion at 256×256 plotted against image size); `conv_as_nn` (the same convolution drawn as a sparse, weight-shared dense layer — the two connection diagrams side by side with parameter counts); `cnn_structure` (conv/act/pool stack → flatten → dense → softmax, tensor shape printed at every stage); `learned_features` (hierarchy of edge → texture → part → object, drawn from Gabor-like and synthetic templates, not from a trained net); `receptive_field` (layer-by-layer growth, the r formula stepped through, shaded on an image); `translation_invariance` (object shifted, feature map shifts with it, pooling collapses the shift) | `rfield`, `conv_arith2` |
| 1.5 | `01_05.html` — Training neural networks<br>`lesson_1_5_training_nn`, `lesson_1_5_script` | `supervised_pipeline` (classical features→classifier vs end-to-end DL, side by side); `loss_functions` (binary and categorical cross-entropy surfaces and curves vs predicted probability); `gradient_descent_2d` (contour plot with the descent path, learning-rate slider showing divergence, crawling and a good value); `backprop_chain` (tiny 2-layer net, forward values and backward gradients annotated on the graph); `batch_size_noise` (full-batch vs mini-batch vs SGD trajectories on the same surface); `momentum_path` (plain SGD vs momentum in a ravine); `lr_schedules` (step, exponential and cosine schedules over epochs) | `cross_ent`, `lr_schedule`, `lr_schedule_cos` |
| 1.6 | `01_06.html` — Overfitting and regularisation<br>`lesson_1_7_overfitting_regularisation`, `lesson_1_7_script` | `bias_variance` (U-shaped total error with bias and variance components; under/optimal/over decision boundaries on one 2D set); `train_val_curves` (the three cases from slide 4: underfit, overfit divergence, good fit); `curse_of_dimensionality` (sample sparsity 1D→3D, required samples vs dimension); `l1_l2_regularisation` (constraint-region diagram, weight histograms with and without penalty, λ slider); `dropout_mask` (network with units dropped per mini-batch, inverted-dropout scaling arithmetic shown); `spatial_dropout` (whole feature-map channels dropped vs individual activations); `data_augmentation` (flip, crop, rotate, colour jitter, cutout on one credited photograph) | `dropout` |

### Section 2 — Image classification with deep networks

Package `engines/s2_classification/` — **built**. Module names as written:
`intro_classification` (2.1), `architectures` (2.2), `resnets` (2.3),
`transfer` (2.4), `visualisation` (2.5). The figure lists below are the
original plan; the engines' docstrings list what was actually built, which
departs from the plan where a computed version turned out to be possible or
where a planned figure would not have been honest (section 10).

| # | Page / source | Engines (figures) | Worked examples |
|---|---|---|---|
| 2.1 | `02_01.html` — Introduction to image classification<br>`lesson_2_1_introduction`, `lesson_2_1_script` | `classification_task` (image → label, single-label assumption); `recognition_challenges` (viewpoint, scale, illumination, deformation, occlusion, clutter, intra-class variation — one credited photo per panel, transformed synthetically); `standard_datasets` (MNIST/CIFAR/ImageNet/COCO compared on size, classes, resolution); `imagenet_lsvrc` (winning top-5 error by year with depth annotated — the classic chart, built from a small data table in the engine); `binary_confusion` (pedestrian/no-pedestrian matrix with TP/FP/FN/TN and the accuracy, precision, recall equations evaluated on it); `multiclass_confusion` (3-class cat/dog/mouse matrix, one-vs-rest TP/FP/FN/TN highlighted per class, macro vs weighted averaging); `top1_top5` (two classifiers' probability bars for one image, top-1 wrong / top-5 right); `class_imbalance_bias` (per-class accuracy against class frequency) | `con_mat` |
| 2.2 | `02_02.html` — Architectures for image classification<br>`lesson_2_2_architectures_image_Classification`, `lesson_2_2_script` | `alexnet_diagram` (layer blocks with tensor shapes); `alexnet_cost` (parameters and MACs per layer as stacked bars — shows conv dominates compute, FC dominates parameters; all numbers computed from the layer table); `vgg_blocks` (3×3 stacking, two 3×3 vs one 5×5 receptive field and parameter comparison); `network_in_network` (1×1 conv as a cross-channel MLP); `global_avg_pool` (flatten+FC vs GAP, parameter counts); `inception_naive_vs_1x1` (naive multi-branch cost vs the 1×1-reduced block, computed); `inception_architecture` (full block and the stacked network); `aux_classifiers` (where they attach, weighted loss); `imagenet_scatter` (top-1 accuracy vs parameters vs MACs for the model zoo, bubble chart from a data table) | `global_avg_pool`, `inception_block` |
| 2.3 | `02_03.html` — Residual networks<br>`lesson_2_3_Residual_Networks`, `lesson_2_3_script` | `degradation_problem` (deeper plain net doing worse — training and test curves); `residual_block` (identity shortcut, F(x)+x, basic vs bottleneck variant with parameter counts); `resnet_connectivity` (projection shortcut when shape changes, stage table for ResNet-18/34/50/101); `gradient_flow` (gradient magnitude by depth with and without shortcuts, on a synthetic chain); `batch_norm` (activation distributions before and after normalisation; the γ, β rescaling; the train-time batch statistics); `batch_norm_inference` (running mean/variance accumulation and the fold into inference); `squeeze_excite` (squeeze → FC → sigmoid → channel rescale, with channel weights drawn); `densenet` (dense connectivity vs residual, growth rate k and channel count per layer) | `bat_norm1`, `bat_norm2` |
| 2.4 | `02_04.html` — Transfer learning<br>`lesson_2_4_transfer_learning`, `lesson_2_4_script` | `transfer_pipeline` (pretrained backbone, head replaced, which layers freeze); `feature_generality` (early layers generic, late layers task-specific — schematic with a similarity-by-depth curve); `freeze_vs_finetune` (decision grid over dataset size × domain similarity, the four quadrants with recommendations); `data_efficiency` (accuracy vs training-set size, from scratch vs fine-tuned — schematic curves labelled as illustrative); `pretrained_zoo` (table/chart of common backbones: input size, parameters, top-1, typical use) | — |
| 2.5 | `02_05.html` — Feature visualisation and adversarial attacks<br>`Lesson_2_5_visualisation`, `lesson_2_5_script` | `reconstruction_from_features` (what is preserved at increasing depth — schematic ladder with progressively abstracted versions of one credited photo); `activation_maximisation` (the optimisation loop diagram; synthetic Gabor/texture targets standing in for the optimised inputs); `feature_hierarchy_gallery` (layer-by-layer preferred-pattern gallery, synthetic); `style_transfer` (content loss and Gram-matrix style loss defined and computed on two real images in NumPy — the Gram matrices are genuine even though no network runs); `style_transfer_training` (the loss-weighting trade-off, α/β slider on the schematic); `deep_dream` (feedback amplification loop diagram); `adversarial_attack` (FGSM arithmetic on a toy linear classifier: x + ε·sign(∇), decision flip shown in 2D and as an image perturbation at several ε); `adversarial_physical` (patch/sticker attack schematic); `generalisation_limits` (distribution shift: accuracy against corruption severity) | — |

### Section 3 — Object detection and segmentation

Package `engines/s3_detection/`.

| # | Page / source | Engines (figures) | Worked examples |
|---|---|---|---|
| 3.1 | `03_01.html` — Introduction to object detection<br>`lesson_3_1_introduction`, `lesson_3_1_script` | `detection_task` (classification vs localisation vs detection vs segmentation on one scene); `iou_boxes` (predicted and ground-truth boxes, intersection and union shaded, IoU recomputed as a box is dragged — Plotly); `tp_fp_fn_rules` (the confidence threshold T1 / IoU threshold T2 decision tree from slide 6, worked on a scene with several predictions); `detection_datasets` (Pascal VOC, COCO, Open Images compared); `pr_curve_threshold` (high vs low confidence threshold, the detections kept in each case and the resulting precision/recall point); `pr_curve_zigzag` (raw zig-zag PR curve built by ranking a table of predictions by confidence); `interpolated_precision` (the same curve with P_interp(r) = max precision at recall ≥ r drawn over it); `average_precision` (AP as the area under the interpolated curve, rectangles shaded; mAP over classes); `detection_errors` (localisation, similarity, background false positives; small/occluded/truncated false negatives — boxes drawn on credited photos); `coco_error_analysis` (the nested PR curves as error sources are relaxed); `detection_augmentation` (flip, scale jitter, crop with box coordinates transformed correctly) | `iou`, `objdet_pre` |
| 3.2 | `03_02.html` — Region proposals<br>`lesson_3_2_region_proposals`, `lesson_3_2_script` | `sliding_window_cost` (windows × scales × aspect ratios counted for a real image size — motivates proposals); `objectness_saliency` (multi-scale spectral saliency map computed on a credited photo); `objectness_colour_contrast` (window vs surrounding-ring colour histogram χ² distance); `objectness_edge_density` (edge density in the box border ring); `objectness_superpixel` (SLIC superpixels straddling a box boundary, the straddling penalty computed); `objectness_combined` (the cues combined into a ranked proposal set); `selective_search_merge` (Felzenszwalb initial segmentation, then the greedy similarity merge stepped through as a slider over iterations); `region_similarity` (colour, texture, size and fill similarity terms computed for two neighbouring regions); `proposal_recall` (recall vs number of proposals at a fixed IoU) | `iou` |
| 3.3 | `03_03.html` — Two-stage object detection<br>`lesson_3_3_obj_det_partA_w_reg_Prop`, `lesson_3_3_script` | `rcnn_pipeline` (proposals → warp → CNN per region → SVM + box regression; the per-image CNN-forward count computed); `conv_feature_sharing` (2000 forwards vs one shared feature map); `spp_layer` (fixed-length output from any input size: the 4×4/2×2/1×1 bin grid and the resulting vector length); `fast_rcnn` (shared conv, RoI pooling, joint classification + regression heads, multi-task loss); `class_activation_map` (CAM computed properly in NumPy from a synthetic GAP layer: weighted sum of feature maps → heat map); `roi_projection` (image coordinates → feature-map coordinates via the cumulative stride, with the quantisation error shown); `region_proposal_network` (anchors at one feature-map cell, objectness + box-delta outputs, the anchor grid over a photo); `two_vs_single_stage` (accuracy/speed trade-off chart for the detector families) | `rfield`, `objdet_pre` |
| 3.4 | `03_04.html` — Single-stage object detection<br>`lesson_3_4_obj_det_partB_single_stage`, `lesson_3_4_script` | `single_stage_principle` (dense prediction grid, one forward pass); `detection_output_tensor` (S×S×(B·5+C) output tensor unpacked, cell responsibility shown); `anchor_boxes` (anchor scales and aspect ratios drawn over a grid cell; matching rule by IoU; positive/negative/ignore assignment); `box_parameterisation` (t_x,t_y,t_w,t_h → box, the sigmoid/exp decoding computed for a worked cell); `detection_loss` (localisation + objectness + classification terms with their weights); `focal_loss` (cross-entropy vs focal loss against p_t for several γ, and the effective gradient contribution of easy vs hard examples); `class_imbalance` (anchors per image, positives vs negatives counted); `ssd_multiscale` (predictions from several feature-map resolutions, default boxes per level, total box count computed); `feature_pyramid` (bottom-up, top-down and lateral connections, resolution and channel count per level); `depthwise_separable` (standard vs depthwise+pointwise, MAC and parameter reduction computed); `nms_steps` (non-maximum suppression stepped through a set of overlapping boxes) | `depthwise_sep_conv`, `iou` |
| 3.5 | `03_05.html` — Semantic segmentation (part A)<br>`lesson_3_5_sem_seg_partA`, `lesson_3_5_script` | `segmentation_types` (semantic vs instance vs panoptic on one scene); `segmentation_datasets` (Pascal VOC, Cityscapes, ADE20K: classes, images, annotation type); `segmentation_output` (H×W×C class-score tensor → argmax label map → colour overlay); `segmentation_metrics` (pixel accuracy, mean accuracy, mean IoU and frequency-weighted IoU all computed on one small prediction/ground-truth pair, with the per-class IoU table); `fully_convolutional` (dense layer rewritten as a convolution — the same weights, the heat-map output for a larger input); `conv_as_matrix` (convolution as a sparse Toeplitz matrix multiply, and the transpose that gives transpose convolution); `transpose_conv_stride1` (input scattered and summed, output size formula); `transpose_conv_stride2` (zero insertion between inputs, checkerboard artefact shown and explained); `encoder_decoder` (downsample/upsample path with resolution at each stage); `skip_connections` (FCN-32s/16s/8s fusion, the detail recovered at each); `unet` (the full U with channel counts and the crop/concatenate step) | `mat_conv`, `transpose_conv1`, `transpose_conv2` |
| 3.6 | `03_06.html` — Context and instance segmentation<br>`lesson_3_6_sem_seg_partB`, `lesson_3_6_script` | `context_knowledge` (an ambiguous patch resolved by its surroundings; receptive field vs object size); `dilated_conv` (dilation rate slider: kernel support, receptive field growth without resolution loss, parameter count unchanged); `dilated_stack` (exponentially growing receptive field over stacked rates, and the gridding artefact); `pyramid_scene_parsing` (PSPNet pooling at 1/2/3/6, concatenation, output); `conditional_random_field` (unary and pairwise potentials on a small label grid; mean-field iterations sharpening a boundary — computed, not schematic); `deeplab` (atrous spatial pyramid pooling, the full pipeline); `instance_segmentation` (semantic vs instance output on overlapping objects); `mask_rcnn` (Faster R-CNN + mask head; RoIAlign's bilinear sampling vs RoIPool's quantisation, worked on a small grid); `mask_rcnn_output` (per-instance box, class, mask composited into a final result) | `dilated_conv` |

### Site-level pages

| HTML | Content |
|---|---|
| `index.html` | Course map: three sections, 17 lessons, links to the labs repo and examples repo |
| `about_examples.html` | How to open the examples repo in Codespaces and run `streamlit run examples/app.py` (still unwritten, as in CE6021) |

Practice pages stay out of scope: students get the practice PDFs through
Brightspace.

**Engine count: 96 figure engines across 17 lessons.** Section 1 and lesson 3.1
are the priority — they carry most of the exam question types.

---

## 5. Why no PyTorch

Worth recording, because it is the one place this course plan gives something
up. CE5021 is a deep-learning module, and a reader may reasonably expect
figures showing real feature maps from a real trained network.

What the NumPy-only decision costs, concretely:

- 2.5 runs every visualisation algorithm for real, but on a *fixed* Gabor-filter
  network (`_fixednet.py`) rather than a trained CNN, so it shows how each
  technique works and how depth changes what survives — not what a trained
  network's deep layers know. Full style transfer is shown as a diagram plus
  genuine Gram-matrix texture synthesis, because with one fixed layer of
  features the full transfer was not convincing. The adversarial and
  generalisation figures attack a genuinely trained classifier on the digits.
- 3.3–3.6 show detector and segmentation *mechanisms* (RoI projection, anchor
  matching, NMS, transpose convolution, RoIAlign, CRF mean-field) computed
  exactly, but not inference output from a pretrained detector on a photograph.

What it buys: the devcontainer stays small, `build_site.py` stays a
seconds-not-minutes operation with no weight downloads, the build is
deterministic and offline-capable, and every number printed in a figure is
computed from a specification the student can check by hand — which is what the
exam asks them to do.

If real network outputs are wanted later, the clean seam is a single new engine
package that requires `torch`/`torchvision`, with those lessons' figures
falling back to the schematic versions when the import fails.

## 6. Build order

1. **Retire CE6021 content.** Delete `content/*.md`, `content/_selfcheck/*` and
   `engines/s1_images … s6_bof`; write the new `content/_course.yaml`. Keep
   `_common.py`, `_vendor/`, `build/`, `publish/`, `data/`, `examples/app.py`.
   (`_course.yaml` is already rewritten — see the file.)
2. **Prove the round trip** on lesson 1.1 end to end: engine → PNG + Plotly div
   → page → Streamlit demo. 1.1 is the right probe because it exercises
   figures, worked examples and self-check together.
3. **Section 1** (6 lessons, ~38 engines). Most of the reusable primitives get
   written here: the tensor-shape block diagram, the parameter/MAC cost table,
   the small-matrix convolution stepper.
4. **Section 2** (5 lessons) — done, with the architecture-spec module
   `_arch.py` from which every parameter and FLOP figure is computed.
5. **Section 3** (6 lessons). `_arch.py` already has MobileNetV2 (for 3.4's
   depthwise separable convolution) and should be extended for SSD, FPN and
   U-Net rather than hard-coding their numbers.
7. **Self-checks** for all 17 lessons, written against the finished prose.
8. **Publish**: update `publish/released.txt`, `publish_site.sh` repo name to
   `CE5021_*`, then `--dry-run` and `--create`.

## 7. Publishing the student site

Unchanged from CE6021 and already working. The student site is **generated
output only**: lesson markdown, figure engines, build system and repository
history are never copied into the published repo, so the course cannot be
rebuilt or re-themed from what is published.

| Step | Command |
|---|---|
| Preview locally, push nothing | `lessons/publish/publish_site.sh --dry-run` |
| First publication | `lessons/publish/publish_site.sh --create` |
| Release more lessons later | edit `released.txt`, then `--update` |
| Remove the remote repo | `lessons/publish/publish_site.sh --delete` |

`released.txt` lists the lesson ids to publish. Unreleased lessons get no page
and no figures in the output, are skipped by previous/next navigation, and
appear greyed out as "coming soon" on the index.

**The published site is always publicly readable.** A private Pages site, where
viewers must authenticate, requires GitHub Enterprise Cloud and an
organization-owned repository. The repository is created private by default,
which only affects github.com: no fork button, not clonable, not in GitHub
search; that needs GitHub Pro, Team or Enterprise, and the script checks the
account plan and stops with an explanation rather than half-failing. The
published site also carries a reserved-rights `LICENSE`, a `robots.txt` that
excludes it from search engines, and a `build-info.txt` provenance stamp.

None of this prevents a third party from saving the rendered pages, which is
not technically possible for any public site. The substantive protections are
that only rendered output is published and that the licence reserves all
rights. If genuine access control is needed, the site is a self-contained
folder of relative-path HTML and can be hosted inside Brightspace instead,
where students already authenticate. KaTeX and Plotly are loaded from CDNs and
would need vendoring locally first if the LMS blocks external scripts.

## 8. Open points

1. **Section 3's self-checks** need authoring or topping up (see section 2).
2. **`autoquiz/` is a temporary clone.** Now that the 22 generators are vendored
   and verified it can be deleted; `engines/_vendor/` is self-contained and
   nothing under `lessons/` imports from outside it.
3. **Two new images** are wanted in `data/sources.yaml` (a multi-object street
   scene for detection, a stuff/thing scene for segmentation). Lessons 1.1–1.3
   need none — every figure in them is synthetic or computed, so `data/` is
   currently unused.
4. **Two things in the slides worth checking.** The lesson 1.3 bottleneck
   example gives 2,654,308 multiplications where 12·3·3·24·32·32 =
   **2,654,208**; the figure computes it at build time and prints the right
   number. And the lesson 1.4 receptive-field slide quotes **7** for neuron C
   with conv(3,1) → pool(2,2) → conv(3,1); the recurrence gives **8**, and 7 is
   what you count if the drawing clips the padded position at the image
   boundary. The lesson uses valid padding so the picture and the formula agree
   exactly; the deck's figure could not be checked because the converted
   markdown holds no numbers for it.
5. **Section 2 points against the sources**, each stated on the page:
   - AlexNet's 55 × 55 conv1 output needs a **227** input; the paper and slide
     say 224. The lesson uses 227 and says why — it is a good exercise in
     lesson 1.1's formula.
   - The lesson 2.2 script gives AlexNet "600 million" operations. Computed:
     **0.72 G** multiply-accumulates for the original two-GPU network and
     **1.14 G** for the single-GPU form the slide draws. Both are shown.
   - GoogLeNet's paper (and the script) says 12× fewer parameters than AlexNet;
     computing from its own layer table gives **~7.0 M vs 61 M, about 9×**.
   - The 2.3 script says squeeze-and-excite gives "a 10% improvement" for
     ResNet-50; that is a roughly 10% *relative* reduction in error, not ten
     percentage points, and the lesson words it that way.
6. **Build time.** Lesson 2.5 takes ~95 s of the ~105 s full build (image
   optimisation by L-BFGS). Acceptable for now; if Section 3 adds similar
   work, a small on-disk cache of figure results keyed by engine source hash
   would be the fix.

## 9. Notes from building 1.1–1.3

- **Volume orientation.** `engines/s1_convnets/_volume()` draws volumes the way
  the CNN literature and the course videos do: the **horizontal x axis is the
  channel axis**, and y/z are the spatial dimensions, so a volume with many
  channels is a *long box* whose square *end cap* is its spatial extent. Deeper
  into a network the boxes get longer and the end caps smaller. A convolution is
  marked by a dashed cube inside the input (`_kernel_cube()`) that runs the
  **full length** of the volume — because a kernel spans every input channel —
  while covering only a k × k patch spatially. Any new volume diagram must
  follow this; the first version of these figures had spatial on the visible
  face with channels receding diagonally, which is the wrong way round.
- Channel extent is **log-compressed** (`_chan_len`) because a 512-channel
  volume drawn to scale beside a 3-channel one runs off the canvas. Spatial
  extent is **sqrt-compressed** (`_spatial_side`), not log: the lessons point
  at spatial halving repeatedly and under log compression a 32 → 16 halving is
  barely visible. `_volume()` returns its right-hand extent so callers lay
  volumes out left to right and set `xlim` from the result rather than guessing.
- When stacking volumes vertically, the pitch must clear the end-cap side **plus
  its oblique rise** (`side * (1 + OBLIQUE[1])`), or they overlap.
- Draw network nodes with `ax.scatter`, not `plt.Circle`: patch radii are in
  data units, so circles come out as ellipses unless the axes aspect is equal.
- Where two curves coincide over part of their range (ReLU and leaky ReLU for
  z > 0), dash one of them or the lower one is invisible.
- Marking *every* kernel position on a grid is unreadable at stride 1. Mark each
  position's anchor cell with a small ring and draw one full kernel outline.
- The worked examples validate against the Grasple solutions where the two
  overlap: the 1.3 multi-channel example reproduces Grasple exercise 107366's
  answer, and 1.1's pooling figure reproduces exercise 107288's.

## 10. Notes from building Section 2

**Shared infrastructure, all under `engines/`:**

- `_arch.py` — layer-spec cost calculator: shapes, parameters and
  multiply-accumulates, using exactly the formulas the lessons teach.
  `verify()` runs on every build of 2.2–2.4 and asserts against published
  counts: VGG-16, ResNet-18/34/50/101/152, DenseNet-121 and MobileNetV2 match
  torchvision **exactly**; the original two-GPU AlexNet gives exactly
  60,965,224; SE-ResNet-50 exactly the paper's 28.09 M. A wrong formula fails
  the build instead of printing a wrong number.
- `_tinynet.py` — a small MLP with hand-written backprop, gradient-checked to
  1e-7, for training experiments on the sklearn digits (2.3 degradation,
  2.4 transfer). Each runs in seconds.
- `_fixednet.py` — Gabor filter bank → ReLU → pooling, with an exact input
  gradient (checked to 1e-9) and the four 2.5 techniques built on it.
  Convolutions are one broadcast FFT call (matches direct correlation to
  1e-15). Note SciPy's `fftconvolve(mode="same")` crops *every* axis to the
  first argument's shape, including un-convolved ones — broadcast first.
- `_volumes.py` — the volume-drawing helpers, moved out of 1.3's engine.
- `build/selfcheck_tools.py` — writes and validates the self-check YAML.
  Use it for every new question: `yaml.dump()` on a scalar appends the
  document-end marker "...", which corrupted eight files once.

**What the experiments showed, and how the pages report it:**

- 2.3: a 40-layer plain network stays at the chance loss (ln 10) while the
  40-layer residual one beats a 3-layer network. Veit's path argument,
  measured in that trained network: 99.98% of the gradient travels along paths
  through ≤ 10 of its 39 blocks.
- 2.4: transfer to a *related* task (digit identity → parity) wins clearly with
  few labels and a frozen extractor matches fine-tuning; transfer to *new
  classes* (0–4 → 5–9) gives no advantage at any size. The first attempt
  sought a small-data win on new classes and could not get one honestly — raw
  digits are nearly linearly separable, so a from-scratch model is not short
  of features — so the pages report the nuance, which is what the slides'
  He et al. and Raghu et al. material argues anyway.
- 2.4's convergence figure uses a shared learning rate of 0.005, stated in the
  caption: at 0.03 both methods reach 96% by epoch 3 and the effect vanishes.

**Lessons for Section 3:**

- Compute caption claims from the same numbers the figure draws, or check
  them after the build. Four captions in Section 2 asserted things the figure
  did not show (a "confident" misclassification at 46%; "not tidy digits" when
  some were; a convergence gap the computed title contradicted; 99.98% printed
  as "100.0%"). Each was caught only by reading the rendered figure.
- When an experiment is meant to demonstrate a claim, put an `assert` on the
  claim (as `momentum_path` and `translation_invariance` do) so that a change
  in parameters cannot silently turn the figure into a counter-example.
- Prefer genuine data (sklearn digits, credited photos) over invented numbers;
  where a number is illustrative (2.1's top-k probabilities, 2.4's detection
  feature map), say so in the caption.
