---
id: "02_02"
title: The Canny algorithm and the Hough transform
section: 2
engines: [s2_edges.canny_hough]
practice: [canny_nms, hys_thresh, hough]
summary: A multi-stage edge detector that returns thin, reliable edges, and a voting method that fits parameterised lines to the edge points it produces.
---

## Learning outcomes

After this lesson you should be able to:

- state the goals of a multi-stage edge detector and list the **stages of the Canny algorithm**;
- apply **non-maximum suppression** and **hysteresis thresholding** to a small grid by hand;
- explain the **Hough transform** in both slope–intercept and **polar** parameterisations;
- explain why voting in parameter space is robust to gaps and noise in an edge map.

## Canny edge detection

The simple kernels of the previous lesson are sensitive to intensity changes, so they find many edges, including edges due to noise, return peaks of different strength depending on the gradient, and give edges several pixels wide. A **multi-stage edge detector** aims for three things:

- **localisation**: the detected points lie close to the true edge;
- **robustness to noise**;
- a **single response**: one-pixel-wide lines at one intensity for each edge.

The Canny algorithm is the standard example. Compared with a Sobel magnitude image it removes the minor edges and returns narrow edges all at the same level.

{{fig sobel_vs_canny}}

### The algorithm

1. **Smooth** the image with a Gaussian to remove noise that would create false edges.
2. Apply **Sobel** filtering and compute the gradient **magnitude and direction**.
3. Apply **non-maximum suppression** to thin the edges.
4. Apply **hysteresis thresholding** to keep only the important edges.

{{fig canny_stages}}

### Non-maximum suppression

Non-maximum suppression keeps only pixels that are a **local maximum in the direction of the gradient**, using both the magnitude and the direction obtained in step 2. Each pixel is compared with its two neighbours along its gradient direction (quantised to 0°, 45°, 90° or 135°); if its magnitude is at least as large as both it is retained, otherwise it is set to zero. A ridge several pixels wide is reduced to a line one pixel thick.

{{fig nms_patch}}

{{example canny_nms seed=7}}

### Hysteresis thresholding

Sobel filtering detects many edges, some of them false or insignificant. Hysteresis thresholding uses **two** levels, $T_{max}$ and $T_{min}$:

- a pixel above $T_{max}$ is a strong edge and is **kept**;
- a pixel below $T_{min}$ is **removed**;
- a pixel between the two is kept **only if it is connected** to a pixel above $T_{max}$.

So a weak pixel that continues a strong edge survives, while an isolated weak pixel is discarded.

{{fig hysteresis}}

{{example hys_thresh seed=2}}

## Lines from edges

Even with a good detector the edges often do not describe an object boundary completely: edges are broken, and clutter produces spurious points. A higher-level description fits **lines described by an equation** to the edge points, after which missing pixels in an edge no longer matter. Doing this directly is computationally hard; the **Hough transform** groups edge points into lines by **voting** over parameterised lines.

### The linear Hough transform

Each point $(x_i, y_i)$ in image space could lie on very many candidate lines $y = mx + b$, each described by two parameters, the slope $m$ and the intercept $b$. The key property: the set of all lines through one image point forms a **line in parameter space**, $b = y_i - m x_i$. For two points on the same edge, their two lines in parameter space intersect at the $(m, b)$ of the true line through both.

This gives the basic algorithm: for every edge point, trace its candidate lines in parameter space; then find the points of parameter space with the **most votes**, which are the intersections. Those cells give the equations of the lines that best fit the edges.

{{fig hough_parameter}}

### The polar representation

The $(m, b)$ parameterisation has a problem: the slope of a vertical line is infinite. The practical version describes a line by its **angle** θ and its **distance** ρ from the origin,

$$\rho = x \cos\theta + y \sin\theta, \tag{1}$$

both of which are bounded. The principle is unchanged, but the candidate lines through an image point now map to a **sinusoid** in $(\rho, \theta)$ space, and intersections of sinusoids are sought instead. The Hough transform is a brute-force method: θ is quantised to limit the number of candidate lines, and the **probabilistic** Hough transform reduces the run time further by voting with only a random subset of the edge points.

{{fig hough_accumulator}}

{{fig hough_photo}}

{{example hough seed=4}}

{{selfcheck 02_02}}
