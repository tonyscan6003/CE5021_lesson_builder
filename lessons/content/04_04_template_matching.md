---
id: "04_04"
title: Template matching
section: 4
engines: [s4_detection.template_matching]
practice: [hausdorff]
summary: "Detecting a specific object by exhaustive comparison with a template: Euclidean and correlation-based similarity, normalised cross-correlation, and shape-based matching with the distance transform, chamfer distance and Hausdorff distance."
---

## Learning outcomes

After this lesson you should be able to:

- describe **template matching** and its similarity measures: Euclidean distance, cross-correlation and **normalised cross-correlation**;
- explain why **shape-based** matching is more robust than pixel correlation;
- use the **distance transform** to compute distances between shapes efficiently;
- describe **chamfer matching** and compute the directed and full **Hausdorff distance**.

## Basic object detection

Basic object detection looks for a specific object, or type of object, in an image. Keypoint matching does this robustly against orientation, scale and illumination changes. An alternative is to use the object, or part of it, as a **template** and search the image exhaustively for the best match. Template matching is a pure image-processing approach without machine learning, which makes it less robust than the Viola-Jones and HOG detectors to changes in orientation, scale and illumination.

## Template matching

The template is moved systematically over every position of the image, like a kernel in convolution or a structuring element in a non-linear filter, and a **similarity metric** is evaluated at each position. A stride of 1 gives the best chance of an exact match; a larger stride reduces computation. The simplest metric is the **Euclidean distance** between corresponding pixels of the template and the image region,

$$d(m, n) = \sqrt{\sum_{i, j} \big( T[i, j] - I[i + m, j + n] \big)^2 }, \tag{1}$$

with the template shifted by $(m, n)$; the position with the **lowest** score is the best match.

{{fig template_search}}

### Correlation-based similarity

**Cross-correlation** between template and image is another similarity measure. Its equation resembles convolution but the template is not reversed, and the **highest** value marks the best match. Plain correlation has a weakness: the score also grows wherever the image is bright, so bright regions produce false local maxima.

### Normalised cross-correlation

**Normalised cross-correlation** keeps the element-wise multiply-and-sum but subtracts the mean and divides by the standard deviation of both template and image region. The template's statistics are constant, but the image mean and standard deviation must be recomputed at every position, so NCC is more expensive than plain correlation. In return the false maxima in bright regions disappear.

{{fig correlation_vs_ncc}}

## Shape-based matching

Correlation detects regions of similar pixels with a sharp peak when the template fits, but the peak collapses with even small changes in orientation or appearance, and its sharpness gives little indication of how close a near-match is. **Shape-based** matching instead extracts the object outline by edge detection and matches it against the edge image of the scene, using a **distance between binary shapes** as the score; this tolerates minor changes of orientation and appearance. As before the template is moved over the image and the distance evaluated at each position.

{{fig shape_matching}}

### Shortest distance

The first candidate metric is the shortest distance between shapes $A$ and $B$: for every point $a \in A$ find the smallest distance to any point $b \in B$, and keep the smallest of these,

$$d_{min}(A, B) = \min_{a \in A} \min_{b \in B} \| a - b \|. \tag{2}$$

The same shortest distance results for shapes in very different relative positions, so it is not an effective measure of overlap. It is, however, cheap to compute with the **distance transform**: the distance transform of the binary image of $B$'s edges gives, at every pixel, the shortest distance to $B$, so sampling it at the pixels of $A$ gives all the required distances at once.

{{fig shortest_distance}}

### Chamfer matching

Chamfer matching is a better alternative. Edges are detected in the scene and the edge image is inverted, so that edges are black; the distance transform of this image gives every pixel its distance to the nearest edge. The result is called the **chamfer image**, because the edges become gradual slopes like a chamfered edge in woodwork. The binary template is applied at every position and the distance values under its pixels are **summed**,

$$D_{chamfer}(m, n) = \frac{1}{|T|} \sum_{(i, j) \in T} DT[i + m, j + n], \tag{3}$$

so that every template pixel, not only the nearest, contributes; the position with the minimum sum is the match. The sum can be computed as a correlation of the binary template with the chamfer image, and the global minimum found with a small further algorithm of erosion, comparison and thresholding.

{{fig chamfer_matching}}

### Hausdorff distance

The **directed Hausdorff distance** from $A$ to $B$ takes, for every $a \in A$, the smallest distance to any point of $B$, and keeps the **largest** of these:

$$h(A, B) = \max_{a \in A} \min_{b \in B} \| a - b \|, \tag{4}$$

which reduces to a maximum over the distance transform of $B$ sampled at $A$. It finds the point of $A$ furthest from $B$ and is far better than the shortest distance at telling overlapping from non-overlapping shapes. The two directed distances are **not symmetric**, $h(A, B) \neq h(B, A)$ in general, and the reverse direction needs the distance transform of $A$. The overall Hausdorff distance is the larger of the two,

$$H(A, B) = \max\big( h(A, B),\, h(B, A) \big). \tag{5}$$

Because it focuses on outlier points, where the shapes diverge most, a percentile of the distances (for example the 75th) is often used instead of the maximum. The Hausdorff distance can match templates despite differences in orientation.

{{fig hausdorff}}

{{example hausdorff seed=3}}

{{selfcheck 04_04}}
