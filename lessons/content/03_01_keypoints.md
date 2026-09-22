---
id: "03_01"
title: Keypoints and feature detection
section: 3
engines: [s3_features.keypoints_log]
summary: "What keypoints are and what they are used for, the properties of a good feature detector, why the Harris detector falls short on scale, and the Laplacian-of-Gaussian blob detector that fixes it."
---

## Learning outcomes

After this lesson you should be able to:

- define a **keypoint** and give applications of keypoint detection;
- list the properties of a good **feature detector**, in particular **scale and rotation invariance**;
- explain why the Harris corner detector is **not scale invariant**;
- describe the **Laplacian of Gaussian** kernel and how a family of LoG kernels detects blobs at their own scale.

## Image keypoints

A **keypoint**, or interest point, is a point in an image that stands out and can be detected reliably. Corners are one kind of image feature that makes a good keypoint; circular blobs or regions are another. A **feature detector** is any algorithm, such as the Harris algorithm, that finds such features. Any feature that a detector finds robustly and repeatably can serve as a keypoint.

### Uses of keypoints

Keypoints underpin image matching, object detection, image alignment and robot navigation. If the same set of keypoints can be detected in two views of an object, the views can be matched; if corresponding keypoints are found in overlapping photographs of a scene, the photographs can be aligned into a panorama. How keypoints are *described* so that they can be matched is the subject of Lesson 3.3.

{{fig keypoint_uses}}

## Properties of a good feature detector

- **Local**: it finds features in small regions of pixels, so many keypoints survive even when the object is partly hidden or surrounded by clutter.
- **Invariant to scale and rotation**: the same features are found when the object appears at a different size or orientation.
- **Robust** to noise and blur, unlike simple edge detectors.
- **Numerous**: many features are found, even on small objects.
- **Accurate**: keypoint locations are precise, which matters for panoramas and navigation.

{{fig detector_criteria}}

### Invariance and the Harris detector

Edges and corners are features we have already met. The Harris detector is rotation invariant, but it is **not scale invariant**: it inspects a small region through a fixed-size window and is therefore sensitive to corners at one scale only. On a photograph of a balloon it finds small corners in the lettering while the large curved outline of the balloon registers as an edge. Trying to match two views of an object at different scales with Harris keypoints therefore fails, because different corners are found in each.

{{fig harris_scale}}

## The Laplacian-of-Gaussian blob detector

We want a detector that is both scale and rotation invariant while keeping the other properties above. The **Laplacian of Gaussian** (LoG) blob detector is one. A LoG kernel is formed by applying the Laplacian operator to a Gaussian:

$$\nabla^2 G(x, y; \sigma) = \frac{x^2 + y^2 - 2\sigma^2}{\sigma^4}\; \frac{1}{2\pi\sigma^2}\, e^{-\frac{x^2 + y^2}{2\sigma^2}}. \tag{1}$$

Larger σ gives a wider kernel. Each kernel responds most strongly to circular regions, or **blobs**, of a size matching the kernel, so convolving an image with a series of LoG kernels of different σ detects blobs at different scales. The origin of this kernel in scale-space theory is explained in the next lesson.

{{fig log_kernels}}

### Response to one kernel

Convolving an image of dots with a single LoG kernel produces white regions where the response is strongest. Squaring and normalising the response, and displaying it as a heat map, shows that the kernel responds most strongly to dots of a particular size, with radius approximately equal to √2 σ. Since a symmetric kernel gives the same result for convolution and cross-correlation, the response is strongest where the image looks like the kernel: a bright disc surrounded by a dark ring.

{{fig log_blob_response}}

### The scale-invariant blob detection algorithm

1. Generate LoG kernels at a series of scales σ.
2. Convolve the greyscale image with each kernel in turn.
3. For each result compute the **normalised square response**.
4. Find the **maxima** of the response; these give the keypoint coordinates and scale. A threshold, or the sign of the response, selects dark blobs, bright blobs or both.
5. Mark the keypoints, drawing each with a circle whose radius corresponds to its σ, and optionally apply non-maximum suppression to remove overlapping detections.

{{fig log_multi_scale}}

{{selfcheck 03_01}}
