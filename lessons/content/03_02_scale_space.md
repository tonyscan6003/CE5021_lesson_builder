---
id: "03_02"
title: Scale spaces
section: 3
engines: [s3_features.scale_space]
summary: "The Gaussian scale space as a general framework: characteristic scale, the normalised Laplacian, how scale invariance arises, the 26-neighbour search, the Harris-Laplacian detector and the difference-of-Gaussians approximation."
---

## Learning outcomes

After this lesson you should be able to:

- define a **Gaussian scale space** and explain why the Gaussian is used to build it;
- state Lindeberg's principle and define the **characteristic scale** of a feature;
- explain how the **normalised Laplacian** gives a scale-invariant detector;
- describe the **26-neighbour** search for maxima in scale and space;
- explain the **Harris-Laplacian** detector and the **difference of Gaussians** approximation.

## Scale space

Images contain structures at many scales, and we cannot know in advance which scales are present. The Gaussian is the canonical function for building a **scale space**: a family of images

$$L(\mathbf{x}; \sigma) = G(\mathbf{x}; \sigma) * I(\mathbf{x}), \tag{1}$$

obtained by convolving the image with Gaussians of increasing σ. The Gaussian is chosen because it introduces no artefacts. Blurring gradually removes information: at each level only structure coarser than about σ remains, so petals and small flowers disappear first and large shapes survive. Viewing detectors through scale space is valuable because many apparently different scale-invariant detectors turn out to be different combinations of **Gaussian derivatives**.

{{fig gaussian_scale_space}}

## Characteristic scale

The scale-invariant detectors in this lesson rest on a principle stated by Lindeberg in 1998: *in the absence of other evidence, assume that a scale level at which some combination of normalised derivatives assumes a local maximum over scales reflects a characteristic length of the corresponding structure in the data.* In other words, if we compute a suitable normalised derivative function at every level of the scale space, at any given image position it has a single maximum over scale, and that scale is the **characteristic scale** of the structure there.

The **normalised Laplacian** is such a function:

$$\nabla^2_{norm} L(\mathbf{x}; \sigma) = \sigma^2 \left( \frac{\partial^2 L}{\partial x^2} + \frac{\partial^2 L}{\partial y^2} \right) = \sigma^2\, (\nabla^2 G) * I. \tag{2}$$

The factor σ² compensates for the loss of edge amplitude caused by Gaussian smoothing and is essential for the correct maximum to be found. The second form shows that the Laplacian scale space is the image convolved with the Laplacian of Gaussian kernel: the LoG blob detector of the previous lesson, now derived from the principle of a single maximum over scale. The scale at which a feature reaches its maximum is related to its size in pixels.

{{fig characteristic_scale}}

{{fig laplacian_kernels}}

## Scale invariance

Scale invariance arises naturally in this framework. Take an image and a zoomed copy of it and compute the squared normalised Laplacian of both. A feature reaches its maximum at a smaller σ in the original than in the zoomed image, exactly as expected, because the feature is larger in the zoomed image. Each feature has one characteristic scale that depends only on its size in the image, so the same keypoint is found in both images.

{{fig scale_invariance}}

## Local maxima detection

The maxima over scale also locate features in space. With the right function, such as the Laplacian, only one maximum in scale occurs at each spatial position, and a nearest-neighbour search finds them: a sample is a maximum if it exceeds its **26 neighbours**, the 8 at the same scale and the 9 at each adjacent scale. This simple search is very effective.

{{fig maxima_26}}

## The Harris-Laplacian detector

The Harris-Laplacian detector combines the Harris corner response with the Laplacian. Recall that Harris uses the corner response $R$ obtained from the second-moment matrix $M$. Here $M$ is written as a function of the image position, a **differentiation scale** γ at which the Gaussian derivatives of $L$ replace the Sobel approximation, and an **integration scale** σ that sets the size of the Gaussian window instead of a fixed window; the two are related by a fixed factor of about 0.7.

The algorithm: at each scale, detect corners with the appropriately sized window and keep the local **spatial** maxima of $R$ (8 neighbours, no normalisation needed); then use the normalised **Laplacian** to check whether each candidate is also a maximum in the **scale** direction (2 neighbours).

Why not use the Laplacian alone? The Laplacian responds to **edges** at a similar scale as well as to blobs, so the outlines of objects light up along with their spots. The Harris response identifies those outlines as edges and rejects them. The Harris-Laplacian therefore finds very robust keypoints, but far fewer of them, because it demands simultaneous maxima of two different functions.

{{fig harris_vs_laplacian}}

## The difference of Gaussians

A third important function is the **difference of Gaussians** (DoG), used by the SIFT detector of Lesson 3.4. The Laplacian scale space is expensive to build, because it needs ever larger Gaussian-derivative kernels convolved with the original image, whereas a Gaussian scale space can be formed by successive convolutions with small kernels (see the supplementary Lesson 3.5). Once a Gaussian scale space exists, the Laplacian can be approximated by subtracting neighbouring levels:

$$\frac{G(\mathbf{x}; \rho\sigma) - G(\mathbf{x}; \sigma)}{\rho - 1} \approx \sigma \frac{\partial G}{\partial \sigma} = \sigma\, \nabla^2 G, \tag{3}$$

where ρ is a factor just above 1. The left-hand side has the form of a finite-difference derivative with respect to σ, and the derivative of a Gaussian with respect to σ can be shown to equal the normalised Laplacian of Gaussian. The approximation improves as ρ approaches 1.

{{fig dog_vs_log}}

## Summary

Given two images of the same scene at very different scales, the goal is to find the same keypoints independently in each. We do this by searching for maxima of a suitable function in scale and space; the Laplacian, the Harris-Laplacian and the difference of Gaussians are all suitable functions.

{{selfcheck 03_02}}
