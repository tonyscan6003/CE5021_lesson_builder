---
id: "02_01"
title: Edge detection
section: 2
engines: [s2_edges.edge_detection]
practice: [grad_dir]
summary: Edges as rapid intensity changes, derivative-based detectors from Prewitt and Sobel to Gaussian derivatives and the Laplacian, and how to display and combine gradient images.
---

## Learning outcomes

After this lesson you should be able to:

- define an **edge** in terms of pixel intensity and explain why derivative kernels detect them;
- apply the **Prewitt** and **Sobel** kernels and combine their outputs into **magnitude and direction**;
- explain the effect of **noise** on edge detection and the role of smoothing;
- describe **Gaussian derivative** kernels and how σ selects the scale of the edges found;
- explain the **Laplacian** as a second-order detector and its double response.

## Edges in images

Edges occur at boundaries between regions of different colour, intensity or texture, and they encode both semantic and shape information about the objects in an image. A human drawing the outline of a scene uses object-level knowledge to complete contours; finding such boundaries with classical machine vision is a hard problem. This lesson covers the simplest tool: derivative-based edge detectors.

## Principle of edge detection

An edge is a location where pixel intensity changes rapidly over a small region. At the left side of a dark object the pixel values fall from bright to dark over a few pixels. Taking the profile of intensity across a vertical band gives a step down followed by a step up. Derivative filters respond to a high rate of change, so the falling edge appears as a **negative** peak and the rising edge as a **positive** peak of the first derivative.

{{fig edge_profile}}

Note that convolution with an edge detector produces negative values, so to display the result as an image the offset must be removed and the range scaled to 0–255. Most plotting tools do this automatically.

## Noise and edge detectors

Because they respond to any rapid change, edge detectors are sensitive to noise, which produces a carpet of false edges. A smoothing filter such as a Gaussian is therefore applied before edge detection. Too much smoothing, however, blurs the real edges and makes them harder to find.

{{fig noise_edges}}

## The Prewitt kernels

The Prewitt kernels are based on the 1D **central difference** approximation to the derivative,

$$f'[n] \approx f[n+1] - f[n-1],$$

applied in the horizontal X or vertical Y direction. Each kernel repeats the central difference three times to make a 3 × 3 array suitable for image processing. Both respond to intensity changes across pixels, but note the orientation: the **X-direction kernel detects vertical edges** and the **Y-direction kernel detects horizontal edges**.

{{fig prewitt_kernels}}

### Magnitude and direction

Convolving a greyscale image with the two kernels gives two images, $G_x$ and $G_y$. They combine into an overall edge magnitude and a direction at every pixel:

$$|G| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \tan^{-1}\!\left(\frac{G_y}{G_x}\right). \tag{1}$$

{{fig gradient_photo}}

### Displaying edges

There are two ways to display an image containing negative values. Removing the offset and scaling to 8 bits gives a mostly grey image in which the **direction** of the intensity change survives: falling edges appear dark and rising edges bright. Taking the **magnitude** (squaring and square-rooting every pixel) makes all edges bright against the background but loses the distinction between rising and falling edges.

{{example grad_dir seed=5}}

## The Sobel kernels

The Sobel kernels are another pair of finite-difference kernels for vertical and horizontal edges. The difference from Prewitt is that they also **average** along the edge direction: the X kernel decomposes into a gradient part $[1, 0, -1]$ and an averaging part $[1, 2, 1]^T$. The extra averaging reduces susceptibility to noise. Magnitude and direction are computed exactly as for Prewitt, and the result looks similar with slightly better noise suppression.

{{fig sobel_decomposition}}

## Gaussian derivative kernels

With simple difference kernels we first smooth with a Gaussian and then differentiate: two sequential convolutions. Instead, we can differentiate the Gaussian itself and convolve once with a **Gaussian derivative** kernel, which smooths and differentiates in one operation. In 1D,

$$G(x) = \frac{1}{\sqrt{2\pi}\,\sigma} e^{-\frac{x^2}{2\sigma^2}}, \qquad \frac{dG}{dx} = -\frac{x}{\sigma^2}\, G(x),$$

and the 2D kernels for the X and Y directions are outer products of the 1D Gaussian and its derivative. As with the Gaussian kernel they are sampled versions of continuous functions and need a window of about $6\sigma + 1$ samples.

The standard deviation controls more than the blur. Large σ removes all fine, small-scale features and leaves only coarse ones, so a set of Gaussian derivative kernels with different σ responds to **edges at different scales** in the image.

{{fig gaussian_derivative}}

## The Laplacian kernel

The filters above are first-order. The **Laplacian** operator is the sum of the second-order partial derivatives in the two directions,

$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2},$$

and, as with Prewitt and Sobel, a finite-difference approximation gives a 3 × 3 kernel. Because it combines both directions, **one kernel** detects edges of any orientation instead of two. An intensity edge becomes a peak after the first derivative and a positive–negative pair after the second, with a **zero crossing** at the true edge. Applied to an image the Laplacian therefore produces pairs of bright and dark lines either side of each edge.

{{fig laplacian}}

{{selfcheck 02_01}}
