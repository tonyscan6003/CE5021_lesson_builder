---
id: "02_03"
title: Corner detection
section: 2
engines: [s2_edges.corners]
practice: [harris]
summary: Corners as points where intensity changes in every direction, the Taylor expansion that leads to the second-moment matrix, the eigenvalue interpretation and the Harris corner response, and what the detector is and is not invariant to.
---

## Learning outcomes

After this lesson you should be able to:

- explain the **window-shift** idea behind corner detection and why edges and flat regions give small responses;
- describe how the Taylor expansion of $E(x, y)$ leads to the **second-moment matrix** $M$;
- interpret the **eigenvalues** of $M$ and compute the **Harris corner response** $R$;
- list the steps of the **Harris algorithm** and state its invariances.

## Basic corner detection

Corners are important image features. They are unaffected by changes in illumination and they are **rotationally invariant**: corners appear in the same place however the image is rotated. Extracted corners are used for image matching, stereo vision and tracking. A corner can be detected by looking at a small region through a **detector window** and monitoring how the intensity changes when the window is shifted slightly. On a **flat** region there is no change in any direction; on an **edge** there is change in one direction only; at a **corner** the intensity changes in multiple directions.

### The window function

For an image with intensity $I$, define $E$ as the change in intensity for a small shift $(x, y)$ of the window from position $(u, v)$:

$$E(x, y) = \sum_{(u, v) \in \text{window}} w(u, v)\,\big[ I(u + x, v + y) - I(u, v) \big]^2. \tag{1}$$

In the simplest case $w$ is a rectangular window. A basic detector evaluates $E$ for the eight possible single-pixel shifts at every point. On an edge, shifting along the edge gives $E = 0$; at a corner every shift gives a positive $E$. So **all** eight values of $E$ must exceed a threshold for a corner to be declared, which distinguishes corners from edges.

{{fig window_shift}}

## The Harris corner detector

The basic detector has drawbacks that the Harris detector (from the paper *A combined corner and edge detector*) addresses in three steps.

### Small shifts in any direction

The basic detector only allows shifts quantised to 45°. Harris considers all small shifts by taking a **Taylor expansion** of $E$ about $(x, y) = (0, 0)$, written compactly as zero-, first- and second-order terms:

$$E(x, y) \approx E(0, 0) + \begin{pmatrix} x & y \end{pmatrix} \nabla E + \frac{1}{2} \begin{pmatrix} x & y \end{pmatrix} H \begin{pmatrix} x \\ y \end{pmatrix}, \tag{2}$$

where $\nabla E$ is the gradient (the first-order partial derivatives) and $H$ the **Hessian** matrix of second-order partial derivatives, all evaluated at the origin.

Evaluating the terms with the chain rule: $E(0, 0)$ is zero because the intensity difference vanishes at zero shift, and so are the first derivatives, since each contains the same difference as a factor. The zero- and first-order terms disappear. Applying the product rule to the second derivatives creates two products in each, only one of which contains the intensity difference; at the origin that term vanishes and what remains is built from **products of first-order partial derivatives of the intensity**:

$$H \approx 2 \sum_{u, v} w \begin{pmatrix} X^2 & XY \\ XY & Y^2 \end{pmatrix}, \qquad X = \frac{\partial I}{\partial x},\; Y = \frac{\partial I}{\partial y}. \tag{3}$$

The partial derivatives $X$ and $Y$ are approximated by convolving the image with gradient kernels, for example Sobel.

### A Gaussian window

The second improvement replaces the rectangular window $w$ with a **Gaussian** window, which reduces the effect of noise on the detection.

### The second-moment matrix

The multiply-and-sum in equation (3) is a convolution of the derivative products with the window function. Writing (and dropping the unimportant factor of 2)

$$A = w * X^2, \qquad B = w * Y^2, \qquad C = w * XY, \qquad M = \begin{pmatrix} A & C \\ C & B \end{pmatrix}, \tag{4}$$

$M$ is the **second-moment matrix**. It summarises the predominant gradient direction in the neighbourhood of a point and how coherent that direction is.

### Eigenvalues of M

The eigenvalues $\lambda_1, \lambda_2$ of $M$ are proportional to the local curvature of $E$ in the two principal directions:

- both small: a **flat** region;
- one large, one small: curvature in one direction only, an **edge**;
- both large: curvature in two directions, a **corner**.

The eigenvalues do not change when the image is rotated (only the eigenvectors do, and they are not used), which is why the detector is rotation invariant.

{{fig second_moment}}

### The corner response function

Rather than perform an eigenvalue decomposition at every pixel, Harris uses the **determinant and trace** of $M$, which are simple to compute from $A$, $B$ and $C$:

$$R = \det M - k\,(\operatorname{trace} M)^2 = (AB - C^2) - k\,(A + B)^2, \tag{5}$$

with $k$ a small constant, typically 0.04 to 0.06. Since $\det M = \lambda_1 \lambda_2$ and $\operatorname{trace} M = \lambda_1 + \lambda_2$, $R$ is large and positive at corners, negative on edges and small in flat regions.

{{fig eigenvalue_regions}}

{{example harris seed=6}}

## The Harris algorithm

Despite the derivation the algorithm is simple:

1. compute the gradient approximations $X$ and $Y$ at each pixel, for example with Sobel kernels;
2. form the products $X^2$, $Y^2$ and $XY$ and convolve each with a Gaussian window (small σ) to obtain $A$, $B$ and $C$;
3. compute $R$ at every pixel;
4. **threshold** $R$ to keep candidate corners and reject edges;
5. find the **local maxima** of $R$ and mark them as corners.

On a grid of windows the corner response is positive at the window corners and negative along the long edges, and the local maxima land on the corners. On irregular patterns the detector responds most strongly to small curved regions; the size of the Gaussian window sets the scale of corner it responds to.

{{fig harris_response}}

## Invariance

Because the response depends only on eigenvalues, the Harris detector is invariant to **rotation** and **translation**: the same corners are found in a rotated or shifted image. It is **not invariant to scale**: the window has a fixed size, so it sees corners of one size only, and a large smooth curve is reported as an edge rather than a corner. The pattern of corners on an object can be used for matching and tracking; extracting many corners for such purposes is called **feature detection**, the subject of the next section.

{{fig harris_invariance}}

{{selfcheck 02_03}}
