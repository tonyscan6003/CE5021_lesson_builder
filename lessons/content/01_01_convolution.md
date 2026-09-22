---
id: "01_01"
title: 2D convolution
section: 1
engines: [s1_convnets.convolution]
practice: [conv2d, conv_arith1, conv_arith3, conv_arith4, max_pool]
summary: The convolution operation at the heart of every convolutional neural network — doing it by hand, zero padding, stride, pooling, and the arithmetic that predicts the output size.
---

## Learning outcomes

After this lesson you should be able to:

- perform a **2D convolution** by hand: flip the kernel, pad, slide, multiply and sum;
- explain what **zero padding** is for and distinguish **valid**, **same** and **full** padding;
- explain what **stride** does and why networks use a stride greater than one;
- apply **max** and **average pooling**, and say why a pooling layer has no parameters;
- calculate the **output size** of a convolution for any combination of input size, kernel size, padding and stride;
- explain how **cross-correlation** differs from convolution and when the two coincide.

## Introduction

Convolution allows an image to be modified by a **kernel**, a small array of numbers also called a filter. Convolving a Gaussian kernel with an image blurs it; other kernels sharpen, find edges, or pick out circular patches. Convolution is a **linear** operation: every point in the image is multiplied by the same set of values.

2D convolution is the core operation in convolutional neural networks, and the next few lessons build directly on it. What changes in a network is not the operation but where the numbers come from: in classical image processing the kernel is chosen in advance, whereas in a CNN the kernel values are **learned during training**.

## The 2D convolution operation

Before looking at the mathematical definition, it is worth seeing that performing convolution by hand is straightforward. There are three steps.

1. **Flip the kernel.** Reverse the elements of each row (a horizontal flip), then reverse the elements of each column (a vertical flip). The order of the two does not matter.
2. **Pad the image** with a ring of zeros so the kernel can be placed wherever it overlaps the image.
3. **Slide** the flipped kernel to every position, **multiply** element-wise with the values underneath and **sum** to give one output pixel.

{{fig conv2d_flip_slide}}

Note that a symmetric kernel is unchanged by the flip. The $2 \times 2$ kernel $\begin{pmatrix} 1 & 0 \\ 0 & 1\end{pmatrix}$, for instance, is identical to its own reversal, so for that kernel step 1 does nothing.

{{example conv2d seed=11}}

## The mathematical statement

With an input image $x$ and a kernel $h$, 2D convolution is defined as

$$(x * h)[m, n] = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} x[i, j]\, h[m-i,\, n-j]. \tag{1}$$

On the left, $[m, n]$ is the position of the output pixel. On the right, the minus signs inside the brackets **reverse** the kernel along both axes, and $[m, n]$ also sets the **shift** applied to the reversed kernel. The values $[i, j]$ index the image values to be multiplied with the kernel. For every $i, j$ a product is formed and all the products are summed.

The sums run from $-\infty$ to $+\infty$, which simply guarantees that every position where the kernel overlaps the image is included. Since the image has only a finite number of non-zero values, in practice we need only evaluate the shifts where the kernel actually overlaps it.

Encountered as equation (1) alone, convolution can be hard to visualise — but the equation describes exactly the manual procedure above, and nothing more.

## Zero padding and output size

Because convolution is defined from $-\infty$ to $+\infty$, unless told otherwise we should slide the kernel to every position where it just overlaps the image, padding with zeros as needed. Padding with zeros creates small edge effects, but these are not significant for a small kernel on a large image.

Padding also changes the **size** of the output. In a neural network we usually choose the padding deliberately to obtain a particular output size — very often one that matches the input.

{{fig padding_modes}}

There is no single universal formula for the output size; instead we pick the appropriate relationship for the case at hand. These come from Dumoulin and Visin's *A guide to convolution arithmetic for deep learning*, with input size $i$, kernel size $k$, padding $p$ and stride $s$.

| Case | Condition | Padding | Output size |
|---|---|---|---|
| **Valid** | $p = 0$, $s = 1$ | none | $o = (i - k) + 1$ |
| **Same** | odd $k$, $s = 1$ | $p = \lfloor k/2 \rfloor$ | $o = i$ |
| **Full** | $s = 1$ | $p = k - 1$ | $o = i + (k - 1)$ |
| No padding, strided | $p = 0$, any $s$ | none | $o = \left\lfloor \dfrac{i - k}{s} \right\rfloor + 1$ |
| General | any $p$, any $s$ | any | $o = \left\lfloor \dfrac{i + 2p - k}{s} \right\rfloor + 1$ |

**Valid** padding uses only real pixels, so a $3 \times 3$ image convolved with a $2 \times 2$ kernel gives a $2 \times 2$ output. **Same** padding adds exactly enough zeros to keep the output the size of the input — a $3 \times 3$ kernel needs one ring — and is very frequently used in convolutional neural networks. **Full** padding follows the mathematical definition strictly, so a $4 \times 4$ image with a $3 \times 3$ kernel needs two rings and produces a $6 \times 6$ output.

{{example conv_arith1 seed=5}}

{{example conv_arith3 seed=5}}

{{example conv_arith4 seed=5}}

## Stride

So far the kernel has moved one pixel at a time. A step of 1 is what the mathematical definition assumes, but real convolutional networks often use a larger **stride**, usually to downsample a feature map.

{{fig stride_grid}}

A larger stride gives a smaller output and, because there are fewer kernel positions, less computation. This is one of the two standard ways a network reduces spatial size; the other is pooling.

## Pooling

Pooling is applied stepwise over the image in the same way as convolution, but there is **no kernel and no multiplication**. Instead the values in each window are aggregated: **max pooling** takes the largest value in the window, **average pooling** takes the mean.

{{fig pooling_ops}}

Pooling aggregates information across a neighbourhood, and a stride of 2 is typical so that the spatial size is halved. Two properties are worth noting. A pooling layer has **no learnable parameters**, because there is no kernel to learn. And max pooling is **non-linear**, unlike convolution.

{{example max_pool seed=4}}

## Cross-correlation

A closely related operation, **cross-correlation**, is identical to convolution except that the kernel is *not* reversed. Where convolution modifies one signal by another, cross-correlation measures their **similarity**: the output is larger in regions where the image resembles the kernel.

{{fig corr_vs_conv}}

When the kernel is symmetric, reversal changes nothing and the two operations give identical results. This matters less than it might seem in deep learning. Frameworks such as PyTorch and TensorFlow label their layers "convolution" but in fact compute cross-correlation — and because the kernel weights are *learned* rather than specified, the network simply learns whatever values it needs. The distinction is essential when you are working a convolution by hand, and essentially cosmetic once the kernel is being trained.

{{selfcheck 01_01}}
