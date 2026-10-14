---
id: "01_01"
title: 2D convolution
section: 1
engines: [s1_convnets.convolution]
practice: [conv2d, conv_arith1, conv_arith3, conv_arith4, max_pool]
summary: The convolution operation at the heart of every convolutional neural network — convolution by hand, zero padding, stride, pooling, and the arithmetic used to calculate the output size.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- perform a **2D convolution** by hand: flip the kernel, pad, slide, multiply and sum;
- explain the purpose of **zero padding** and distinguish **valid**, **same** and **full** padding;
- explain the effect of **stride** and why a stride greater than one is used in networks;
- apply **max** and **average pooling**, and explain why a pooling layer has no parameters;
- calculate the **output size** of a convolution for any combination of input size, kernel size, padding and stride;
- explain how **cross-correlation** differs from convolution and when the two are the same.

## Introduction

Convolution allows an image to be modified by a **kernel**, a small array of numbers also called a filter. When a Gaussian kernel is convolved with an image, the image is blurred; other kernels can be used to sharpen an image, to find edges or to detect circular regions. Convolution is a **linear** operation, in which every point in the image is multiplied by the same set of values.

2D convolution is the core operation in convolutional neural networks, and the following lessons build directly on it. The operation itself is the same in a network as in image processing. The difference is in how the kernel is obtained: in classical image processing the kernel is specified in advance, whereas in a CNN the kernel values are **learned during training**.

## The 2D convolution operation

Before the mathematical definition is introduced, the manual procedure for convolution is described. It consists of three steps.

1. **Flip the kernel.** The elements of each row are reversed (a horizontal flip), and then the elements of each column are reversed (a vertical flip). The order of the two flips does not matter.
2. **Pad the image** with a ring of zeros so that the kernel can be placed at every position where it overlaps the image.
3. **Slide** the flipped kernel to every position, **multiply** element-wise with the values underneath and **sum** the products to give one output pixel.

{{fig conv2d_flip_slide}}

It should be noted that a symmetric kernel is unchanged by the flip. For example, the $2 \times 2$ kernel $\begin{pmatrix} 1 & 0 \\ 0 & 1\end{pmatrix}$ is identical to its own reversal, so step 1 has no effect for this kernel.

{{example conv2d seed=11}}

## The mathematical statement

For an input image $x$ and a kernel $h$, 2D convolution is defined as

$$(x * h)[m, n] = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} x[i, j]\, h[m-i,\, n-j]. \tag{1}$$

On the left-hand side, $[m, n]$ is the position of the output pixel. On the right-hand side, the minus signs inside the brackets **reverse** the kernel along both axes, and $[m, n]$ also sets the **shift** applied to the reversed kernel. The values $[i, j]$ index the image values that are multiplied with the kernel. A product is formed for every $i, j$ and all of the products are summed.

The sums run from $-\infty$ to $+\infty$, which ensures that every position at which the kernel overlaps the image is included. As the image has only a finite number of non-zero values, in practice only the shifts at which the kernel overlaps the image need to be evaluated.

Equation (1) can be difficult to visualise on its own, but it describes exactly the manual procedure given above.

## Zero padding and output size

As convolution is defined from $-\infty$ to $+\infty$, the kernel is, unless otherwise stated, moved to every position at which it just overlaps the image, and the image is padded with zeros as required. Zero padding introduces small edge effects, but these are not significant when a small kernel is applied to a large image.

Padding also changes the **size** of the output. In a neural network the padding is usually chosen to obtain a particular output size, very often one that matches the input size.

{{fig padding_modes}}

There is no single formula for the output size that covers every case; instead the appropriate relationship is selected for the case being considered. The relationships below are taken from Dumoulin and Visin, *A guide to convolution arithmetic for deep learning*, with input size $i$, kernel size $k$, padding $p$ and stride $s$.

| Case | Condition | Padding | Output size |
|---|---|---|---|
| **Valid** | $p = 0$, $s = 1$ | none | $o = (i - k) + 1$ |
| **Same** | odd $k$, $s = 1$ | $p = \lfloor k/2 \rfloor$ | $o = i$ |
| **Full** | $s = 1$ | $p = k - 1$ | $o = i + (k - 1)$ |
| No padding, strided | $p = 0$, any $s$ | none | $o = \left\lfloor \dfrac{i - k}{s} \right\rfloor + 1$ |
| General | any $p$, any $s$ | any | $o = \left\lfloor \dfrac{i + 2p - k}{s} \right\rfloor + 1$ |

With **valid** padding only real pixels are used, so a $3 \times 3$ image convolved with a $2 \times 2$ kernel gives a $2 \times 2$ output. With **same** padding, enough zeros are added to keep the output the same size as the input (one ring is needed for a $3 \times 3$ kernel); this is very frequently used in convolutional neural networks. **Full** padding follows the mathematical definition, so a $4 \times 4$ image with a $3 \times 3$ kernel requires two rings of zeros and produces a $6 \times 6$ output.

{{example conv_arith1 seed=5}}

{{example conv_arith3 seed=5}}

{{example conv_arith4 seed=5}}

## Stride

In the examples so far the kernel has been moved one pixel at a time. A step of 1 is assumed in the mathematical definition, but a larger **stride** is often used in convolutional networks, usually to downsample a feature map.

{{fig stride_grid}}

A larger stride results in a smaller output and, as there are fewer kernel positions, less computation. Stride is one of the two standard methods used to reduce spatial size in a network; the other is pooling.

## Pooling

Pooling is applied stepwise over the image in the same way as convolution, but **no kernel and no multiplication** are involved. Instead, the values in each window are aggregated: in **max pooling** the largest value in the window is taken, and in **average pooling** the mean is taken.

{{fig pooling_ops}}

Pooling aggregates information across a neighbourhood, and a stride of 2 is typically used so that the spatial size is halved. Two properties of pooling should be noted. A pooling layer has **no learnable parameters**, as there is no kernel to be learned. In addition, max pooling is **non-linear**, unlike convolution.

{{example max_pool seed=4}}

## Cross-correlation

A closely related operation, **cross-correlation**, is identical to convolution except that the kernel is *not* reversed. Convolution is used to modify one signal by another, whereas cross-correlation is a measure of **similarity**: the output is larger in regions where the image resembles the kernel.

{{fig corr_vs_conv}}

When the kernel is symmetric, reversal has no effect and the two operations give identical results. In deep learning the distinction is of limited importance. In frameworks such as PyTorch and TensorFlow, layers are labelled "convolution" but cross-correlation is actually computed. As the kernel weights are *learned* rather than specified, the network learns whatever kernel values are required. The distinction is therefore essential when a convolution is calculated by hand, but has little practical effect when the kernel is learned during training.

{{selfcheck 01_01}}
