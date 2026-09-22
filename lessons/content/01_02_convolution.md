---
id: "01_02"
title: Image processing with convolution
section: 1
engines: [s1_images.convolution]
practice: [conv1d, conv2d, conv_arith1, conv_arith2, conv_arith3, conv_arith4, conv_arith5, conv_arith6]
summary: Linear filtering with kernels, the definition of convolution in 1D and 2D, how to do it by hand, and how to work out the output size for valid, same and full padding.
---

## Learning outcomes

After this lesson you should be able to:

- explain what makes a filter **linear** and what a **kernel** is;
- compute a **1D convolution** from its definition, including kernel reversal and shifting;
- perform a **2D convolution** by hand: flip the kernel, slide, multiply and sum;
- distinguish convolution from **cross-correlation** and say when they coincide;
- calculate the **output size** of a convolution for valid, same and full zero padding.

## Linear image processing

Images are filtered to sharpen or blur them, or to find and enhance edges. The processing in this lesson is **linear**: each pixel of the output is a linear combination of the pixels in a small neighbourhood of the input. The weights of that combination are held in a **kernel**, a small array of numbers that represents the mathematical filter being applied. Convolving the kernel with the image produces the filtered image.

{{fig kernel_gallery}}

## 1D convolution

What is convolution? It is the modification of one signal by another; equivalently, it is the result of passing a sequence through a **linear time-invariant (LTI) system**, which is exactly what filtering is. *Linear* means the filter obeys superposition; *time-invariant* means the kernel does not change as the input passes through it.

For a sequence $x[n]$ and kernel $h[n]$ the convolution is

$$(x * h)[n] = \sum_{k=-\infty}^{\infty} x[k]\, h[n-k]. \tag{1}$$

Read on its own the equation is hard to interpret. Rewriting the kernel term as

$$h[n-k] = h[-(k-n)] \tag{2}$$

makes two operations visible. The minus sign outside the bracket **reverses** the kernel. The value of $n$ **shifts** it: a positive $n$ delays the reversed kernel (moves it right) and a negative $n$ advances it (moves it left).

### Worked example: reversal and shift

Take a kernel $h[k]$ with values $1, 0, 2$ at $k = -1, 0, 1$ and zero elsewhere.

- $h[k-n]$ with $n = 1$: evaluating at $k = 0, 1, 2$ gives $h[-1], h[0], h[1] = 1, 0, 2$. The kernel is **delayed** by one sample.
- $h[-(k-n)]$ with $n = 1$: evaluating at $k = 0, 1, 2$ gives $h[1], h[0], h[-1] = 2, 0, 1$. The kernel is delayed by one sample **and reversed**.

### Computing the full convolution

To evaluate equation (1) we write both sequences in terms of the dummy variable $k$. For each output index $n$ we form the reversed, shifted kernel $h[n-k]$, multiply it sample by sample with $x[k]$, and sum. Doing this for every $n$ amounts to **sliding the reversed kernel along the sequence** and recording one output value at each position. Verify a few of the values in the figure yourself.

{{fig conv1d_slide}}

{{example conv1d seed=3}}

## Convolution and cross-correlation

If the kernel is **not** reversed we obtain a different operation, **cross-correlation**:

$$(x \star h)[n] = \sum_{k} x[k]\, h[k-n].$$

Cross-correlation measures the *similarity* of two signals, whereas convolution is the *modification* of one signal by another. When the kernel is symmetric, $h[k] = h[-k]$, reversal changes nothing and the two operations give identical results. Many kernels in image processing are symmetric, which is why the distinction is often blurred in practice.

## 2D convolution

For images the same idea applies in two dimensions. With an image $x$ and kernel $h$,

$$(x * h)[m, n] = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} x[i, j]\, h[m-i,\, n-j]. \tag{3}$$

The minus signs inside the brackets reverse the kernel along **both** axes; $[m, n]$ is the position of the output pixel and also the shift applied to the reversed kernel. At each shift the overlapping values are multiplied and summed. The infinite sums simply guarantee that every position where the kernel overlaps the image is included.

### Symmetric kernels

Reversing a 2D kernel means reversing the order of its rows *and* the order of its columns (the order in which you do the two does not matter). A **symmetric kernel** is unchanged by this, so once again convolution and cross-correlation coincide.

{{fig corr_vs_conv}}

### Doing 2D convolution by hand

The manual procedure is straightforward:

1. **Flip the kernel**: reverse each row (horizontal flip) then reverse each column (vertical flip).
2. **Pad** the image with a ring of zeros so the kernel can be placed where it just overlaps the image.
3. **Slide** the flipped kernel to every position, **multiply** element-wise with the pixels underneath and **sum** to give one output pixel.

{{fig conv2d_flip_slide}}

{{example conv2d seed=11}}

## Zero padding and output size

The convolution sum is defined from $-\infty$ to $\infty$, so unless told otherwise we should slide the kernel to every position where it just overlaps the image, padding with zeros as required. Padding creates small edge effects, but with a small kernel on a large image they are negligible. Padding also changes the size of the output. Three cases are common; the formulae come from Dumoulin and Visin's *A guide to convolution arithmetic for deep learning*, with input size $i$, kernel size $k$, padding $p$ (rings of zeros on each side) and stride $s$ (how far the kernel moves each step; the mathematical definition assumes $s = 1$).

| Case | Padding | Output size (unit stride) |
|---|---|---|
| **Valid** | $p = 0$ | $o = (i - k) + 1$ |
| **Same** (odd $k$) | $p = \lfloor k/2 \rfloor$ | $o = i$ |
| **Full** | $p = k - 1$ | $o = i + (k - 1)$ |
| General | any $p$, any $s$ | $o = \left\lfloor \dfrac{i + 2p - k}{s} \right\rfloor + 1$ |

**Valid** padding uses only real pixels, so a $3 \times 3$ image convolved with a $2 \times 2$ kernel gives a $2 \times 2$ output. **Same** padding chooses exactly enough zeros to keep the output the size of the input; a $3 \times 3$ kernel needs one ring. **Full** padding follows the mathematical definition, so a $4 \times 4$ image with a $3 \times 3$ kernel needs two rings and produces a $6 \times 6$ output.

{{fig padding_modes}}

{{example conv_arith1 seed=5}}

{{example conv_arith3 seed=5}}

{{example conv_arith4 seed=5}}

In practice we use OpenCV or SciPy to perform convolution, but being able to predict the output size, and to work small cases by hand, is essential for understanding what those functions are doing.

{{selfcheck 01_02}}
