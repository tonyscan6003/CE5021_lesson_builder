---
id: "01_04"
title: Linear filters, image noise and non-linear filters
section: 1
engines: [s1_images.linear_nonlinear]
practice: [sep_conv, med_filt]
summary: The Gaussian kernel and how to sample it, separable convolution, the properties of linear filters, the two common noise models, and the median and bilateral filters that beat linear filters at removing noise.
---

## Learning outcomes

After this lesson you should be able to:

- construct a **sampled Gaussian kernel** of the right size for a given σ;
- explain **separable convolution** and why it is cheaper than direct 2D convolution;
- state the **superposition**, **shift invariance**, **commutative** and **associative** properties of linear filtering;
- describe the **Gaussian** and **impulse (salt-and-pepper)** noise models;
- explain how the **median** and **bilateral** filters work and why they are non-linear.

## Linear image processing

As in the previous lesson, a linear filter produces each output pixel as a linear combination of the input pixels in a small neighbourhood, and the filtering is carried out by convolving the image $I$ with a kernel $H$. Choosing the kernel selects the effect: blurring, sharpening or edge enhancement.

## The Gaussian kernel

The Gaussian is the most important kernel in image processing. It is the first step of many algorithms, used to remove noise, and it introduces no artefacts into the image. As we will see in Section 3 it is also used to build *scale spaces*, sequences of progressively blurred images. The continuous 2D Gaussian with standard deviation σ is

$$G(x, y) = \frac{1}{2\pi\sigma^2}\, e^{-\frac{x^2 + y^2}{2\sigma^2}}.$$

The larger σ, the wider the bell and the stronger the blur when the kernel is convolved with an image. Heavy blurring removes all fine detail and leaves only large-scale structure.

### Sampling the Gaussian

The Gaussian is a continuous function, but a kernel is a finite array of numbers, so we must sample it. Consider the 1D case with zero mean,

$$g(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\, e^{-\frac{x^2}{2\sigma^2}}.$$

We evaluate it at integer $x$, corresponding to pixel centres, over a finite window. The rule of thumb is a window of

$$M = 6\sigma + 1$$

samples, covering $-M/2$ to $M/2$ about the centre, which captures the bell to within a fraction of a percent. The 2D kernel samples the 2D Gaussian over the same range in both directions. The sampled kernel is normalised to sum to one so that it does not change the overall brightness.

{{fig gaussian_kernel}}

## Separable kernels

A 2D kernel is **separable** if it can be written as the **outer product** of two 1D kernels:

$$H = u\, v^{T}.$$

Multiplying a column vector by a row vector gives a matrix; for a 3-element $u$ and $v$ the result is $3 \times 3$. Many kernels in image processing are symmetric and are formed as the outer product of a 1D kernel with its own transpose. The outer product of two 1D vectors is also equal to their convolution.

### Separable convolution

Convolving an image with a large kernel, such as a Gaussian with large σ, is expensive: every output pixel needs $k^2$ multiplications. If the kernel is separable,

$$L = I * H = I * (H_1 H_1^{T}) = (I * H_1) * H_1^{T}, \tag{1}$$

so the single 2D convolution can be replaced by two 1D convolutions, first down the columns and then along the rows. Each pass needs only $k$ multiplications per pixel, $2k$ in total, a large saving for large kernels. In the worked example a symmetric $3 \times 3$ kernel is factorised and applied in two passes, with a ring of zero padding to keep the output the same size.

{{fig separable_kernel}}

{{example sep_conv seed=2}}

## Properties of linear filters

### Superposition

Linear filters obey the **superposition principle**. The response to a sum of images is the sum of the responses:

$$(I_1 + I_2) * H = I_1 * H + I_2 * H. \tag{2}$$

The same holds with one image and two kernels, $I * (H_1 + H_2) = I * H_1 + I * H_2$, and scaling an image before convolution is the same as scaling afterwards, $(aI) * H = a\,(I * H)$.

{{fig superposition}}

### Shift invariance

If the input is shifted spatially, the output is shifted by the same amount. This follows from convolution applying the same kernel at every position without distorting the image dimensions.

{{fig shift_invariance}}

### Commutativity and associativity

Convolution is **commutative**, $I * H_1 * H_2 = I * H_2 * H_1$, so the order in which kernels are applied does not matter, and **associative**,

$$((I * H_1) * H_2) * H_3 = I * (H_1 * H_2 * H_3), \tag{3}$$

so a chain of kernels can be combined into one kernel first and applied once.

{{fig commutativity}}

## Noise in images

Storage and transmission of digital images are robust, so most noise originates during acquisition in the camera.

**Gaussian noise** has a Gaussian probability distribution,

$$P(g) = \frac{1}{\sqrt{2\pi}\,\sigma_n}\, e^{-\frac{(g - \bar g)^2}{2\sigma_n^2}},$$

where $g$ is the noise value and $P(g)$ its probability. Note the difference from the Gaussian *kernel*: there we sampled the shape of the bell to make a filter; here we draw random *samples* $n(x, y)$ with probability $P(g)$ and add them to the image. The usual source is the sensor electronics that convert charge to voltage and amplify it. With zero mean, the amount of noise depends only on $\sigma_n$.

**Impulse noise** arises from large impulses during digitisation that set pixels to fixed extreme values, usually 0 or 255. Because it contains both black and white pixels it is called **salt-and-pepper** noise. Its probability density is non-zero only at the two values $a$ and $b$, and it can be modelled by replacing a random fraction of the pixels.

{{fig image_noise}}

## Linear versus non-linear filters

Everything so far has been linear: implemented by convolution, obeying superposition and shift invariance. **Non-linear filters** do not obey superposition. They are often more expensive than a convolution, but for some tasks, particularly removing noise without damaging edges, they perform much better. Two examples follow: the median filter and the bilateral filter.

### The median filter

The median filter resembles a mean (average) filter but is not affected by outliers and preserves edges better. Under a $3 \times 3$ **structuring element** (the term used instead of *kernel* for non-linear operations) the nine pixel values are sorted and the middle value, the fifth of nine, is output. The operation is applied at every position of the image just like a convolution, but with a sorting step instead of a multiply-and-sum. Salt-and-pepper noise is removed almost perfectly because the impulses are outliers that never reach the middle of the sorted list. Applying the median repeatedly removes fine texture while the main edges survive.

{{fig median_filter}}

{{example med_filt seed=6}}

### The bilateral filter

The Gaussian filter averages pixels that are close together *in space*, so when it straddles an edge it mixes both sides and blurs the edge. In many applications edges carry the useful information and we want to keep them. The **bilateral filter** adds a **range** term: a second Gaussian, this time in *intensity*, that gives weight only to pixels whose values are close to the centre pixel's. The filter weight at a neighbour is the product of the spatial Gaussian and the range Gaussian. Near an edge the range term cuts off the far side, so the effective neighbourhood becomes a half-disc and the edge is not smeared. Because the weights depend on the image content and change at every pixel, the filter is non-linear even though it is built from two Gaussians.

{{fig bilateral_filter}}

{{selfcheck 01_04}}
