---
id: "01_03"
title: Image sampling and interpolation
section: 1
engines: [s1_images.sampling_interpolation]
practice: [bi_lin]
summary: The sampling theorem and aliasing, why sampled signals have image frequencies, interpolation as convolution, and the practical methods used to resize images.
---

## Learning outcomes

After this lesson you should be able to:

- state the **Nyquist–Shannon** sampling theorem and explain **aliasing** in signals and images;
- explain why a sampled signal's spectrum repeats and what **image frequencies** are;
- describe **interpolation as convolution** with a kernel and compare box, triangle and sinc kernels;
- explain why **resampling** can alias, especially when reducing size;
- compute **bilinear interpolation** by the area-weighting method.

## Sampling and aliasing

Digital images are captured by sampling: the light falling on the sensor is sampled by the pixels, at an instant set by the shutter. To understand what sampling does we will work with one-dimensional signals throughout most of this lesson.

The **Nyquist–Shannon sampling theorem** states that a sampled signal can only represent frequencies up to **half the sample rate**, $f_s/2$. Any higher frequency is **aliased**: its samples are indistinguishable from those of a lower frequency. In the frequency domain the high frequency $f_1$ and its alias $f_{alias}$ are mirror images about $f_s/2$. Aliasing is mitigated by applying a **low-pass filter before sampling**, which attenuates $f_1$ so that the alias it produces is much weaker.

{{fig aliasing_1d}}

## Frequency information in images

In a 1D signal, frequency measures how fast the amplitude changes with time. In an image it measures how fast **pixel intensity changes with position**. Regions of constant or slowly varying intensity, such as sky, contain low frequencies. Rapid changes, edges and repeating patterns such as zebra stripes or brickwork contain high frequencies.

{{fig image_frequency}}

### Aliasing in images

Capturing fine detail needs a sensor with enough pixels: each pixel averages the light over its own area, and a coarse grid cannot follow a fine pattern. When it cannot, aliasing appears as broad ripples, known as **moiré**, in bricks, fabrics and stripes. Just as in the 1D case the remedy is a low-pass filter before sampling; in a camera this is an **optical low-pass filter** placed over the sensor that slightly blurs the finest patterns.

{{fig image_aliasing}}

{{fig moire_photo}}

## The spectrum of a digital signal

A second consequence of the sampling theorem is that a given set of samples represents not one but **many** frequencies: the low frequency we intend, and every higher frequency that aliases onto it. The spectrum of a sampled signal is therefore periodic, with **image frequencies** repeating around every multiple of $f_s$. These images do not interfere with processing in the digital domain, but as we will now see they matter as soon as we try to reconstruct a continuous signal.

{{fig digital_spectrum}}

## Reconstruction of signals

Often we want the continuous signal back, for example to drive a loudspeaker. Conceptually this means filling in the missing values between samples with an **interpolation filter**. Can the original sinusoid be recovered exactly? The **Whittaker–Shannon interpolation formula** says yes, provided the interpolation kernel is the **sinc** function,

$$\operatorname{sinc}(x) = \frac{\sin \pi x}{\pi x}.$$

In the frequency domain the sinc is a perfect band-limiting filter: its response drops to zero exactly at $f_s/2$, so every image frequency is removed and only the original frequency survives.

### Practical interpolation filters

The sinc kernel is expensive: it decays slowly, so representing it accurately needs a large kernel. Two cheap alternatives are

- **replication**, which holds each sample value until the next one, and is equivalent to convolving the samples with a **box** kernel;
- **linear interpolation**, which joins consecutive samples with straight lines, equivalent to a **triangle** kernel.

Both are less effective at removing image frequencies. The box kernel's frequency response falls off gradually and has side lobes extending far beyond $f_s/2$, so the image frequencies are only partly suppressed and distort the reconstruction.

{{fig interp_kernels}}

### Resampling

To change the sample rate of a signal we reconstruct it and then sample it again at the new rate. If a poor interpolation filter was used, the residual image frequencies are themselves sampled, producing aliasing and distortion. This applies whether the rate goes up or down, but **decreasing** the rate is far more prone to aliasing.

{{fig signal_resampling}}

## Image resizing

Resizing an image is a change of sample rate in two dimensions. Rather than fully reconstructing a continuous image, 2D interpolation is applied directly to produce intensity values at the new pixel positions. This is equivalent to interpolation followed by resampling, so aliasing can still occur. Common methods, in increasing order of cost:

| Method | Neighbours used | 1D equivalent |
|---|---|---|
| Nearest neighbour | 1 | replication (box) |
| Bilinear | 4 | linear (triangle) |
| Bicubic | 16 | cubic |
| Sinc / Lanczos | many | sinc |

Nearest-neighbour interpolation assigns each new pixel the value of the closest old pixel. It can enlarge or shrink an image, and shrinking with it introduces exactly the aliasing seen earlier.

{{fig interp_methods}}

### Bilinear interpolation

Bilinear interpolation is the most common resizing method. The intensity at a new pixel is a linear combination of the **four** surrounding original pixels, with more weight given to the nearer ones. The easiest way to see it is geometrically: the new pixel divides the $2 \times 2$ cell into four rectangles, and each original pixel's weight is the area of the rectangle **diagonally opposite** it. The interpolated value is the weighted sum divided by the total area:

$$I(x_i, y_j) = \frac{\sum_c I_c\, A_c}{\sum_c A_c}.$$

{{fig bilinear_area}}

{{example bi_lin seed=4}}

### Examples of interpolation

The differences between methods on a high-resolution photo can be subtle and may need side-by-side comparison. Nearest-neighbour produces jagged edges; bilinear, bicubic and sinc interpolation are progressively smoother. The sinc filter, however, introduces faint **ripples** beside sharp edges. In theory the sinc is perfect, but in practice the kernel must be truncated, which distorts its frequency response and causes ringing, the **Gibbs phenomenon**.

{{selfcheck 01_03}}
