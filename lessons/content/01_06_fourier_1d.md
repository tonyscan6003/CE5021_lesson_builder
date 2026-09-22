---
id: "01_06"
title: 1D Fourier series and the discrete Fourier transform
section: 1
engines: [s1_images.fourier_1d]
summary: An intuitive introduction to sinusoids, complex sinusoids and positive and negative frequencies, the Fourier series, the discrete Fourier transform with its mirror frequencies and leakage, and the convolution theorem.
---

## Learning outcomes

After this lesson you should be able to:

- describe a sinusoid by its **amplitude, frequency and phase** and rewrite it in **in-phase and quadrature** form;
- interpret a **complex sinusoid** as circular motion and explain **positive and negative frequencies**;
- explain how a **Fourier series** builds a periodic signal from harmonics and read its **spectrum**;
- describe the **DFT** of N samples, including the mirror frequencies above N/2 and **spectral leakage**;
- state the **convolution theorem** and explain why it allows filtering in the frequency domain.

This lesson uses a minimal amount of mathematics. The aim is an intuitive picture of the Fourier series and the DFT, because they give an alternative way to filter images, which we take up in the next lesson.

## Periodic signals

A signal is **periodic** if it repeats with some period $T$: $x(t) = x(t + T)$, and therefore also at every multiple of $T$. A sine wave is periodic, a pulse train is periodic, and so is any irregular waveform that repeats. A periodic signal can contain components that vary much faster than its period. As we will see, a sum of sinusoids at multiples of the fundamental frequency can approximate such a signal.

{{fig periodic_signals}}

## Sinusoids

A sinusoid is described by three parameters:

$$x(t) = A \sin(\omega t + \varphi).$$

- The **amplitude** $A$ is how far it swings above and below zero.
- The **angular frequency** $\omega = 2\pi f$, in radians per second, is how quickly it repeats; $f$ is the frequency in hertz and the period is $1/f$.
- The **phase** $\varphi$, in radians, fixes where the sinusoid sits in time.

### Phase

A sine wave is zero at $t = 0$. A positive phase shifts it to the left, *advancing* it; a negative phase shifts it right, *delaying* it. Sine and cosine have the same shape: the cosine reaches its full amplitude at $t = 0$, so a sine advanced by $\varphi = \pi/2$ is a cosine.

{{fig sinusoid_params}}

### In-phase and quadrature components

A phase-shifted sinusoid can be written as the sum of a cosine and a sine of the same frequency **without** phase terms, using two separate amplitudes instead:

$$A \sin(\omega t + \varphi) = a \cos \omega t + b \sin \omega t, \qquad a = A \sin\varphi,\; b = A\cos\varphi.$$

We call the $\cos \omega t$ term the **in-phase** component and the $\sin \omega t$ term the **quadrature** component, since it is 90° out of phase with the cosine. The Fourier series uses exactly this form.

{{fig iq_components}}

## Complex sinusoids

A third representation gathers the in-phase and quadrature parts into one expression. By **Euler's identity**,

$$e^{j\omega t} = \cos \omega t + j \sin \omega t.$$

The real part is the in-phase cosine and the imaginary part the quadrature sine. Geometrically $e^{j\omega t}$ describes a point moving around a circle in the complex plane: at each time $t$ its projection on the real axis is $\cos \omega t$ and on the imaginary axis $\sin \omega t$. The complex sinusoid is not a single oscillation but **circular motion**, from which both a real cosine and an imaginary sine can be read off.

### Positive and negative frequency

Euler's identity also gives the cosine and sine directly in terms of exponentials:

$$\cos \omega t = \frac{e^{j\omega t} + e^{-j\omega t}}{2}, \qquad \sin \omega t = \frac{e^{j\omega t} - e^{-j\omega t}}{2j}.$$

$e^{j\omega t}$ rotates anticlockwise and $e^{-j\omega t}$ clockwise, which is the origin of the idea of **positive and negative frequencies**. Adding the two, the imaginary sine parts cancel and the real parts reinforce to give $2\cos \omega t$; subtracting them, the real parts cancel instead. This cancelling and reinforcing of real and imaginary terms is exactly what happens in the exponential form of the Fourier series.

{{fig complex_phasor}}

## The 1D Fourier series

Any periodic signal $x(t)$ with period $T$ can be decomposed into a sum of sines and cosines:

$$x(t) = a_0 + \sum_{k=1}^{\infty} \left[ a_k \cos\left(\frac{2\pi k t}{T}\right) + b_k \sin\left(\frac{2\pi k t}{T}\right) \right]. \tag{1}$$

The term $a_0$ is an offset. Inside the sum, the sine and cosine at the $k$-th multiple of the fundamental frequency $2\pi/T$ are scaled by separate coefficients $a_k$ and $b_k$: together they are the in-phase and quadrature form of one sinusoid. We will not derive how the coefficients are calculated; the point is that they measure how much of each harmonic is needed to compose the waveform. For a square wave only the odd harmonics are needed, and adding more of them improves the approximation. The ripples narrow, but a small overshoot at each edge never fully disappears: this is the **Gibbs phenomenon**, seen whenever an edge or discontinuity is approximated by a Fourier series.

{{fig fourier_series}}

### Exponential form

The series can equally be written as a sum of complex sinusoids,

$$x(t) = \sum_{k=-K}^{K} c_k\, e^{j 2\pi k t / T}, \tag{2}$$

which now runs over **both positive and negative** $k$, because each complex sinusoid rotates in only one direction and both directions are needed to reconstruct a real signal. The complex coefficients $c_k$ determine which parts of the exponentials add and which cancel. For the square wave the imaginary parts cancel and the reconstruction is the same as before.

## The frequency spectrum

A Fourier series is usually visualised as a **spectrum**: the magnitude of the coefficients plotted against frequency. The series only has components at the discrete frequencies $k \cdot 2\pi/T$, so the spectrum is a set of lines. In the trigonometric form the magnitude at harmonic $k$ is $\sqrt{a_k^2 + b_k^2}$; for the square wave the $a_k$ are zero and only the odd $b_k$ survive. In the exponential form the lines appear at both positive and negative $k$ with half the magnitude, so the total power is the same.

The spectrum is a **complete description of the signal**. A signal can be specified either by its value at each instant (the waveform) or by how much of each frequency it contains; these are two equivalent views of the same information, and the tool that moves between them is the Fourier transform.

{{fig frequency_spectrum}}

## The discrete Fourier transform

The Fourier series is defined for continuous signals, but signals and images are processed digitally. The **discrete Fourier transform** (DFT) is the digital version. It takes $N$ samples $x[n]$, treats them as one period of a periodic signal, and represents them as

$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k]\, e^{j 2\pi k n / N}. \tag{3}$$

Compared with the Fourier series, the period $T$ has become the number of samples $N$, continuous time $t$ has become the sample index $n$, and the sum runs over **positive $k$ only**. The DFT is computed in practice by the **fast Fourier transform** (FFT) algorithm, used by NumPy and every other numerical package.

### Mirror frequencies

Taking 16 samples of a sine with two cycles in the window gives a spectrum with the expected peak at $k = 2$, but also a second peak at $k = 14$. The DFT does not use negative frequencies; instead its positive range extends beyond $N/2$, and the values above $N/2$ are **aliases of the negative frequencies**. A phasor at $k = -2$ rotating clockwise reaches, after one sample, the same point as a faster phasor at $k = 14$ rotating anticlockwise. The spectrum of a real signal is therefore mirrored about $k = N/2$.

{{fig dft_16}}

### Windowing and spectral leakage

The DFT implicitly assumes the signal is periodic with period $N$, as if a rectangular window had been cut from it. If the window does not contain a whole number of cycles, the implied signal has a jump where the window repeats, and the energy is spread across many frequency bins: **spectral leakage**. It is unavoidable whenever the signal does not have an integer number of periods in the window. Multiplying the samples by a **window function** that tapers to zero at both ends, such as a Hann window, confines the leakage to the bins next to the true frequency, at the cost of attenuating the signal. Images rarely consist of a single frequency that fits the image exactly, so we will see leakage again in 2D.

{{fig dft_leakage}}

## The convolution theorem

Of the many properties of Fourier transforms, one matters most for image filtering. The **convolution theorem** states that convolution in the time domain is equivalent to **multiplication** in the frequency domain:

$$\mathrm{DFT}(x * h) = \mathrm{DFT}(x) \cdot \mathrm{DFT}(h). \tag{4}$$

Filtering can therefore be done by transforming the signal, multiplying its spectrum by the desired filter response, and transforming back, instead of convolving with a kernel. The example applies a brick-wall low-pass filter to a square wave in the spectral domain. Note that the filter must be **mirrored about the centre of the spectrum** to act on the negative frequencies too. The reconstructed waveform is identical to convolving the original with the kernel corresponding to the brick-wall filter. In the next lesson we will use exactly this method to filter images.

{{fig convolution_theorem}}

{{selfcheck 01_06}}
