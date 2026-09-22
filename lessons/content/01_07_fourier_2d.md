---
id: "01_07"
title: The 2D discrete Fourier transform
section: 1
engines: [s1_images.fourier_2d]
summary: Images decomposed into 2D corrugations, how to read a centred spectrum, leakage and windowing in 2D, why phase matters more than magnitude, and filtering by multiplying spectra including notch filters for periodic noise.
---

## Learning outcomes

After this lesson you should be able to:

- describe the **2D DFT** and its **basis functions**;
- explain why spectra are **centred** with fftshift and how to read them;
- recognise **spectral leakage** in 2D and the effect of a window;
- explain the roles of **magnitude and phase** in reconstructing an image;
- perform **low-pass, high-pass and notch filtering** in the frequency domain and predict when **ringing** occurs.

## The 2D DFT

An image is a 2D grid of pixels, and the 2D DFT works like the 1D case but with two **spatial frequencies**, one horizontal and one vertical. The image is reconstructed from coefficients multiplied by complex exponentials containing both frequencies:

$$x[m, n] = \frac{1}{MN} \sum_{k=0}^{M-1} \sum_{l=0}^{N-1} X[k, l]\, e^{j 2\pi \left( \frac{k m}{M} + \frac{l n}{N} \right)}. \tag{1}$$

The magnitudes of the coefficients $X[k, l]$ form the **spectrum**, which is now a grid the same size as the image.

### Basis functions

In the Fourier series, discrete frequencies were combined to reconstruct a signal. In the 2D DFT the two frequencies $k$ and $l$ combine into a 2D **corrugation**, and the image is a weighted sum of these basis functions. Taking the real part of the complex exponential: with $l = 0$ the intensity varies sinusoidally along $x$ only, with $k = 0$ along $y$ only, and increasing either frequency changes both the spacing and the orientation of the stripes.

{{fig dft_basis_2d}}

### Centring the spectrum

The spectrum is computed for $k$ up to $M$ and $l$ up to $N$. As in 1D the lowest frequency is at zero, which for the 2D grid is the top-left corner. The 2D spectrum has the same symmetry as the 1D one: the top-left quadrant mirrors the bottom-right, and the top-right mirrors the bottom-left. Because images contain a great deal of low-frequency energy from their smooth regions, it is far easier to interpret the spectrum with the low frequencies **in the centre**, which is achieved by swapping the quadrants with NumPy's `fftshift`. A properly shifted image spectrum always has a bright spot in the middle.

{{fig spectrum_2d}}

## Spectral leakage

As in 1D, the DFT assumes the image period equals the image size, in both directions. A corrugation that does not fit a whole number of cycles into the image spreads its energy. A crosshatch pattern made of two diagonal corrugations produces horizontal and vertical streaks emanating from each frequency. Applying a 2D **Hann window** before the DFT reduces the leakage so that the two frequencies and the central DC term stand out; as before the window attenuates the signal, and reconstructing from the windowed spectrum would give an image that darkens towards its edges.

{{fig leakage_2d}}

## Magnitude and phase

So far we have looked at magnitudes, which form the spectrum. The complex coefficients also carry **phase**, and for reconstructing an image the phase is far more important. From a coefficient with real part $R$ and imaginary part $I$,

$$|X| = \sqrt{R^2 + I^2}, \qquad \angle X = \tan^{-1}(I / R).$$

The experiment of throwing away one or the other is dramatic. Keep the magnitudes and set every phase to zero, and the structure of the image is completely lost. Keep the phases and set every magnitude to one, and the structure is clearly visible.

{{fig magnitude_phase}}

## Common transforms

Consider three narrow features in the image domain: a small square, a small Gaussian blob and a small disc. The first observation is that a **small feature in the image domain produces a large response in the frequency domain**. The second is that the sharp-edged square and disc produce extensive **ringing**: the square along the $k$ and $l$ directions, the disc as concentric rings (a Bessel function, corresponding to waves of the same frequency at every orientation). The Gaussian is converted into another Gaussian, wider in the frequency domain, with no ringing at all.

{{fig common_dfts}}

## Filtering with the DFT

The convolution theorem carries over to images: the DFT of a convolution is the product of the DFTs,

$$\mathrm{DFT}(I_1 * I_2) = \mathrm{DFT}(I_1) \cdot \mathrm{DFT}(I_2). \tag{2}$$

The DFT is therefore an alternative to direct convolution, and a more efficient one for large images and kernels: convolution scales with the square of the image size, whereas the FFT scales as $N \log N$.

### Low-pass and high-pass with a disc

With the spectrum centred, the low frequencies are in the middle, so a **disc** is a basic low-pass filter: multiply the spectrum by the disc and take the inverse DFT. The image is blurred as expected, but **ringing** appears beside the edges. This is because, by the symmetry of the DFT and its inverse, a disc in the frequency domain becomes a Bessel function in the image domain, and a Bessel kernel rings. The **inverse** of the disc, a black circle on a white background, is a high-pass filter: the low frequencies are removed and only the edges remain.

### Filtering with a Gaussian

A Gaussian in the frequency domain gives a Gaussian in the image domain, so filtering with a Gaussian mask blurs **without ringing**. The widths are inversely related: a wide Gaussian in the frequency domain passes most frequencies and filters very lightly, while a narrow one removes most of the high frequencies and filters heavily.

{{fig freq_filter}}

### Removing periodic noise

A final use of the DFT is the removal of **periodic noise**, such as a pattern of parallel stripes from electrical interference. No convolution kernel removes such a pattern easily, but in the frequency domain the interference is concentrated at a single pair of peaks. A **notch filter**, which zeros the spectrum precisely at those peaks, followed by the inverse DFT greatly reduces the visibility of the stripes while leaving the rest of the image intact.

{{fig periodic_noise}}

{{selfcheck 01_07}}
