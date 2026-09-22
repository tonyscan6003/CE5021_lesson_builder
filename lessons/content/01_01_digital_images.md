---
id: "01_01"
title: Digital images
section: 1
engines: [s1_images.digital_images]
summary: How a camera turns light into an array of numbers, why colour needs three channels, and what "resolution" does and does not mean.
---

## Learning outcomes

After this lesson you should be able to:

- describe how a camera sensor **samples** and **quantises** incident light to form a digital image;
- explain how a single sensor captures colour through a **Bayer filter** and demosaicing;
- describe the **RGB colour model** and the size of the 8-bit RGB colour space;
- list the camera effects (lens, shutter, sensor noise, compression) that limit image quality.

## Human eye versus camera

Human vision and the camera are functionally similar: both use a lens to focus incident light onto a light-sensitive surface. In the eye that surface is the retina; in a digital camera it is the CCD or CMOS image sensor. The eye focuses by changing the *shape* of its lens, whereas a camera lens is rigid, so its *position* relative to the sensor is moved instead. The iris controls how much light enters the eye so that the retina is not over-exposed; the adjustable aperture behind a camera lens does the same job.

| Function | Eye | Camera |
|---|---|---|
| Sensor | Retina | Image sensor (or film) |
| Focus | Lens (changes shape) | Lens (changes position) |
| Brightness | Iris | Aperture |

### Human colour perception

Light focused on the retina is converted into electrical impulses that the brain interprets as vision. The **Young–Helmholtz trichromatic theory** (proposed in the 19th century and confirmed in the 20th) states that humans have three types of colour receptor, the **cone** cells, each sensitive to a different band of wavelengths: long (red), medium (green) and short (blue). The retina also contains **rod** cells, which are more light-sensitive than cones and provide monochrome vision at night. How the eye detects colour has direct consequences for how we design colour sensors and displays.

## Digital image acquisition

Modern image processing operates exclusively on digital images acquired by a camera. Light is focused by the lens onto the sensor; the aperture controls the amount of light and the shutter (mechanical or electronic) sets the instant at which the sensor is read.

The sensor is a rectangular grid of discrete light-sensing elements, the **pixels** (picture elements). While the shutter is open, light falling on a pixel releases electrical charge through the photoelectric effect. In a CMOS sensor this charge is converted to a voltage inside each pixel and the voltages are read out sequentially. CMOS sensors dominate phone cameras because the amplifier, analogue-to-digital converter and signal processing can all be integrated on one low-cost chip.

### Sampling and quantisation

Two operations turn the continuous scene into numbers:

1. **Sampling.** The scene is sampled *spatially* by the grid of pixels and *in time* by the shutter. The result is a spatially sampled scene: one measurement per pixel.
2. **Quantisation.** Each pixel's voltage is converted by an analogue-to-digital converter into an integer. In the simplest case a greyscale image stores one 8-bit value per pixel, in the range 0 to 255. A colour image stores three such values per pixel, one for each of red, green and blue.

{{fig sample_quantise}}

## Colour image capture

A sensor pixel responds to a broad range of wavelengths, so on its own a sensor produces only a greyscale image. To capture colour we mimic the eye's receptors and make different pixels sensitive to different wavelengths.

The highest-quality solution uses **three separate sensors**, each behind a colour filter, so that each sensor delivers one channel of the RGB image. It is also the most expensive. A typical camera uses a **single sensor** with a mosaic of tiny colour filters over the pixels, arranged in the **Bayer pattern**: alternating rows of red/green and green/blue filters. Each pixel therefore records only one colour. The camera's signal processor fills in the two missing colours at every pixel by interpolating from its neighbours, a process called **demosaicing**.

Notice that there are twice as many green pixels as red or blue. This deliberately mimics the eye, which draws most of its luminance (brightness) information from green wavelengths.

{{fig bayer_demosaic}}

## Digital image resolution

The camera delivers an RGB image: three channels, each holding an 8-bit intensity for every pixel, which a display with red, green and blue sub-pixels can reproduce directly.

{{fig rgb_channels}}

A key parameter of a digital image is its **number of recorded pixels**, usually quoted as $n \times m$ and often called the "resolution". Strictly, that is a misuse of the word. The pixel count is only an *upper bound* on resolution. The true resolution of an image, its ability to separate closely spaced dark and light lines, also depends on the optical quality of the lens. In general, though, a modern camera with more pixels does capture finer detail.

{{fig resolution_pixels}}

## The RGB colour model and colour space

A **colour model** is a way, usually mathematical, of describing the colours a human observer can perceive. In the **additive RGB model** three coloured lights, red, green and blue, are mixed in varying amounts to produce other colours.

{{fig additive_mixing}}

Closely related is a **colour space**: the set of colours a particular device can actually display. In the 8-bit RGB colour space each channel takes one of 256 values, so the space contains

$$256 \times 256 \times 256 = 16{,}777{,}216 \text{ colours.}$$

Some cameras and displays use 10 or 12 bits per channel to reach even more.

{{fig rgb_colour_space}}

## Colour image display

Modern displays are fundamentally RGB devices. Each display pixel contains red, green and blue light-emitting sub-pixels which, viewed from a distance, blend according to the additive model. Image and video processing frequently uses *other* colour spaces that separate luminance from colour so the two can be processed independently, but because the display is RGB, every image must be converted back to RGB before it is shown.

{{fig display_rgb}}

## Optical performance: cameras are not perfect

Camera design (lenses, sensors and in-camera processing) is a large topic in its own right and not the focus of this module. From an image-processing point of view, though, we must remember that every stage can degrade the image:

- **Lens.** The pixel count sets an upper bound on resolution; lens quality decides whether it is reached.
- **Shutter.** Electronic rolling shutters distort fast-moving objects, the classic example being bent helicopter blades.
- **Sensor.** Raising the sensor gain to shoot in low light amplifies noise along with the signal.
- **Compression.** JPEG compression introduces blocking artefacts, especially at high compression ratios.

{{fig camera_effects}}

When a machine vision algorithm is deployed in the real world its performance may be lower than in the lab because of any of these effects. Keep them in mind whenever an algorithm fails on real camera data.

{{selfcheck 01_01}}
