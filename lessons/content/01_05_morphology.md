---
id: "01_05"
title: Binary images and morphology
section: 1
engines: [s1_images.morphology]
practice: [morph_opr1, morph_opr2, morph_opr3, morph_opr4]
summary: Binary images and structuring elements, the fit/hit/miss rules behind erosion and dilation, opening and closing, the distance transform, connected-component labelling, and the greyscale versions.
---

## Learning outcomes

After this lesson you should be able to:

- create a **binary image** by thresholding and explain what a **structuring element** is;
- apply the **fit / hit / miss** rules to compute **erosion** and **dilation** by hand;
- explain what **opening** and **closing** do and why the two operations are paired;
- compute a **distance transform** with the L1 and L2 metrics;
- label the **connected components** of a binary image;
- describe **greyscale morphology** in terms of minimum and maximum.

## Binary images

Besides RGB and greyscale images, **binary images** are common in image processing. A binary image has been quantised to two values. Textbooks write them as 0 and 1; OpenCV stores black as 0 and white as 255. Binary images are usually made from a greyscale image by a **threshold** operation: pixels above the threshold become 1 and the rest become 0. They are useful as **masks** that separate foreground objects from the background, a topic we return to in the segmentation lessons.

{{fig binary_threshold}}

## Morphology

Morphology is a family of image operations that process an image according to the **shape** of its regions. They are simple **non-linear** operations: the output pixel is not proportional to the input pixels, as will be obvious once we see the rules. Like a filter, a morphological operation is applied at every pixel of the image. The two primary operations are:

- **Erosion**, which removes pixels from the boundary of a region: a letter is thinned, a square shrinks and a hole inside it widens.
- **Dilation**, the opposite, which adds pixels to the boundary: the letter thickens, the square grows and its hole shrinks.

### The structuring element

The operation is applied at each pixel over a neighbourhood defined by a small binary array called the **structuring element** (SE). The element may be a cross, a square or an approximate disc, and it comes in different sizes: the larger the element, the larger its effect on the boundary.

{{fig structuring_elements}}

At any position, the degree of overlap between the 1s of the structuring element and the 1s of the image decides the outcome:

- a **fit**: every 1 of the element lies over a 1 of the image;
- a **hit**: some, but not all, of the element's 1s lie over image 1s;
- a **miss**: none of them do.

### Erosion and dilation

The **erosion** rule: output 1 if the structuring element **fits** the image at that position, otherwise 0. Since only a fit produces a 1, erosion reduces the number of white pixels.

The **dilation** rule: output 1 if the element **fits or hits**, and 0 only for a miss. Fits and hits both produce 1s, so dilation extends object boundaries and is useful for filling holes and gaps.

{{fig fit_hit_miss}}

{{fig erosion_dilation}}

{{example morph_opr1 seed=3}}

{{example morph_opr2 seed=3}}

## Opening and closing

Combining erosion and dilation gives two further operations. **Opening** is an erosion followed by a dilation with the same structuring element. It removes small regions of pixels or noise, and separates large objects from small ones. The reason for the second step is to restore the size of the objects of interest: erosion alone removes the small cells in a microscope image but also shrinks the large cells; the following dilation brings the large cells back to their correct size, giving a clean mask.

**Closing** is the reverse: a dilation followed by an erosion. Its purpose is to remove unwanted holes or gaps. The dilation extends every boundary outward, closing gaps and small holes; the erosion then shrinks the objects back to their original size. Holes larger than the structuring element survive, so a larger element closes larger holes, while the outer size of the objects is preserved either way.

{{fig opening_closing}}

{{example morph_opr3 seed=8}}

{{example morph_opr4 seed=8}}

## The distance transform

The **distance transform** of a binary image gives, for each foreground pixel $(i, j)$, its distance to the nearest background pixel $(k, l)$:

$$D(i, j) = \min_{(k, l):\, I(k, l) = 0} d\big((i, j), (k, l)\big). \tag{1}$$

The result is a greyscale image whose brightness grows with distance from the background, so the highest values sit at the centres of objects. This makes the transform useful for finding object centres, including centres of overlapping objects, and for building skeletons. Any distance metric can be used. The **L1** or Manhattan distance adds the coordinate differences,

$$d_1 = |i - k| + |j - l|, \tag{2}$$

and the **L2** or Euclidean distance is

$$d_2 = \sqrt{(i - k)^2 + (j - l)^2}. \tag{3}$$

Evaluating the minimum over all background pixels directly is expensive; practical implementations use two-pass algorithms.

{{fig distance_transform}}

## Connected components

Regions of 1-valued pixels that are not separated by a boundary are **connected components**. Labelling each component lets us extract it as a mask for one object in the original image. A simple two-pass algorithm scans the image from top-left to bottom-right, row by row. Using either the 4-neighbours or the 8-neighbours of each pixel, only the already-visited neighbours above and to the left are consulted. When a 1-pixel $p$ is met:

1. if all valid neighbours are 0, assign a **new label**;
2. if exactly one labelled neighbour exists, assign **its label**;
3. if neighbours with **different labels** exist, choose one and **record the equivalence**.

After the scan, equivalent labels are merged into a single class in a second pass. Using 8 rather than 4 neighbours changes the result, because diagonal neighbours then count as connected.

{{fig connected_components}}

## Greyscale morphology

The binary operations extend to greyscale images by replacing the fit/hit rules with **minimum** and **maximum** over the structuring element. Greyscale **erosion** outputs the minimum value under the element, reducing the brightness of pixels surrounded by darker neighbours; greyscale **dilation** outputs the maximum, brightening pixels surrounded by brighter neighbours. Opening (erosion then dilation) removes small bright spots in dark regions and closing (dilation then erosion) removes small dark spots in bright regions; both leave the image closer to the original than either primitive alone.

{{fig grey_morphology}}

{{selfcheck 01_05}}
