---
id: "02_04"
title: Region-based segmentation
section: 2
engines: [s2_edges.region_segmentation]
practice: [otsu_thresh]
summary: "Partitioning an image into regions by thresholding, choosing the threshold automatically with Otsu's method, and the split-and-merge family: the watershed and graph-based segmentation."
---

## Learning outcomes

After this lesson you should be able to:

- define **image segmentation** and give examples of its uses;
- segment a foreground object by **thresholding** a histogram and compute **Otsu's threshold**;
- explain the limitations of thresholding;
- describe the **watershed** algorithm, why it over-segments and how **markers** fix this;
- outline **graph-based split and merge** segmentation.

## Image segmentation

Segmentation partitions an image into parts or regions. It can be as simple as separating a foreground object from the background, or it can group regions of similar pixels, as in a medical image. Regions often have a semantic meaning: they define an object or part of one. There are many segmentation techniques (Szeliski's *Computer Vision* gives a comprehensive overview). This lesson covers thresholding and split-and-merge methods; the next covers clustering.

## Thresholding

The simplest segmentation is a threshold, typically used to separate a foreground object from the background. A **histogram** counts the pixels at each intensity; if foreground and background have their pixels concentrated at different intensities, a threshold between the two groups separates them. All pixels above the threshold become white and the rest black. The method can be applied to a greyscale image or to any channel of an RGB image.

{{fig threshold_segment}}

### Otsu's method

Instead of choosing the threshold by inspecting the histogram, **Otsu's method** finds the best threshold automatically by trying every possible one. For a threshold $t$, let $w_0$ and $w_1$ be the fractions of pixels in the two classes and $\mu_0$, $\mu_1$ their mean intensities. Otsu selects the $t$ that maximises the **between-class variance**

$$\sigma_B^2(t) = w_0(t)\, w_1(t)\, \big(\mu_0(t) - \mu_1(t)\big)^2, \tag{1}$$

which is the same as minimising the spread of intensities within the foreground and within the background.

{{fig otsu}}

{{example otsu_thresh seed=3}}

### Problems with thresholding

Thresholding is generally unsatisfactory unless combined with other techniques. Most images do not have a histogram with an obvious threshold in any channel. Otsu's method also assumes comparable numbers of foreground and background pixels, so it tends to fail for **small foreground objects**.

{{fig threshold_problems}}

## Segmenting multiple objects

Thresholding a set of touching objects of similar colour returns them as one connected blob. To split the blob into one region per object we use the **watershed** algorithm, an example of **split and merge** segmentation: divide the image into many initial regions, then merge them into larger regions that correspond to objects or object parts.

### The image as a surface

The watershed interprets a greyscale image as a **surface**: bright regions are hills or maxima and dark regions are valleys or minima.

{{fig image_surface}}

### The watershed algorithm

Imagine the water table rising from every local minimum on the surface. As the water rises, the pools flood the valleys; where water from two different sources is about to meet, a **dam** is built. The dams are the segment boundaries, and they generally coincide with edges in the image.

The basic watershed **over-segments**: every small unimportant minimum becomes a source and produces its own region. The fix is to use **markers** that select which regions will be flooding sources, which greatly reduces the number of segments. Markers can be placed by hand, or object centres can be found automatically, for example from the peaks of the distance transform of a thresholded mask, combined with simple morphological operations.

{{fig watershed_flood}}

## Graph-based split and merge

A more sophisticated split-and-merge method, introduced by Felzenszwalb and Huttenlocher in 2004, represents the image as a **graph**. Each vertex is a pixel, and each edge between neighbouring pixels carries a weight equal to their intensity difference. The goal is to assign every vertex to a region.

Initially every pixel is its own region. Regions are then merged according to a **dissimilarity** measure that compares two quantities: the **difference between regions**, the minimum-weight edge joining them, and the **internal difference** of a region, the largest weight in its minimum spanning tree. If the difference between two regions is large compared with at least one of their internal differences, the regions are dissimilar and are not merged.

The scheme: sort the $m$ edges by increasing weight, so that edges linking similar pixels are considered first; for each edge in turn, if its two vertices lie in different regions and the regions are not dissimilar, merge them. The result tends to over-segment a scene but is a very useful basis for merging regions into objects.

{{fig graph_merge}}

{{selfcheck 02_04}}
