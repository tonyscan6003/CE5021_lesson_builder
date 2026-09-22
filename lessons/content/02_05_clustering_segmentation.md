---
id: "02_05"
title: Clustering-based segmentation
section: 2
engines: [s2_edges.clustering_segmentation]
practice: [kmeans_iter]
summary: "Segmenting by grouping pixels in feature space: k-means and its shortcomings, density estimation with a Parzen window, and the mean-shift procedure that climbs to the modes of the density."
---

## Learning outcomes

After this lesson you should be able to:

- perform an iteration of **k-means** and explain why it segments images poorly;
- describe **Parzen window** density estimation and the role of the bandwidth;
- explain the **mean-shift** procedure and why the mean-shift vector points up the density gradient;
- list the advantages and disadvantages of mean shift, and explain how spatial information is included.

## Simple clustering

Images can be segmented by grouping similar pixels into **clusters**, for example by colour. The obvious algorithm is **k-means**, which operates in **feature space**: it clusters the RGB values of the pixels, and the spatial location of each pixel is not used. The number of clusters $k$ must be chosen in advance. Each iteration assigns every point to its nearest centroid and then moves each centroid to the mean of its points.

{{fig kmeans_steps}}

{{example kmeans_iter seed=5}}

k-means is not usually used for segmentation. Because it ignores position, separate objects of similar colour are assigned to the same cluster and the segmentation quality is poor.

{{fig kmeans_colour}}

## Mean-shift clustering

The rest of this lesson covers **mean shift**, which can produce high-quality segmentations.

### Intuition

When features extracted from an image are plotted in feature space, clusters are the regions where the density of points is highest. The feature points can be viewed as discrete samples of an unknown **probability density function**. Mean shift is an iterative procedure for locating the **maxima** of that density. A kernel window is placed in feature space and the weighted mean of the samples inside it is computed; this mean reflects the local density. The **mean-shift vector**, from the window centre to the weighted mean, points in the direction of steepest increase in density. The window centre is moved to the mean and the process repeats until it converges at a local maximum, the peak of the density.

### Density estimation

The simplest density estimate is a histogram, which quantises the space and counts the points in each cell: a crude estimate. A **Parzen window** estimates the density at a point $x$ as the sum of a kernel function applied to the distances from $x$ to the $n$ samples,

$$\hat f(x) = \frac{1}{n h^d} \sum_{i=1}^{n} K\!\left(\frac{x - x_i}{h}\right), \tag{1}$$

where $h$ is the **bandwidth** of the window. Summing a kernel placed on every sample is mathematically a convolution.

{{fig parzen_density}}

### The gradient of the density

Evaluating the Parzen window everywhere and searching for maxima would be very inefficient. Mean shift instead uses the **derivative** of the Parzen estimate to move uphill. Writing the kernel in terms of an unnormalised *profile* $k$ (for convenience) and its derivative $g = -k'$, the gradient of the density estimate expands to

$$\nabla \hat f(x) \propto \left[ \sum_i g\!\left(\left\|\frac{x - x_i}{h}\right\|^2\right) \right] \left[ \frac{\sum_i x_i\, g\!\left(\left\|\frac{x - x_i}{h}\right\|^2\right)}{\sum_i g\!\left(\left\|\frac{x - x_i}{h}\right\|^2\right)} - x \right]. \tag{2}$$

The first bracket is a density estimate with the profile $g$; the second is the **mean-shift vector**, the difference between the weighted mean of the samples and the current position. The vector always points towards the maximum increase in density.

### The mean-shift procedure

For the current window position, compute the mean-shift vector and translate the window by it; repeat until convergence at a local maximum. For clustering and segmentation we need **all** the maxima, so the procedure is started from many random points. Each maximum has a **basin of attraction**: every start inside it converges to that maximum, and the points a window passes on the way are labelled with that cluster.

{{fig mean_shift_path}}

### Advantages and disadvantages

Mean shift is an application-independent tool that clusters any data, not just image features. It assumes no particular cluster shape, handles multi-dimensional feature spaces easily, is robust to outliers (unlike k-means), and does not require the number of clusters in advance. Its disadvantages are that it is computationally slow, and that the bandwidth $h$ is not trivial to choose since it is hard to know what value is appropriate in a given feature space.

### Segmentation with mean shift

For segmentation the features are usually taken in the **Luv** colour space, with a luminance channel L and two colour channels u and v, rather than RGB. Mean shift is available in scikit-learn; the one parameter to set is the bandwidth, chosen by trial and error. Because the algorithm handles larger feature spaces, the **spatial position** can be added as further features: pixels that are physically close are likely to belong together. As the features come from different domains, the kernel becomes the product of a colour kernel and a spatial kernel. Including position further improves the segmentation.

{{fig mean_shift_segment}}

All the segmentation methods in this and the previous lesson can segment images to varying degrees of quality, but they share one drawback: the segmentation is not **semantic**. The algorithm does not know what its regions are.

{{selfcheck 02_05}}
