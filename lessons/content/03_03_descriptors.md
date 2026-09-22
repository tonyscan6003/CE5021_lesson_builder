---
id: "03_03"
title: Keypoint descriptors and matching
section: 3
engines: [s3_features.descriptors]
summary: "Why keypoints need descriptors, how gradient histograms turn a patch into a vector, how a characteristic orientation makes the descriptor rotation invariant, and how descriptors are matched with brute force and the ratio test."
---

## Learning outcomes

After this lesson you should be able to:

- explain why a detected keypoint must be **described** before it can be matched;
- build a simple **gradient-histogram descriptor** for a patch;
- explain the role of the **characteristic orientation** in rotation invariance;
- describe **brute-force matching**, its relation to k-nearest neighbours, and the **ratio test**.

## Matching keypoints

The previous lessons showed how scale-invariant detectors find the majority of the same keypoints in two similar images. The remaining question is how to **describe** a keypoint so that its counterpart in the other image can be recognised. A descriptor is a vector computed from the pixels around the keypoint; two keypoints match if their descriptors are close.

## Gradient-based descriptors

One good way to describe a keypoint is with the **image gradients** around it. This is the basis of both the SIFT descriptor (next lesson) and the histogram of oriented gradients used for object detection in Section 4. A finite-difference approximation gives the gradient of the image, or of the image in scale space, and the magnitude and direction of each pixel follow from the usual formulae.

### Histograms

Take a region of pixels around the keypoint, say an 8 × 8 grid, and compute the gradient magnitude and angle of every pixel. Divide the grid into **cells** of 4 × 4 pixels, and in each cell form a **histogram** of gradient directions: each pixel goes into the bin that contains its angle, and the pixel's magnitude is added to that bin. With eight bins of 45°, a pixel at 35° belongs to the first bin and one at 145° to the fourth. More elaborate schemes share a pixel's magnitude between neighbouring bins, so that a pixel at 91° contributes to the bins either side of 90°; this smooths the histogram and improves matching, since small changes in angle no longer produce very different histograms.

### The descriptor vector

Concatenating the cell histograms gives a single vector: four cells of eight bins give a vector of length 32. This vector is the keypoint descriptor. SIFT and HOG are both built on this principle, with different layouts and hence different vector lengths.

{{fig gradient_patch}}

## Rotation-invariant descriptors

Feature detection of blobs is scale and rotation invariant, so the same feature is detected in an image and in a rotated copy. But a gradient descriptor computed on the same patch in the two images would differ, because all the local gradient directions have rotated. To make descriptors invariant we compute a **characteristic orientation** for each keypoint from the gradients in its patch and express all directions relative to it. After this normalisation the two patches produce the same descriptor.

{{fig characteristic_orientation}}

## Feature matching

With descriptors we can match keypoints, which allows simple matching of a single object between two images. The keypoints of a reference image and a test image are computed, and a **brute-force** matcher compares every descriptor of the test image with every descriptor of the reference image. A match is declared where the distance between the vectors, usually the Euclidean (L2) distance, is smallest.

### Relation to k-nearest neighbours

Brute-force matching is the k-nearest-neighbours algorithm with k = 1, where every descriptor of the reference image is its own class. To improve the quality of the matches, the distances to the **first and second** nearest neighbours are compared: if the nearest neighbour is at less than about 70% of the distance to the second nearest, the match is unambiguous and is accepted. This is **Lowe's ratio test**.

{{fig ratio_test}}

{{fig matching_scaled}}

## Application: robot localisation

Feature matching has been proposed for localising the Mars rover, which has no GPS. The rover estimates its position from wheel rotations and the position of the sun, but errors accumulate over hours and days, and finding the rover in orbital imagery by eye is hard after a long traverse. Matching keypoints on rocks between the rover's own camera images and orbital images identifies the same pattern of rocks in both, and hence the rover's position.

{{selfcheck 03_03}}
