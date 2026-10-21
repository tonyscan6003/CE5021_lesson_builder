---
id: "03_02"
title: Region proposals
section: 3
engines: [s3_detection.region_proposals]
practice: []
summary: Region proposals as an alternative to the sliding window — the objectness measure and its cues (saliency, colour contrast, edge density, superpixel straddling), selective search by hierarchical grouping of segmented regions, region similarity, proposal recall, and training a classifier with hard negatives.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain why an object detector needs a **region proposal** stage and list the characteristics of a good proposal algorithm;
- describe the **objectness** measure and its cues: multi-scale saliency, colour contrast, edge density, superpixel straddling, and location and size;
- explain the failure cases of the individual objectness cues;
- describe **selective search** and how it produces proposals at all scales;
- describe the **similarity measures** used to merge regions;
- explain how proposal methods are compared by **recall** against the number of proposals;
- explain how **hard negative** examples are used to train a classifier on region proposals.

## Introduction

In this lesson the topic of object detection is continued with region proposals, methods of determining the location of objects in an image that are an alternative to sliding windows and image pyramids.

## Object detection and region proposals

In the object detection task the location of the objects of interest is normally unknown. In the HOG and Viola–Jones detectors, a sliding window and an image pyramid are used to search the image exhaustively for objects at different positions and scales. In the HOG pedestrian detector the classifier is applied at every window position, which is inefficient when the window contains no object of interest; in the Viola–Jones face detector a cascade of classifiers is used to reject background regions quickly and avoid wasted computation. In general it is preferable to have a stage that identifies the location of candidate objects, so that only these candidate regions are sent to the classifier. In the methods described in this lesson, regions of different sizes that contain an object are found directly. These are called **region proposals**.

{{fig proposal_quality}}

A good region proposal algorithm has the following characteristics:

- far fewer regions are sent to the classifier than with a sliding window and image pyramid;
- objects of any type, size or scale can be found;
- few **false positive** proposals are generated, which reduces the number of calls to the classifier and helps the precision of the detector;
- few **false negative** proposals are generated, as a missed object cannot be recovered by the classifier, which reduces the recall of the detector.

## Objectness

The first region proposal algorithm described is **objectness**, a measure of how likely it is that an image window covers an object of any class. Five image cues are combined to determine whether a window is likely to contain an object; no single cue works for all images, so the cues are combined to improve the robustness of the algorithm.

### Multi-scale saliency

Salient pixels are those that stand out from their surroundings. A saliency map can be generated at different scales using the Fourier transform. A large object appears as a blob in the coarse-scale saliency map, while small objects appear in the fine-scale map, so that objects of both sizes are found.

{{fig objectness_saliency}}

### Colour contrast, edge density and superpixel straddling

There is often a high degree of **colour contrast** between an object and its surroundings, such as white sheep in a green field. The contrast is detected by comparing the histograms, in LAB colour space, of the pixels inside the candidate window and of a ring of pixels just outside it. If the histograms are sufficiently different, an object is likely to be inside the window. An object with a colour similar to its surroundings is difficult to detect with this cue.

The **edge density** cue is based on the assumption that an object has many edges near its boundary, compared with a uniform region. The image is passed through the Canny edge detector, and the density of edges is measured in a ring just inside the boundary of the window. Regions such as water and trees also produce many edges, which results in false positive proposals.

**Superpixels** are a segmentation in which pixels are grouped by similarity of colour and texture. A key property of superpixels is that they do not straddle the boundary of an object. For each window, the amount of each superpixel inside and outside the window is compared; if a large part of a superpixel lies outside the window, the window is likely to cut across an object rather than contain it. **Superpixel straddling** is the best performing objectness cue, but the boundaries of small objects may not be preserved by the superpixels, and false negatives can occur.

{{fig objectness_cues}}

The final cue is the **location and size** of the window itself. Based on the statistics of real images, some windows are more likely to contain an object than others; for example, square windows in the centre of an image are more likely than small windows in the corners.

### Region proposals with objectness

The candidate windows are generated at random, and the windows most likely to contain an object are selected by their score on the combined cues. It is complex to train a proposal generator with these cues. Performance is improved when more random candidate windows are generated, at a higher computational cost, but an exhaustive search would be slower still and could contain more than 100,000 windows for a large image.

{{fig objectness_combined}}

## Selective search

The second region proposal algorithm is **selective search**, which is based on image segmentation. If an image can be segmented, the region proposals can be determined directly from the segmented image, and random windows no longer need to be generated and evaluated.

Region proposals are created at different scales, starting with small regions and working up to larger ones. The image is first **over-segmented**, so that only the very smallest objects consist of a single colour region. At this scale there are many small candidate regions, many of which do not contain an object. Similar neighbouring regions are then **combined iteratively** to create larger regions, so that complete objects are formed as the smaller regions are joined. The bounding box of every region, at every step, is a region proposal, which may contain an object or part of an object.

{{fig selective_search}}

## Region similarity

The similarity of two regions is based on several measures:

- **colour similarity**: histograms of the colours of the two regions in the original image are compared;
- **texture similarity**: gradients are compared with a HOG-like descriptor;
- **size similarity**: regions of similar, small size are merged first;
- **shape compatibility**: regions whose shapes fit together may form part of a larger object.

{{fig region_similarity}}

## Comparing proposal methods

Proposal methods are compared by the **recall** they achieve, the fraction of the objects in a test set that are covered by at least one proposal with sufficient overlap, as a function of the number of proposals per image.

{{fig proposal_recall}}

## Object recognition with selective search

The training of an object recognition system with selective search is more complex than the training procedures described previously. In the original work a support vector machine (SVM) is used as the classifier. A set of labelled training images with bounding boxes around the objects is supplied, and the ground-truth boxes are used to generate positive examples. The region proposals from selective search that overlap the ground truth by only 20–50% are retained and used as **difficult negative examples**. The SVM is trained with these positive and negative examples. After training, a search for false positives is performed, and these are added to the training data and the classifier is retrained to improve its performance.

{{fig svm_training}}

{{selfcheck 03_02}}
