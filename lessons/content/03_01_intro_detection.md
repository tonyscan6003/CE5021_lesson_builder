---
id: "03_01"
title: Introduction to object detection
section: 3
engines: [s3_detection.intro_detection]
practice: [iou, objdet_pre]
summary: The object detection task, the sliding-window and region-proposal approaches, detection datasets, intersection over union, precision and recall for detectors, average precision and mAP, the sources of detection error, and data augmentation with bounding boxes.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain why **object detection** is more difficult than image classification;
- explain why a **sliding window** with an image classifier is impractical, and how **region proposals** reduce the computation;
- describe the main **detection datasets**;
- calculate the **intersection over union** of two bounding boxes;
- classify predictions as **true positives, false positives and false negatives** using confidence and IoU thresholds;
- construct a **precision–recall curve**, interpolate it and calculate the **average precision** and **mean average precision**;
- describe the main **sources of error** in object detectors and how they are diagnosed;
- explain how **data augmentation** is applied when bounding boxes are present.

## Introduction

In this lesson the application of deep learning to the object detection task is introduced, and the datasets and metrics used to train and compare object detectors are described.

## Object detection

The object detection task is more challenging than image classification. In image classification one label is assigned to the entire image, whereas in object detection multiple objects of different classes must be localised within the image. For each object, a class is predicted together with a bounding box around the spatial extent of the object.

{{fig detection_task}}

## Naive object detection

A pre-trained image classifier provides a good starting point for an object detection system. In the naive approach, the classifier is used as a detector with a **sliding window**. A window of fixed size is placed over the image, the region under the window is passed to the classifier, and if a valid class is identified, an object of that class is assumed to be present within the window. The window is moved exhaustively across the entire image, in steps of a few pixels, so that objects can be found at any position.

The window has a fixed size, as the classifier requires a fixed input size, so an object is only detected if it is approximately the same size as the window. Objects that are larger than the window cannot be detected directly. To detect objects of different sizes, an **image pyramid** is used: the image is reduced in size repeatedly, and the sliding window is applied at every level of the pyramid. A large object becomes small enough to fit the window at one of the reduced levels.

{{fig window_pyramid}}

This approach is how classical object detectors, such as the HOG pedestrian detector and the Viola–Jones face detector, operate, and it was acceptable for them as their classifiers are computationally lightweight. For a large neural network, however, it is far too computationally expensive and slow. A high-resolution image can contain thousands of window positions, the image pyramid multiplies them again, and windows of several aspect ratios may also be needed to fit objects of different shapes.

{{fig sliding_window}}

## Region proposals

The number of candidate regions can be reduced with a **region proposal** algorithm, in which the regions of an image most likely to contain an object are identified. Image segmentation is fundamental to many of these algorithms, as the region occupied by an object can be found by segmentation and passed to the image classifier. Region proposal algorithms are described in the next lesson.

{{fig proposal_pipeline}}

A deep learning based object detector can be constructed by combining an image classifier with a region proposal algorithm, as in R-CNN, one of the first deep learning object detectors. Compared with the sliding window, fewer regions must be classified, and regions of any shape or size can be proposed without a fixed window size or an image pyramid. A good region proposal algorithm also helps to reduce the number of false positives, and the number of missed objects can be reduced if regions are proposed around all of the objects in the image.

There are, however, several problems with this approach. The region proposal algorithm is trained separately from the image classifier, so end-to-end training is not achieved. The proposed regions have different shapes and sizes, while most neural networks require a fixed input size. Finally, overlapping regions are processed several times by the classifier, which is inefficient. These problems and their solutions are discussed in the lessons on object detection architectures.

## Datasets

As with image classification, high-quality datasets have been a key driver in the development of object detection systems. The development of a detection dataset is time consuming, as a ground-truth bounding box must be annotated around every identifiable object.

{{fig detection_datasets}}

Pascal VOC was one of the first dedicated object detection datasets, and competitions were run on it from 2005 to 2012. ImageNet can also be used for object detection: an object localisation task was run from 2011 to 2014, and from 2014 a detection task with 200 classes was included. As ImageNet was assembled for image classification, many of its images contain a single centred object. The Microsoft Common Objects in Context (MS COCO) dataset addresses this problem with real-world scenes in which multiple objects appear at different locations and scales, and it is the benchmark dataset for object detection. MS COCO and the Open Images dataset also contain instance segmentation annotations, which are described in a later lesson.

## Intersection over union

A key function of an object detector is to estimate a bounding box around each object. The quality of a predicted box is measured by the **intersection over union** (IoU):

$$IoU = \frac{area(b \cap b_{g})}{area(b \cup b_{g})}, \tag{1}$$

where $b$ is the predicted bounding box and $b_g$ is the ground-truth bounding box.

{{fig iou_boxes}}

When the predicted box overlaps the ground truth exactly, the IoU is equal to 1. It is lower for a partial overlap, and reaches zero when the boxes do not overlap.

{{example iou seed=3}}

## True positives, false positives and false negatives

A deep neural network based object detector usually outputs a large number of predictions, some of which are not valid. A **confidence** value is output with each predicted box and label, which represents the probability that the box contains a valid object. The condition for a true positive, false positive or false negative is determined by a threshold $T_1$ on the confidence and a threshold $T_2$ on the IoU:

- predictions with a confidence below $T_1$ are ignored;
- a prediction with a confidence above $T_1$ and an IoU greater than $T_2$ is a **true positive**;
- a prediction with a confidence above $T_1$ and an IoU less than $T_2$ is a **false positive**;
- an object that is not detected by any prediction is a **false negative**.

{{fig tp_fp_fn}}

The choice of threshold values affects the balance between the precision and the recall of the detector.

## Precision and recall

In object detection the concept of a true negative is not used. Most of the image is background, which could be interpreted as true negative examples, so the number of true negatives would be far greater than the number of true positives, giving a very imbalanced set of data. Only precision and recall, which do not depend on true negatives, are therefore used:

$$Precision = \frac{TP}{TP + FP}, \qquad Recall = \frac{TP}{TP + FN}. \tag{2}$$

The precision is the proportion of positive predictions that are correct, and the recall is the proportion of the objects that are correctly identified. With a high confidence threshold only the best predictions are retained, so few false positives are produced but many objects are missed: the precision is high and the recall low. With a low confidence threshold most predictions are retained, so the false positives increase and the false negatives decrease: the precision falls and the recall increases.

{{example objdet_pre seed=4}}

## The precision–recall curve and average precision

When precision is plotted against recall as the confidence threshold is varied, for a fixed IoU threshold, a **precision–recall curve** is obtained. In real detectors the curve has a zig-zag shape: as the confidence threshold is reduced, the recall increases monotonically while the precision moves up and down. For an ideal detector the precision would remain high as the recall increases, so the area under the curve is used as a measure of performance. Before the area is calculated, the curve is interpolated. The interpolated precision at recall $r$ is the maximum precision at any recall greater than or equal to $r$:

$$p_{interp}(r) = \max_{r' \geq r} p(r'). \tag{3}$$

{{fig pr_curve}}

The area under the interpolated curve is known as the **average precision** (AP). It is calculated as a sum of rectangles, each defined by an interval between two recall values and the interpolated precision:

$$AP = \sum_{i=1}^{n-1} (r_{i+1} - r_i)\, p_{interp}(r_{i+1}). \tag{4}$$

The average precision is defined for a single class. With $C$ object classes, the **mean average precision** is the mean of the AP values of the individual classes:

$$mAP = \frac{1}{C}\sum_{i=1}^{C} AP_i. \tag{5}$$

{{fig mean_average_precision}}

The AP and mAP depend on the IoU threshold $T_2$ at which the precision–recall curves were obtained, and this threshold must be recorded when detectors are compared. The AP also varies across object categories; in published results, objects such as chairs and plants are detected less well than motorcycles or aeroplanes.

## Object detection errors

The types of error that reduce the average precision were investigated by Hoiem et al. False positives are composed of **localisation** errors, in which the bounding box around a correctly classified object is inaccurate; **similarity** errors, in which an object is misclassified as a similar object from a different class, as easily occurs with animals; and **background** errors, in which an object is falsely identified in a pattern in the background. False negatives typically occur for **small** objects, and for objects that are **occluded** or **truncated** by the border of the image.

{{fig detection_errors}}

It is difficult to determine the primary sources of error by inspection of the output images. An analysis function is provided with the COCO API, which produces a series of precision–recall curves with progressively relaxed conditions. Each time a source of error is removed, the curve moves to the right and the area under it increases; the size of each increase indicates how much the average precision would be improved by the removal of that source of error.

{{fig coco_error_analysis}}

## Data augmentation for object detection

Data augmentation may be used to improve the precision of a detector, or to prevent overfitting when data is limited. Photometric operations such as equalisation and changes of contrast or brightness can be applied, as can geometric operations such as rotation, shearing, translation and flipping. Many geometric transformations also require the bounding-box coordinates to be modified, which is an additional complication in object detection. The operations can be applied to the entire image or only to the region inside the bounding box, and combinations of operations can be applied; an optimal policy for a given application is described by Zoph et al.

{{fig detection_augmentation}}

In PyTorch, the `torchvision.transforms.v2` transforms accept bounding boxes together with the image and update the box coordinates automatically. Many object detection systems use simple policies, such as image scaling, translation and random cropping, as these are simpler to perform than operations such as rotation and shearing.

{{selfcheck 03_01}}
