---
id: "04_01"
title: Introduction to object detection
section: 4
engines: [s4_detection.intro_detection]
practice: [iou, objdet_pre]
summary: "Image classification and object detection, why real-world recognition is hard, classification and detection metrics, and the sliding-window detector with an image pyramid for scale."
---

## Learning outcomes

After this lesson you should be able to:

- distinguish **image classification** from **object detection**;
- explain the role of **feature engineering** in recognition with machine learning;
- compute **accuracy, precision and recall** from a confusion matrix;
- list the **challenges** of real-world recognition;
- describe **sliding-window** detection, the use of an **image pyramid** for scale, and the **IoU**-based precision and recall used for detectors.

## Image classification

In the image classification task an image is labelled according to what it contains: "dog", "cat" or "ship". The simplest classifier outputs binary labels and distinguishes two classes; more complex systems distinguish many. The examples are trivial for a person, yet image classification is a very hard problem for a computer. Until neural networks and deep learning, computer classification lagged far behind human ability, because of the enormous variability of real-world images.

## Object detection

Object detection is closely related: it **localises** the objects of a given type within an image, usually with a rectangular **bounding box** around each. Detectors can handle several object types at once, and because they provide positions they also enable **tracking**. Detection is more computationally intensive than classification because of the localisation step.

{{fig classification_vs_detection}}

## Feature engineering

Both tasks are usually solved with machine learning, the subject of the next lesson. Images are first converted into an abstract **representation** that makes the classes easier to discriminate; the representations in this module are mostly built from extracted image features, and designing one for a given application is called **feature engineering**. The classifier is trained on a dataset of images in this representation, and every test image must be converted to the same representation before it is classified.

{{fig feature_engineering}}

## Classification metrics

The commonest metrics are accuracy, precision and recall, defined from the **confusion matrix**. For a binary pedestrian detector there are four outcomes: a **true positive** (pedestrian predicted, pedestrian present), a **false positive** (pedestrian predicted, none present), a **false negative** (no pedestrian predicted, one present) and a **true negative**.

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad \text{precision} = \frac{TP}{TP + FP}, \qquad \text{recall} = \frac{TP}{TP + FN}. \tag{1}$$

High precision means few false positives and confident positive predictions; high recall means few missed detections.

{{fig confusion_binary}}

## Image recognition challenges

Recognition is difficult because of the variability of real images:

- **viewpoint**: the same object looks different from different angles;
- **illumination**: lighting changes the appearance of the same object;
- **scale**: objects appear at different sizes depending on distance;
- **occlusion**: objects are partly hidden and must be recognised from their visible parts;
- **background clutter**;
- **deformation**: many objects are not rigid;
- **intra-class variation**: common objects come in many shapes and sizes.

Knowing these helps to understand why a system fails under certain conditions and how to address the failures in its design or use.

{{fig recognition_challenges}}

## Sliding-window object detection

The simplest detector is a **binary classifier** that recognises a specific object and labels everything else as background, trained on small image patches of a fixed size such as 28 × 28 or 64 × 64 pixels. The patch size sets the **detector window**. To find objects in a test image the window is slid across it, typically stepping by half the window size, and the classifier is applied at every position; equivalently the image is cut into patches the size of the training images and each is classified. Two problems follow. The approach is **computationally inefficient**, since a large image needs hundreds or thousands of classifications, which is acceptable for simple classifiers but not for complex ones. And the **fixed window size** means only objects of one scale are detected.

{{fig sliding_window}}

### An image pyramid for scale

A small, distant object fits the window and is detected; a large, nearby one never does. The solution is a **Gaussian image pyramid** of the test image: the image is repeatedly blurred (to prevent aliasing) and subsampled, and sliding-window detection is applied at every level. Small objects are found in the large images and large objects in the small ones. More pyramid levels give a better chance of finding every object at the price of more classifications.

{{fig pyramid_for_scale}}

## Object detection metrics

### Intersection over union

How well a predicted bounding box fits the ground truth is measured by the **intersection over union**:

$$\text{IoU} = \frac{\text{area}(B_p \cap B_{gt})}{\text{area}(B_p \cup B_{gt})}. \tag{2}$$

Little overlap gives a small intersection and a large union, so a low IoU; a good fit gives a high IoU. A threshold on the IoU, for instance 0.5, defines a true positive. False positives are spurious detections or boxes that barely overlap an object; false negatives are objects with no detection.

{{fig iou_boxes}}

{{example iou seed=2}}

### Precision and recall for detectors

Unlike classification, object detection does not use true negatives: the entire background would count as one, and the huge number of true negatives would distort any accuracy figure. Detectors are therefore assessed with **precision and recall only**. High precision with low recall means few false alarms but many missed objects; low precision with high recall means few misses but many false alarms.

{{fig detection_pr}}

{{example objdet_pre seed=4}}

## Two sliding-window detectors

The rest of this section studies two important sliding-window detectors. The **Viola-Jones** algorithm, developed for face detection, uses a cascade of weak classifiers on simple image features and is efficient enough to run on embedded devices. The **histogram of oriented gradients** (HOG), developed for pedestrian detection, uses a SIFT-like gradient descriptor as the feature vector and a support vector machine as the classifier, finding the patterns of edges that make up a pedestrian.

{{selfcheck 04_01}}
