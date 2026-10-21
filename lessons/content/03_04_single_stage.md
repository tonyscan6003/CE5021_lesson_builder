---
id: "03_04"
title: Single-stage object detection
section: 3
engines: [s3_detection.single_stage]
practice: [depthwise_sep_conv, iou]
summary: Single-stage detectors — the prediction head and its output, anchor boxes and ground-truth assignment, bounding-box parameterisation, the detection loss, focal loss for class imbalance, multi-scale prediction with SSD and feature pyramid networks, depthwise separable convolution and non-maximum suppression.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain how a single-stage detector obtains object locations from the **feature maps** of an image classifier;
- describe the **prediction head** and the structure of its **output**;
- explain the role of **anchor boxes** and how ground truth is **assigned** to them;
- calculate the **bounding-box parameterisation** $t_x, t_y, t_w, t_h$ and decode it;
- describe the **detection loss**, including the **smooth L1** loss;
- explain the **imbalance** between positive and negative examples and how **focal loss** addresses it;
- explain the **scale problem** and how **SSD** and **feature pyramid networks** use multi-scale feature maps;
- explain **depthwise separable convolution** and **non-maximum suppression**.

## Introduction

In the previous lesson the R-CNN family of two-stage detectors, which use region proposals, was described. Two-stage detectors are considered to be the most accurate object detectors, but they are computationally expensive. In this lesson, **single-stage** object detectors are described. They are still based on deep CNN architectures, but are less computationally expensive and faster, which often makes them suitable for mobile and embedded applications.

## Principles of single-stage detectors

In the naive approach, an image classifier is used as a sliding-window detector. As image classifiers are trained on a fixed image size, such as $224 \times 224$ pixels, the network would be applied at overlapping windows of a large image, and an image pyramid would be needed to detect objects larger than the window. This is technically feasible, and is how classical object detectors work, but it is very inefficient for a deep network, which may be applied thousands of times to one image.

Instead, the information about the whole image that is available in the convolutional part of an image classifier is used. The output feature maps contain both the spatial position and semantic information about the objects in an image.

{{fig feature_location}}

The location of the activation gives coarse information about the location of an object, at the resolution of the feature map. The feature maps are also sparse, as most channels are not activated at any given position, so the bounding box cannot be read directly from them; instead, a prediction head is trained to regress the box from the features. Each image needs to be processed only once by the convolution layers. Accurate localisation, particularly of small objects, requires feature maps of higher resolution, which is addressed by the multi-scale detectors described later in this lesson.

## The prediction head

A single-stage detector is constructed by adding a **prediction head**, itself a small convolutional network usually with $3 \times 3$ kernels, to the output feature map of a pre-trained image classifier. As the head moves over the feature map, it predicts a confidence, which indicates whether an object is present in the current region of the feature map, together with the class and the bounding box of the object. Each cell of the feature map has a large receptive field, so the features at a cell are computed from a large region of the image, and small kernels are sufficient for the head.

{{fig detection_system}}

{{fig head_sliding}}

{{fig prediction_head}}

The bounding box predicted by the head is relative to the current position in the feature map, so it must be corrected by that coarse position to obtain coordinates in the image. The predictions are output as a vector at each cell: for each box, four coordinate values and a confidence, together with a set of class probabilities, usually a softmax output. Several boxes are commonly predicted at each cell, as a consequence of the use of anchor boxes. The confidence does not always have to be predicted separately; instead a background class can be added to the classes, although the background class then dominates training. The class is usually predicted once for each cell, although it can be predicted for each anchor at the cost of a larger network.

## Anchor boxes and ground-truth assignment

Training an object detector is more difficult than training an image classifier, as there are multiple ground-truth objects of different sizes and classes at different locations in each image. The feature-map cells that contain valid objects must be identified and assigned the correct ground truth. **Anchor boxes** are used for this purpose. If an anchor box overlaps a ground-truth box with an IoU above a threshold, the anchor is selected as a positive prediction, and the head is trained to predict the offset and scale of the ground truth from the anchor.

{{fig gt_assignment}}

{{fig anchor_assignment}}

Anchor boxes act as a prior, which can give faster training convergence, as some of the anchors are already close to the size and shape of the objects in the dataset. The anchor dimensions are set in advance for the types of object in the dataset, which can bias the detector. Anchors work well for large objects with wide aspect ratios, such as aeroplanes, but small objects can be difficult to learn. As a set of anchors is placed at each cell, multiple anchors can be selected when objects overlap at the same location. The computation of the IoU of every anchor with every object adds to the computation during training, but allows several anchors in the same or neighbouring cells to be positive, which increases the number of positive examples. All anchors whose IoU is below the threshold are used as negative examples for the confidence, so there are always many more negative than positive examples.

## Bounding-box prediction

The head does not predict the bounding-box coordinates directly. Instead, it predicts the displacement of the box centre from the anchor centre, and the scaling of the box height and width from the anchor height and width:

$$t_x = \frac{x - x_a}{w_a}, \qquad t_y = \frac{y - y_a}{h_a}, \qquad t_w = \log\frac{w}{w_a}, \qquad t_h = \log\frac{h}{h_a}, \tag{1}$$

where $x_a, y_a, w_a, h_a$ are the centre, width and height of the anchor, which is centred on the current cell, and $x, y, w, h$ those of the predicted box. The ground-truth values $t_x^{gt}, t_y^{gt}, t_w^{gt}, t_h^{gt}$ are obtained with the same equations from the ground-truth box. These equations are those of the Faster R-CNN region proposal network, which is functionally equivalent to a single-stage detector; many variations are used in the literature.

{{fig box_dimensions}}

{{fig box_parameterisation}}

## The detection loss

The overall loss is a function of the predicted object probabilities $\{p_i\}$ and the predicted box vectors $\{t_i\}$ of all anchors:

$$L(\{p_i\}, \{t_i\}) = \frac{1}{N_{conf}} \sum_i L_{conf}(p_i, p_i^{gt}) + \lambda \frac{1}{N_{reg}} \sum_i p_i^{gt} L_{reg}(t_i, t_i^{gt}), \tag{2}$$

where $p_i^{gt}$ is 1 when the anchor overlaps a ground-truth box with an IoU above the threshold and 0 otherwise. The two terms are normalised, with normalisations that vary between implementations, and balanced by $\lambda$. The confidence loss $L_{conf}$ is a binary cross-entropy, as the confidence is a binary classification between object and background, and both positive and negative anchors contribute to it. The factor $p_i^{gt}$ ensures that only positive anchors contribute to the regression loss. $L_{reg}$ is the sum of **smooth L1** losses over the four box values $t_k$, $k \in \{x, y, w, h\}$, of the anchor:

$$L_{reg}(t_i, t_i^{gt}) = \sum_{k \in \{x, y, w, h\}} \text{smooth}_{L1}\left(t_k - t_k^{gt}\right), \tag{3}$$

where each smooth L1 loss is calculated on the difference between a predicted and a ground-truth value:

$$\text{smooth}_{L1}\left(t_k - t_k^{gt}\right) = \begin{cases} 0.5\, \left(t_k - t_k^{gt}\right)^2 / \beta & \left|t_k - t_k^{gt}\right| < \beta \\ \left|t_k - t_k^{gt}\right| - 0.5\, \beta & \text{otherwise.} \end{cases} \tag{4}$$

The threshold $\beta$ is commonly set to 1.

{{fig detection_loss}}

The classification loss can be included as a third term, using only the positive examples. It is the same as for image classification: a categorical cross-entropy over the softmax outputs, or a binary cross-entropy for each class against all others when multi-label outputs are required.

## Focal loss for training imbalance

The large imbalance between positive and negative examples means that the easily classified negative examples dominate the loss and may prevent the network from learning the more difficult positive examples. The cross-entropy loss can be written with the probability of the ground-truth class, $p_t$, which is $p_i$ when $p_i^{gt} = 1$ and $1 - p_i$ otherwise:

$$CE(p_t) = -\log(p_t). \tag{5}$$

The cross-entropy does not fall to zero unless an example is exactly correct, so many well-classified examples together produce a large loss, which is exactly the situation with the negatives in object detection. The **focal loss**, introduced by Lin et al. in 2017, adds a scaling factor:

$$FL(p_t) = -(1 - p_t)^{\gamma} \log(p_t). \tag{6}$$

As $\gamma$ is increased, the loss of well-classified examples is reduced towards zero, and a value of $\gamma = 2$ is typically chosen. The effect of the negative examples dominating the loss is removed, and the hard positive examples can be learned.

{{fig focal_loss}}

Focal loss is widely used in object detection. When cross-entropy is also used for the individual object classes, focal loss can be used there too, to help with rare categories for which there are few examples.

## The scale problem

A basic single-stage detector makes its predictions from a single output feature map, usually the last convolution layer of the image classifier, which has a coarse spatial resolution and a wide receptive field and can usually be trained to localise only large objects. Small objects can be detected by increasing the image resolution or by using an image pyramid, at the cost of extra computation. A more efficient approach is to use feature maps of the image classifier at different scales: larger objects are detected in coarse feature maps with large receptive fields, and smaller objects in fine feature maps with smaller receptive fields. Each feature map is assigned a nominal scale, and the anchor boxes of that map are sized for objects around it.

{{fig scale_grids}}

{{fig fpn_levels}}

The fine feature maps of the image classifier may, however, not perform as well as expected, as their filters extract more generic features and may not contain explicit semantic information about small objects. SSD and feature pyramid networks take different approaches to ensuring semantic information in the fine feature maps.

## SSD

The **Single Shot MultiBox Detector** (SSD) is a well-known detector that uses multi-scale feature maps. SSD is based on VGG-16. The earliest, finest feature maps of VGG-16 are not used, as they contain little semantic information about objects. The finest prediction map is taken from the conv4_3 layer, at a stride of 8 ($38 \times 38$ for a $300 \times 300$ input), with an L2 normalisation applied to it, and the remaining maps are taken from the converted fully connected layer fc7, after conv5_3, and from extra convolution layers that progressively downsample it. An increased input size is used to allow enough spatial resolution for downsampling between the maps. Boxes and classes are predicted at each of the maps with $3 \times 3$ convolutions, relative to anchor (default) boxes.

{{fig ssd_architecture}}

{{fig ssd_multiscale}}

## Feature pyramid networks

The **feature pyramid network** (FPN) is another multi-scale approach. It also uses a pre-trained image classifier, here ResNet-50, as the backbone, but takes a different approach to the poor semantic information of the fine feature maps: a parallel pyramid is generated, starting from the semantic output of the image classifier. This starting feature map is upsampled and combined with the fine feature maps, and two further levels are created by downsampling. A common prediction head is used for all levels of the pyramid, so all levels must have the same depth.

{{fig fpn_detector}}

FPN gives good performance for small objects. It can be trained on input images of different sizes, with larger images giving better performance on small objects at the cost of computation, and it is used in many modern object detectors.

## Depthwise separable convolution

Lightweight detectors for mobile and embedded applications use backbones built from **depthwise separable convolutions**, as in MobileNet. A standard convolution is split into a depthwise convolution, which filters each channel separately, and a $1 \times 1$ pointwise convolution, which combines the channels, with a large reduction in parameters and computation.

{{fig depthwise_separable}}

{{example depthwise_sep_conv seed=4}}

## Non-maximum suppression

A dense detector produces several overlapping detections of the same object, from neighbouring cells and anchors. **Non-maximum suppression** (NMS) keeps the detection with the highest confidence and removes the others that overlap it by more than an IoU threshold, so that each object ends with one detection.

{{fig nms_steps}}

{{example iou seed=12}}

{{selfcheck 03_04}}
