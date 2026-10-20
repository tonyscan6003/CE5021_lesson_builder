---
id: "03_03"
title: Two-stage object detection
section: 3
engines: [s3_detection.two_stage]
practice: [rfield]
summary: The R-CNN family of two-stage detectors — R-CNN with CNN features, spatial pyramid pooling, Fast R-CNN with region of interest projection and pooling, class activation maps, and Faster R-CNN with its region proposal network — and how two-stage detectors compare with single-stage detectors.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- describe the **R-CNN** pipeline and explain its problems;
- explain how **spatial pyramid pooling** produces a fixed-length vector from feature maps of any size;
- describe **Fast R-CNN**, including **region of interest projection** and **RoI pooling**;
- explain what a **class activation map** shows and why it matters for object detection;
- describe the **region proposal network** of **Faster R-CNN** and its anchors;
- compare **two-stage** and **single-stage** detectors in accuracy and speed.

## Introduction

In this lesson object detection with deep convolutional neural networks is discussed, with a focus on frameworks that use region proposals to achieve high-accuracy object recognition: R-CNN, where R stands for region, and its successors Fast R-CNN and Faster R-CNN.

The original R-CNN has several problems, which were introduced in Lesson 3.1. The region proposal generator produces regions of different shapes and sizes, which is a problem for neural networks that require a fixed input size; in R-CNN the regions are warped to a fixed size, but a modification of the network architecture to accept inputs of any size, through spatial pyramid pooling and region of interest pooling, is a more elegant solution. The image classifier operates on overlapping regions several times, which is inefficient; with region of interest projection, introduced in Fast R-CNN, the whole image is processed once by the convolution layers. Finally, the separate selective search algorithm prevents end-to-end training; in Faster R-CNN it is replaced by a neural network equivalent.

{{fig rcnn_family}}

## R-CNN

R-CNN was one of the first examples of transfer learning. In a typical machine learning system, images are converted into feature vectors that are classified by a support vector machine. In R-CNN the engineered features are replaced by a pre-trained convolutional neural network, which has learned its features on the image classification task. The classification is still performed by an SVM, so the classifier is trained independently of the CNN and end-to-end learning is not possible.

R-CNN is a multi-stage pipeline. Selective search is used to identify region proposals. As the proposals have different sizes, each must be warped to the square input size of AlexNet. The image patches are used to fine-tune AlexNet, modified for a reduced number of classes. Once AlexNet has been fine-tuned, its fully connected part is removed, and the convolution part is used as a feature generator. The warped proposals are sorted into binary categories, such as "car" and "not car", features are extracted from them, and an SVM is trained for each class; a bounding-box regressor is also trained on the CNN features. At inference, proposals are generated for the new image, warped, passed through the CNN, and classified and refined by the SVMs and the regressor.

Each stage must be trained individually. The system is also computationally slow, as every one of the hundreds or thousands of proposals for an image is processed by the CNN, and the features of every training example must be stored in order to train the SVMs and the regressor, which requires hundreds of gigabytes.

{{fig rcnn_cost}}

## Towards end-to-end learning

R-CNN and its variants are conceptually two-stage detectors. The first stage consists of the region proposal generator and the convolution layers, which convert a proposal into a feature vector. The second stage operates on this vector and predicts the class of the object and a refined bounding box within the proposal. The first step towards end-to-end training is to replace the SVM and the bounding-box regressor with fully connected layers, which perform the same functions but are connected directly to the convolution layers. However, a fixed-size vector is still required for each proposal.

## Spatial pyramid pooling

The size limitation arises from the fully connected layers of the network, not the convolution layers. Convolution layers can take inputs of any size, and the spatial size of the output feature maps changes with the input. The fully connected layers, however, require a fixed-size input. In **spatial pyramid pooling** (SPP), introduced by He et al. in 2014, max pooling is applied at several scales to produce a fixed-length vector. At each level of the pyramid a different number of pooling bins is used, and the size of the pooling windows changes with the size of the feature map, so that each level produces the same number of values. The values are concatenated into one vector and passed to the fully connected layers.

{{fig spp_layer}}

SPP can be included in any network, but the network must be retrained to include it.

## Fast R-CNN

Fast R-CNN has a shorter training time, reduced computation and a higher speed than the original R-CNN. The separate SVM classifiers and bounding-box regressor are replaced by fully connected layers, which perform both the class prediction and the bounding-box regression. These are trained together with a **multi-task loss** function, which is discussed further in the next lesson. In addition, the whole image is processed once by the convolution layers, and the regions are defined on the output feature map by **region of interest projection**.

The fully connected layers still require a fixed-size input, so **RoI pooling** is used to reduce each region of the feature map to a fixed-size volume. RoI pooling is a single level of spatial pyramid pooling, and its output is always $7 \times 7$ per channel, regardless of the size of the region.

## Class activation maps

**Class activation maps** demonstrate that information about the spatial location of objects is present in the output feature maps of a convolutional neural network. It was shown by Oquab et al. and by Zhou et al. that image classification networks already contain an indication of object location in their feature maps. In the final convolution layer, each channel of the feature map has a spatial distribution of low and high activation values. With global average pooling, each channel is converted to a single value, and the resulting vector is processed by one fully connected layer, which makes a decision on the class from a weighting of these values. For the identified class, the same weights are used to sum the feature maps of the last convolution layer, producing the class activation map.

{{fig class_activation_map}}

Class activation maps show that CNNs also learn to perform unsupervised object localisation while being trained for image classification with labels only. This spatial information allows region of interest projection, and can also be used to predict region proposals directly, replacing selective search.

## Region of interest projection

As the spatial position of objects is retained in the output feature maps of the backbone network, the region proposals generated by selective search on the original image can be projected onto the lower-resolution feature maps. The main advantage of region of interest projection is that the convolution is performed just once, rather than once per proposal as in R-CNN.

{{fig roi_projection}}

The projected regions of varying sizes are converted to vectors of a standard size by RoI pooling, and these vectors are processed by the second stage of the network to predict classes and bounding boxes.

{{fig roi_pooling}}

{{example rfield seed=7}}

The disadvantage of Fast R-CNN is that selective search is still used to determine the region proposals. It is trained independently of the rest of the network, so the system is not yet trained fully end to end.

## Faster R-CNN and the region proposal network

In Faster R-CNN, the external region proposal generator is eliminated. As shown by class activation maps, object location information is already available in the backbone network, and the convolutional features are used as the input to a small **region proposal network** (RPN). The RPN can be considered as part of a larger network that shares its convolution layers with the feature-generating CNN. At every position of the feature map, a set of reference boxes called **anchors**, of several sizes and aspect ratios, is scored for objectness, and an adjustment to each anchor is predicted.

{{fig rpn_proposals}}

Training uses an alternating scheme with the Fast R-CNN part of the system. The RPN is trained first, starting from a backbone with ImageNet weights; the Fast R-CNN part is then trained with the RPN proposals; the common convolution layers are then fixed and only the RPN-specific layers are fine-tuned; finally, the common layers are again fixed and the Fast R-CNN part is fine-tuned.

{{fig rpn_recall}}

## Comparison of two-stage and single-stage detectors

Faster R-CNN can be described as a two-stage detector. In the first stage, bounding boxes are predicted by the region proposal network and projected onto the output feature maps; in the second stage, the regions are converted to vectors by RoI pooling, and refined bounding boxes and classes are predicted. A single-stage detector predicts bounding boxes and classes directly from the output feature map, so the region proposal network and a single-stage detector are functionally equivalent. The additional stage of a two-stage detector provides more accurate bounding-box estimation and classification, at the cost of a slower frame rate. Modern single-stage detectors have closed the gap in accuracy and have higher frame rates, which makes them the default choice in many applications. Single-stage detectors are described in the next lesson.

{{fig two_vs_single_stage}}

## Summary

Object detection with deep convolutional neural networks has advanced rapidly in parallel with image classification. The R-CNN family uses region proposals, which helps to achieve high accuracy, and can be considered as an object detection framework, with many published variations using different enhancements and backbone networks. The key advances of the family are as follows:

- **R-CNN**: the use of CNN features for object detection, in place of hand-crafted features;
- **Fast R-CNN**: region of interest pooling, CNN-based classification and regression for end-to-end training of the second stage, and a reduction in computation, as the CNN is applied once per image;
- **Faster R-CNN**: a region proposal network, which removes the external region proposal generator.

{{selfcheck 03_03}}
