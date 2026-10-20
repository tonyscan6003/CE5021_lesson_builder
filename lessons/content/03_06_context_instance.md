---
id: "03_06"
title: Context and instance segmentation
section: 3
engines: [s3_detection.context_instance]
practice: [dilated_conv]
summary: Context in semantic segmentation — the receptive field, dilated convolution, pyramid scene parsing, conditional random fields and the DeepLab framework — and instance segmentation with Mask R-CNN and RoIAlign.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain why **context** matters in semantic segmentation and how it relates to the **receptive field**;
- describe **dilated convolution** and calculate the receptive field of stacked dilated convolutions;
- describe **pyramid scene parsing**;
- explain the role of a **conditional random field** in segmentation;
- describe the **DeepLab** framework;
- explain the approaches to **instance segmentation**, and describe **Mask R-CNN** and **RoIAlign**.

## Introduction

In this lesson the use of context in semantic segmentation is discussed, and instance segmentation is briefly introduced.

## Context knowledge

As was seen in the previous lesson, image classification networks are used, with some modification, for pixel-level prediction. The spatial information about a scene is reduced by the pooling and downsampling layers of the CNN, while the semantic information about the scene and the objects in it increases through the network. Semantic segmentation networks can mislabel objects because of a lack of context: in published examples, pixels of a bedroom window are labelled as a door, and a boat on a lake is classified as a car. A system with high-level knowledge of the context of the image would recognise that a car is unlikely to be found on water and that the object is more likely to be a boat. More generally, it is assumed that a network with a larger receptive field allows the deep neurons to capture the context of the scene more accurately.

{{fig effective_receptive_field}}

In the previous lesson, spatial information from skip connections and unpooling was used to improve the accuracy at the pixel level. In this lesson the techniques of dilated convolution, pyramid scene parsing and conditional random fields are described.

## Dilated convolutions

A popular way to increase the receptive field of the neurons in a network is to use **dilated convolutions**, also known as **atrous convolutions**. Dilated convolutions should not be confused with the transpose, or fractionally strided, convolutions of the previous lesson. In a dilated convolution the input samples are spaced out, and the convolution is performed with these samples; the number of parameters stays the same. When a regular $3 \times 3$ convolution is applied, 9 adjacent pixels contribute to each output pixel, and the receptive field is $3 \times 3$. A $3 \times 3$ convolution with a dilation factor of 2 covers a $5 \times 5$ region, and when it is applied to the output of the first convolution, each pixel of the final output has a receptive field of $7 \times 7$.

{{fig dilated_conv}}

{{example dilated_conv seed=5}}

Cascading dilated convolutions produces an exponential growth in the receptive field for only a linear growth in the number of parameters. This property makes dilated convolutions well suited to semantic segmentation, and the technique has been widely adopted: the regular convolution layers of an architecture such as VGG can be replaced with dilated convolutions at no additional computational cost, and the receptive field is greatly widened.

{{fig dilated_stack}}

## Pyramid scene parsing

**Pyramid scene parsing** can also improve semantic segmentation. It is similar to the spatial pyramid pooling used in object detection, in that information about the location of features in the output feature map is retained. A ResNet backbone is used, with some convolutions replaced by dilated convolutions to increase the receptive field. The output feature map is pooled to further feature maps of different spatial dimensions, whose depth is reduced by convolution; each is then upsampled and concatenated with the original feature map, so that spatial information at different scales is aggregated. A final convolution layer produces the pixel-level prediction.

{{fig pyramid_scene_parsing}}

This approach differs from those of the previous lesson. A decoder network and skip connections are no longer relied on to reconstruct the fine detail of the image: with dilated convolution the effective downsampling of the network is reduced, and together with pyramid scene parsing or other techniques that retain spatial information, a good semantic reconstruction of the input image can be obtained without a decoder.

## Conditional random fields

**Conditional random fields** (CRFs) can be used to model the relationships between pixels explicitly: the intensity and position of the pixels are used to predict the pixel labels. The CRF can be considered as a fully connected graph, in which the weighting between nodes helps to determine whether pixels are likely to belong to the same class. CRFs were used for semantic segmentation in machine vision before the adoption of deep learning, and, like a neural network, a CRF is a discriminative model. It is computationally intractable to learn a CRF for a large image at the pixel level, but efficient algorithms such as the dense CRF allow approximate solutions to be found with reasonable computation. CRFs alone can produce reasonable performance, but they are often included as a post-processing step in deep learning segmentation systems to achieve state-of-the-art results.

{{fig dense_crf}}

## DeepLab

**DeepLab** is a well-known, state-of-the-art segmentation framework. A backbone network such as VGG or ResNet is used, with dilated convolutions replacing some of the regular convolutions, which reduces the effective downsampling of the network without increasing the number of parameters. A spatial pyramid pooling stage is included to deal with objects of different scales. Instead of a learned deconvolution stage, bilinear interpolation is used to upsample the output feature map, and no skip connections are required. In DeepLab a conditional random field is used as a post-processing method to further improve the segmentation; the CRF can be trained together with the network, or the whole system can be interpreted as a recurrent neural network.

{{fig deeplab_framework}}

## Instance segmentation

**Instance segmentation** combines the object detection and semantic segmentation tasks, with the aim of identifying the different instances of the same object in a scene. As was seen in the lesson on region proposals, segmentation is a more fundamental task than object detection, and many instance segmentation systems perform segmentation before recognition. If segmentation proposals can be generated, a deep network can classify them and the pixels of each proposal can be labelled as a unique instance. Alternatively, a semantic segmentation of the scene is performed first, in which multiple instances of an object are labelled with the same class, and object proposals are then used to cut the semantically segmented regions into instances. Both methods are cumbersome and may not be trainable end to end.

{{fig instance_vs_semantic}}

The state of the art in instance segmentation is **Mask R-CNN**, a refinement of Faster R-CNN, in which class labels and a binary mask identifying the region of each object instance are predicted simultaneously, with a very low additional computational cost.

## Mask R-CNN

In Mask R-CNN an $m \times m$ mask is predicted for each object by a fully convolutional network applied to the region of interest determined by the region proposal network. As convolution is used in the mask path, the outline of the object is retained from the feature map. To produce an accurate mask, the RoI pooling stage of Faster R-CNN is replaced by **RoIAlign**. In RoI pooling, a region of interest of any shape or size is converted into a fixed-size output: the region proposal is quantised to the feature map, and pooling is applied within bins of the region. In RoIAlign, no quantisation of the region proposal is performed, and the values of the feature map are instead interpolated at points within the bins of the region. This helps to preserve the spatial correspondence between the features used to generate the mask and the proposed region, and the removal of quantisation was found to be the key to good mask prediction.

{{fig roi_align}}

The output masks of Mask R-CNN are of low resolution and are projected onto the original image; the resolution is limited by the size of the mask path. One mask is predicted for each class, and only the mask that corresponds to the class predicted by the classification branch is used.

{{fig mask_rcnn_output}}

{{selfcheck 03_06}}
