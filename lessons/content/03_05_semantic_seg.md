---
id: "03_05"
title: Semantic segmentation
section: 3
engines: [s3_detection.semantic_seg]
practice: [mat_conv, transpose_conv1, transpose_conv2]
summary: Semantic and instance segmentation, segmentation datasets, the output of a segmentation network, IoU for segmentation, fully convolutional networks, transpose convolution, encoder–decoder networks with unpooling, and the U-Net.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- distinguish **semantic** and **instance** segmentation;
- name the main **segmentation datasets**;
- describe the **output** of a segmentation network and how it is converted to a label map;
- calculate the **IoU** and **mean IoU** of a segmentation;
- explain how a classifier is converted into a **fully convolutional network**, and why its output must be upsampled;
- express a convolution as a **matrix multiplication** and explain the **transpose convolution**;
- calculate a **transpose convolution** emulated by direct convolution;
- describe **encoder–decoder** networks with **unpooling**, and the **U-Net**.

## Introduction

In this lesson the semantic segmentation of images with neural networks is described.

## Segmentation

Simple segmentation methods, such as the watershed algorithm and mean-shift segmentation, divide an image into regions based on the similarity of the pixels. In **semantic segmentation** the image is also divided into regions, but each pixel is assigned a specific class, based on the object to which it belongs. Semantic segmentation can be considered as image classification with an individual class prediction at each pixel. In the related task of **instance segmentation**, the information produced by object detection is combined with semantic segmentation, so that multiple instances of objects are individually segmented. Instance segmentation has become the key task in competitions such as those run on the MS COCO dataset, as it is more challenging than either semantic segmentation or object detection.

{{fig segmentation_types}}

## Datasets

As with object detection, many datasets are available for semantic segmentation. The development of a segmentation dataset is difficult, as it requires time-consuming human annotation of every pixel. Many classification and detection datasets also contain segmentation annotations, for example Pascal VOC, and the MS COCO dataset contains annotations for instance segmentation. The ADE20K dataset was developed specifically for segmentation, and contains annotations for object parts as well as objects. Semantic segmentation is an important component of autonomous driving systems, as highly detailed representations of the environment are required, and datasets of driving scenes such as Cityscapes and KITTI have been developed for it.

{{fig segmentation_datasets}}

## The output of a segmentation network

In image classification and object detection, a vector of class probabilities is output, and the class is usually found with an argmax operation across the vector. In semantic segmentation a class must be predicted at every pixel, so an $H \times W \times N_{classes}$ tensor is output. The argmax is applied across all classes at each pixel position to find the class with the highest probability, and the classes are gathered into a single RGB image, with a false colour assigned to each class.

{{fig output_tensor}}

{{fig segmentation_output}}

## Evaluating semantic segmentation

The **intersection over union** is the primary measure of the performance of a semantic segmentation system. It is calculated for each class at the pixel level:

$$IoU = \frac{TP}{TP + FP + FN}. \tag{1}$$

Where the ground truth and the prediction overlap, a true positive is recorded; a pixel of the class that is not predicted is a false negative; and a pixel incorrectly predicted as the class is a false positive. The IoU is obtained for each class in the image and the mean value is calculated. It is the same measure as is used for object detection: the ratio of the intersection to the union of the ground truth and the prediction.

{{fig segmentation_metrics}}

## Fully convolutional networks

The first neural network for semantic segmentation described here is the **fully convolutional network** (FCN). It was not the first system to use neural networks for semantic segmentation, but it is reasonably simple, can be trained end to end, and is recognised as an important development; similar architectures were used in many later systems. FCN is based on a pre-trained image classifier. In an image classifier, information from across the image is aggregated into an output feature map by several convolution stages, and the feature map is converted into a vector of class probabilities by a fully connected network. For semantic segmentation the class of each pixel must be predicted, so the classifier is modified. First, the fully connected stages are converted into convolution stages, as in SSD, so that the network is fully convolutional and accepts inputs of any size.

{{fig fcn_architecture}}

{{fig fully_convolutional}}

At this point the semantic content is high, but the spatial information is greatly reduced, as the feature maps are very coarse. To produce predictions at the pixel level, the output of the network must be upsampled. FCN has several output paths with different degrees of upsampling, and the upsampling is performed with transpose convolutions. When the output is upsampled directly by a factor of 32, the result is ill defined. A key finding of the FCN authors was that, with feedforward **skip connections** from earlier stages of the network, the spatial information still present in those stages can be used to improve the quality of the prediction.

{{fig output_resolution}}

## Transpose convolution

The transpose convolution is used by FCN and many other networks to upsample feature maps. A convolution can be performed as a matrix multiplication: when a $3 \times 3$ input is convolved with a $2 \times 2$ kernel, the input can be reshaped as a $9 \times 1$ vector and multiplied by a $4 \times 9$ matrix containing repeated values of the kernel, giving a $4 \times 1$ output that is reshaped into a $2 \times 2$ matrix. The transpose of this matrix, $9 \times 4$, multiplied by a $4 \times 1$ vector gives a $9 \times 1$ vector, which is reshaped into a $3 \times 3$ matrix. A regular convolution always reduces the spatial dimensions, and a transpose convolution increases them. The transpose convolution is also used in the backpropagation algorithm.

{{fig conv_as_matrix}}

{{example mat_conv seed=3}}

In deep learning frameworks such as PyTorch, the transpose convolution is not performed with the transposed matrix. Instead, it is emulated by direct convolution, and the weights for the upsampling are learned with the other weights of the network. The emulated transpose convolution is also known as **fractionally strided convolution**. When a $3 \times 3$ kernel is convolved with a $4 \times 4$ input, a $2 \times 2$ output results; for the transpose, a $2 \times 2$ input must give a $4 \times 4$ output, which is achieved by padding the input with two zeros on each side and convolving with a $3 \times 3$ kernel. When a $5 \times 5$ input is convolved with a $3 \times 3$ kernel at stride 2, the transpose is obtained by inserting a zero between the samples as well as padding, followed by a direct convolution with a $3 \times 3$ kernel at stride 1, which gives a $5 \times 5$ output. The second case is the more common in semantic segmentation, where convolution with stride 2 is used for downsampling and fractionally strided convolution for upsampling.

{{fig transpose_1d}}

{{fig transpose_conv}}

{{example transpose_conv1 seed=4}}

{{example transpose_conv2 seed=4}}

## Encoder–decoder networks

Encoder–decoder networks are an extension of the fully convolutional network. DeconvNet and SegNet proposed the same architecture, in which a complete decoding network is included after the encoding part. These networks use the encoding, or downsampling, part of VGG-16, and learn the decoding, or upsampling, part with end-to-end training. To obtain good-quality predictions, they retain and reuse the spatial information that is normally discarded by the pooling stages of the encoder. In a regular max pooling operation, the largest activation in each region is retained, and the spatial information is lost. In the encoder–decoder networks, the positions of the maxima are stored during pooling; in the decoder, an **unpooling** operation places the input values at the stored positions, and the other elements of the output remain zero. A transpose convolution is then used to interpolate this sparse feature map.

{{fig segnet_architecture}}

{{fig unpooling}}

FCN does not use unpooling, but only upsamples its feature maps. Unpooling helps to recover spatial information, and allows these networks to recover finer detail than FCN. However, unpooling operations are not standard in most deep learning frameworks and may be difficult to implement.

## U-Net

Another well-known encoder–decoder network is the **U-Net**. In contrast to DeconvNet and SegNet, U-Net does not use unpooling layers. Instead, like FCN, it uses skip connections, which transfer the spatial information of the encoding path to the decoding path, where the feature maps are **concatenated**. As with R-CNN in object detection, the U-Net architecture can be used as a framework, and the approach can be used to convert any image classification network into a segmentation network.

{{fig unet}}

{{selfcheck 03_05}}
