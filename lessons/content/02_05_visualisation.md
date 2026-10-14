---
id: "02_05"
title: Feature visualisation and adversarial attacks
section: 2
engines: [s2_classification.visualisation]
practice: []
summary: Four techniques for visualising what a network has learned — reconstruction, activation maximisation, style transfer and deep dream — and the limitations of trained networks — adversarial attacks and failures of generalisation.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain why the visualisation of learned features is important for **explainable** deployment;
- describe **reconstruction** (inversion) and what it shows about depth;
- describe **activation maximisation** and why gradient **ascent** with a regulariser is used;
- explain how **style** is captured by **Gram matrices** and how style and content losses are combined in **style transfer**;
- describe **deep dream**;
- explain what is meant by an **adversarial example**, how one is produced by the **fast gradient sign method**, and the significance of physical attacks;
- explain the **limits of generalisation** of trained networks.

## Introduction

In the previous lessons, networks have been treated as image classifiers, and it has been noted that features are learned by their convolution layers during training. However, it is not clear what has been learned by a network, or which features it responds to when a test image is classified. The investigation of learned features makes networks more **explainable**, which is helpful when they are deployed in real applications.

In the first part of this lesson, four techniques used to understand what a network has learned are described. All four are based on the same principle: the network is kept **fixed**, and gradient-based optimisation is used to modify the **input image**.

The figures in this lesson are produced with a small **fixed** network, so that the techniques can be run without a deep learning framework. The network consists of a bank of oriented Gabor filters followed by ReLU and pooling, with deeper layers obtained by further pooling. Gabor filters are similar to the filters typically learned in the first layer of a trained CNN, so the network is a reasonable model of the *early* layers, and each algorithm can be run exactly as described. However, the network has not been trained, so the figures show how each technique operates and how depth affects the information retained by a layer, but not what is learned in the deep layers of a trained network. Where a figure is produced with a trained model, this is stated in the caption.

## Reconstruction

Reconstruction, also known as **inversion**, is used to determine what information about an image is retained in the feature maps of a network.

{{fig reconstruction}}

The target image is applied to the network and the features at the layer of interest are stored. An image of random noise is then applied, and its features at the same layer are compared with the stored features in a loss function. The noise image is updated by gradient descent on each iteration, so that the loss approaches zero and the noise image converges to an image that resembles the target. Regularisation terms are added to the loss to help produce natural-looking images.

Reconstruction from early layers is very accurate, as many low-level features are captured. Reconstruction from deeper layers is less accurate, and eventually only an abstract composition of parts of the image can be recovered. In a trained network this is consistent with **concepts** about objects being learned in the deep layers from large numbers of training images.

## Activation maximisation

Activation maximisation is used to visualise the features learned by the convolution layers. A neuron, channel or layer is chosen, and the input image that **maximises** its output is found. This is the feature to which the neuron would respond most strongly if it appeared in a real image.

{{fig activation_maximisation}}

Starting from random noise, the value of the neuron is monitored and the pixels are updated by **gradient ascent**, rather than descent, as the quantity of interest is to be *maximised*. A **regularisation penalty** is required, as the neuron would also be maximised by many unnatural noise patterns.

For VGG-16, when one point of a feature map in each layer is maximised, it is found that the images become larger in deeper layers, as the **receptive field** increases. Early layers respond to colour blobs and edges, the following layers to more complex patterns that may be textures, and the deeper layers to objects or parts of objects.

## Visualisation for understanding

Visualisation of a network can also be used to show the features it detects in images of a given class.

{{fig class_templates}}

When images produced by activation maximisation are compared with the training images to which a neuron responds most strongly, it is found that objects may be recognised by properties that are simply common to the training images, such as the texture of clouds or the presence of sky behind buildings. In many cases the object itself is not recognised; instead, the network relies on features that **co-occur** with the object, and this can cause confusion between classes when the network is deployed.

## Style transfer

In style transfer, the style of one image, usually an artwork, is transferred onto the content of another. This is very difficult with traditional computer vision, but is possible with a pre-trained network as a result of what the network has learned.

As shown by reconstruction, sufficient information is held in the early layers for an image to be reconstructed accurately, while information about objects and their relationships, the **content**, is held in the deeper layers without fine detail. **Style**, or texture, information is also retained by a network, and is obtained from the inner products of the feature maps, which form the **Gram matrix**.

{{fig texture_statistics}}

Texture appears at different scales, with small-scale texture obtained from early layers and large-scale texture from deeper layers.

{{fig style_transfer_loop}}

The style-transferred image is created as follows:

1. the **content** image is passed through the pre-trained network and a deep feature map (conv4 in the original work) is stored;
2. the **style** image is passed through the network and the texture information, the Gram matrices, of all the convolution layers is stored;
3. a **random noise** image is passed through the network, and its features and Gram matrices are calculated;
4. these are compared with the stored content and style information in the loss function, and the pixels of the noise image are adjusted by gradient descent to minimise both losses together.

The noise image gradually converges to an image with the content of the content image in the style of the style image. Style transfer shows that both content and style information are held by a network, and also that the two can be **recombined** to form new images.

## Deep dream

Deep dream is similar to activation maximisation, except that a **content image** is applied to the network instead of random noise.

{{fig deep_dream}}

The network responds to features that are already present in the content image, and these features are then **amplified** by gradient ascent. When the outputs of many layers are maximised together, several features tend to be amplified at once. Particular content biases the network towards particular features: images with clear horizon lines produce towers and pagodas, trees produce features of buildings, and leaves produce birds and insects. These results reveal associations that have been learned by the network between image features and concepts. When deep dream was applied iteratively, with the result enlarged between iterations, a series of abstract images was produced, which were among the earliest examples of AI-generated art.

## Adversarial attack

The high performance of trained networks has been described in the previous lessons. However, before a network is deployed, especially in a safety-critical application, its **robustness** must be understood.

With **adversarial examples**, a network is made to misclassify an input that has been manipulated. In a well-known example, only a very small amount of a noise pattern is added to an image of a panda for a completely different label to be output by the network, with *higher* confidence than the original prediction, while to a human observer the image appears unchanged.

{{fig adversarial_fgsm}}

The added pattern causes the incorrect output neuron to be activated. As a result of the internal structure of the network, with many multiplications and non-linear activation stages, many small changes at the input can accumulate to a large change at the output. Adversarial attacks that affect one network can also affect others, and the development of more robust networks is an active area of research.

## Real-world adversarial attack

As these examples involve the addition of noise to an image, they may appear to be irrelevant to real-world applications. However, adversarial attacks have also been produced physically.

{{fig patch_attack}}

When black and white pieces of tape were placed at specific points on a stop sign, the sign was classified by a network as an 80 km/h speed-limit sign, and the attack was effective over a wide range of distances and viewpoints. The implications for safety-critical applications are clear, and similar attacks have been demonstrated against commercial self-driving software.

## Lack of generalisation

In contrast to an adversarial attack, which is designed to cause misclassification, errors are also made by a trained network simply as a result of a **failure to generalise**.

{{fig pose_generalisation}}

During training, overfitting is avoided so that the network generalises to the test set. However, the training and test sets are usually drawn from the same pool of images and share many characteristics. When the network is tested with images from outside that pool, unexpected behaviour can occur. It has been shown that networks are easily fooled by changes in the **pose** of objects that are otherwise recognised easily: a school bus, a scooter and a fire truck are recognised in poses likely to be found in the ImageNet training set, and are misclassified when the pose is varied.

This again shows that the deployment of a network in a safety-critical application can be dangerous. If a self-driving car failed to recognise a pedestrian lying in the road because the orientation of the pedestrian differed from that in the training data, an extremely dangerous error would result.

{{selfcheck 02_05}}
