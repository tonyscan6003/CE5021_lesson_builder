---
id: "02_05"
title: Feature visualisation and adversarial attacks
section: 2
engines: [s2_classification.visualisation]
practice: []
summary: Four techniques for seeing what a network has learned — reconstruction, activation maximisation, style transfer and deep dream — and the limits of trained networks — adversarial attacks and failures of generalisation.
---

## Learning outcomes

After this lesson you should be able to:

- explain why visualising learned features matters for **explainable** deployment;
- describe **reconstruction** (inversion) and what it shows about depth;
- describe **activation maximisation** and why it uses gradient **ascent** with a regulariser;
- explain how **style** is captured by **Gram matrices** and how **style transfer** combines style and content losses;
- describe **deep dream**;
- explain what an **adversarial example** is, how the **fast gradient sign method** produces one, and why physical attacks matter;
- explain the **limits of generalisation** of trained networks.

## Introduction

So far we have treated networks as image classifiers and noted that their convolution layers learn features during training. But it is not at all transparent what a network has learned, or what features it responds to when it classifies a test image. Investigating learned features makes networks more **explainable**, which helps when deploying them in real applications.

The first part of this lesson covers four techniques for understanding what a network has learned. All four share one idea, and it is worth holding on to: they keep the network **fixed** and use gradient-based optimisation to change the **input image**.

A note on the figures. To run these techniques for real without a deep learning framework, the lesson uses a small **fixed** network — a bank of oriented Gabor filters followed by ReLU and pooling, where "deeper" means more pooling. Gabor filters are what the first layer of a trained CNN typically learns, so the network is a fair stand-in for the *early* layers, and it lets every algorithm run exactly as described. It has not learned anything, though, so it shows how each technique works and how depth changes what a layer retains — not what a trained network's deep layers know. Where a figure uses a genuinely trained model, the caption says so.

## Reconstruction

Reconstruction, also called **inversion**, asks what information about an image is retained by a network's feature maps.

{{fig reconstruction}}

The target image is input to the network and the features at the layer of interest are stored. An image of random noise is then input, and its features at the same layer are compared with the stored ones in a loss function. Gradient descent updates the noise image on each iteration so that the loss approaches zero and the noise converges to an image resembling the target. Regularisation terms are added to the loss to help produce natural-looking images.

Reconstruction from early layers is very accurate: they capture many low-level features. From deeper layers it is less accurate, and eventually only an abstract composition of parts of the image can be recovered. In a trained network this is consistent with the deep layers learning **concepts** about objects from large numbers of training images.

## Activation maximisation

Activation maximisation visualises the features the convolution layers have learned. We choose a neuron, channel or layer, and find the input image that **maximises** its output: the feature that neuron would respond to most strongly if it appeared in a real image.

{{fig activation_maximisation}}

Starting from random noise, we monitor the neuron's value and use **gradient ascent** — not descent, because the quantity we want is to be *maximised* — to update the pixels. A **regularisation penalty** is needed, because many unnatural noise patterns would also maximise the neuron.

For VGG-16, maximising one point of a feature map in each layer shows that the images grow larger in deeper layers, because the **receptive field** grows. Early layers respond to colour blobs and edges, the next to complex patterns that may be textures, and the deeper layers to objects or parts of objects.

## Visualisation for understanding

Visualising a network also shows what it looks for in images of a given class.

{{fig class_templates}}

Comparing images produced by activation maximisation with the training images a neuron responds to most strongly reveals that a network may recognise objects by properties that are merely common to the training images — the fluffiness of clouds, the presence of sky behind buildings. In many cases the network is not really recognising the object itself but relying on features that **co-occur** with it, which can cause confusion between classes once the network is deployed.

## Style transfer

Style transfer moves the style of one image, usually an artwork, onto the content of another. It is very difficult with traditional computer vision but possible with a pre-trained network, because of what the network has learned.

As reconstruction showed, early layers hold enough information to reconstruct an image accurately, while deeper layers hold information about objects and their relationships — **content** — but not fine detail. A network also retains **style**, or texture, information, which is obtained from the inner products of the feature maps: the **Gram matrix**.

{{fig texture_statistics}}

Texture appears at different scales, small from early layers and large from deeper ones.

{{fig style_transfer_loop}}

To create the style-transferred image:

1. pass the **content** image through the pre-trained network and store a deep feature map (conv4 in the original work);
2. pass the **style** image through and store the texture information — the Gram matrices — of all the convolution layers;
3. pass a **random noise** image through, and compute its features and Gram matrices;
4. compare them with the stored content and style information in the loss function, and use gradient descent to adjust the noise image's pixels to minimise both losses together.

The noise image gradually converges to an image with the content of the content image in the style of the style image. Style transfer shows not only that a network holds both content and style information, but that the two can be **recombined** into new images.

## Deep dream

Deep dream is similar to activation maximisation, but a **content image** is applied to the network instead of random noise.

{{fig deep_dream}}

The content image makes the network respond to features already in it, and gradient ascent then **amplifies** them. Maximising the outputs of many layers at once tends to amplify several features together. Particular content biases the network towards particular features: images with clear horizon lines produce towers and pagodas, trees produce features of buildings, leaves produce birds and insects — revealing associations the network has learned between image features and concepts. Applied iteratively, zooming in on the result between iterations, deep dream produced a series of abstract images that were among the earliest examples of AI-generated art.

## Adversarial attack

We have seen how well trained networks can perform. Before deploying one, however — especially in a safety-critical application — we need to understand how **robust** it is.

**Adversarial examples** trick a network into misclassifying an input by manipulating it. In the well-known example, an image of a panda needs only a very small amount of a noisy pattern added for the network to output a completely different label — with *higher* confidence than its original prediction — while to a human observer the image is unchanged.

{{fig adversarial_fgsm}}

The added pattern forces the wrong output neuron to be activated. Because of the network's internal structure — many multiplications and non-linear activation stages — many small changes at the input can add up to a large change at the output. Adversarial attacks that affect one network can also affect others, and developing more robust networks is an active area of research.

## Real-world adversarial attack

It is tempting to dismiss these examples as irrelevant to the real world, since they involve adding noise to an image. But adversarial attacks have been produced physically.

{{fig patch_attack}}

By placing black and white pieces of tape at specific points on a stop sign, researchers forced a network to classify it as an 80 km/h speed-limit sign — and the attack worked over a broad range of distances and viewpoints. The implications for safety-critical applications are obvious, and similar attacks have been demonstrated against commercial self-driving software.

## Lack of generalisation

In contrast to an adversarial attack, which deliberately sets out to cause misclassification, a trained network will also make errors simply because it **fails to generalise**.

{{fig pose_generalisation}}

During training we avoid overfitting so that the network generalises to the test set — but the training and test sets are usually drawn from the same pool of images and share many characteristics. Test the network with images from outside that pool and unexpected behaviour appears. Networks have been shown to be easily fooled by changes in the **pose** of objects they otherwise recognise easily: a school bus, a scooter and a fire truck are recognised in the poses likely to be in the ImageNet training set, and misclassified once the pose varies.

Once again this shows how deploying a network in a safety-critical application can be dangerous. A self-driving car that failed to recognise a pedestrian lying in the road, because the pedestrian's orientation differed from its training data, would be making an extremely dangerous error.

{{selfcheck 02_05}}
