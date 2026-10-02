---
id: "02_02"
title: Architectures for image classification
section: 2
engines: [s2_classification.architectures]
practice: [global_avg_pool, inception_block, conv_param1]
summary: AlexNet and its memory and computational cost, VGG and stacked 3×3 convolutions, the model zoo, network in network, global average pooling, and the Inception architecture with its bottlenecks and auxiliary classifiers.
---

## Learning outcomes

After this lesson you should be able to:

- describe **AlexNet** and say how it differs from LeNet-5;
- calculate the **memory and computation** of a network layer by layer, and explain why dense layers dominate one and convolution the other;
- explain why **VGG** uses stacks of 3 × 3 convolutions;
- read the **model zoo** trade-off between accuracy, computation and parameters;
- explain the **network-in-network** idea and **global average pooling**;
- describe the **Inception module**, why it needs **1 × 1 bottlenecks**, and the role of **auxiliary classifiers** in GoogLeNet.

## AlexNet

AlexNet was the first neural network to win the ImageNet challenge. Its win proved that, given large amounts of training data, a large neural network could learn features from images that surpassed all the hand-engineered computer vision approaches.

{{fig alexnet_diagram}}

Structurally it is similar to LeNet-5: 5 convolution, 3 pooling and 3 fully connected layers, with softmax on the last. Because it was trained on ImageNet, it outputs 1000 class probabilities. The key difference is **scale**. The input is much larger, so the first convolution uses large 11 × 11 kernels, and there are 96 of them rather than 6. Deeper in the network the number of feature maps rises to 384 while pooling reduces their spatial size — as the channels increase the spatial size decreases, which balances the number of parameters per layer — and the kernels get smaller.

The original AlexNet was split across two GPUs because of the memory limits of the hardware available to its authors; the form drawn here is the single-GPU equivalent. It also introduced several things that are now standard: **ReLU** activations, which avoid the saturation and vanishing gradients of sigmoid and tanh; **dropout** on the fully connected layers; and **data augmentation** during training.

## Memory and computation

The **memory** requirement is set by the number of parameters. For a convolution layer that is the sum of the parameters in all its kernels; for a fully connected layer it is the number of inputs multiplied by the number of neurons.

{{fig alexnet_cost}}

The convolution layers hold tens to hundreds of thousands of parameters each; the fully connected layers hold tens of millions. AlexNet has around 60 million parameters in total — at 16-bit precision about 120 MB, more than many mobile and embedded devices can hold. Worse, the computation rate a GPU can achieve on fully connected layers is limited by **memory bandwidth** rather than by its arithmetic speed, which slows training.

The **computation**, on the other hand, is dominated by the convolution layers. Stepping a kernel around the input multiplies every weight many times, so convolution is computationally expensive — but it is not bandwidth-limited and can use the GPU's full speed. Even so, training a complex network on a large dataset can take days on a single GPU.

{{example conv_param1 seed=8}}

## VGG networks

The VGG family, of which VGG-16 is the best known, follows the same overall plan as AlexNet but with a key difference in its convolutions.

{{fig vgg_stacking}}

Each block of VGG-16 is a cascade of **3 × 3** convolutions, giving 13 convolution layers in total rather than 5, with up to 512 channels. The authors observed that back-to-back small convolutions have the same receptive field as one larger convolution, but are **less computationally expensive** than AlexNet's 11 × 11 and 5 × 5 kernels.

VGG keeps the large fully connected layers at the end, and as with AlexNet they are the dominant source of parameters. The extra parameters can cause overfitting, but they also raise accuracy on ImageNet.

## The model zoo

{{fig model_zoo}}

Plotting accuracy against computation, with the parameter count as bubble size, shows the trade-offs at a glance. AlexNet is cheap and not very accurate, but has a large parameter count. VGG is more accurate but needs far more computation and has an extremely large parameter count, which makes it unsuitable for mobile devices. Both owe their parameter counts to their large fully connected output layers.

Those concerns pushed research towards architectures that **eliminate the large fully connected layers**. That meant putting more processing into the convolution layers, so that they learn representations good enough to separate the classes without a large classifier on top. The rest of this lesson follows that idea into the Inception architecture; the next lesson covers residual networks.

## Network in network

The network-in-network structure, proposed by Lin, approaches convolution differently.

{{fig network_in_network}}

The convolution kernel is replaced by a small **multilayer perceptron** — a small fully connected network — which is still applied across the input in a convolutional manner. The rationale is that convolution is a **linear** operation, so the representational power of the features it produces may be low. Those features are then hard to separate, which is what drives the need for large fully connected layers at the output of networks such as AlexNet and VGG. Adding small fully connected layers *inside* the convolution raises the representational power locally, and the classes become easier to discriminate.

The output layers can then be replaced by a single **global average pooling** layer. In the Inception architecture, the small fully connected layers are implemented as equivalent **1 × 1 convolutions**, which is much more efficient.

## Global average pooling

{{fig global_avg_pool}}

Global average pooling automatically pools a feature map of any size down to a single value, giving a 1D array with one element per feature map. It is typically followed by one dense layer with softmax to produce class probabilities.

Dense layers contribute greatly to the parameter count and are prone to overfitting, needing regularisation such as dropout to train properly. Global average pooling avoids both — but it should only replace them when the convolution layers can extract sufficiently abstract features, or the network is deep enough to do so.

{{example global_avg_pool seed=3}}

## Inception

The Inception architecture departs from the sequential convolution-and-pooling structure of AlexNet and VGG. It is based on a theoretical result by Arora and colleagues: a data distribution can be represented by a large **sparse** network. A sparse network has fewer connections, so fewer parameters and less computation. The networks we train are dense, but many learned parameters end up close to zero, and the computation spent on them is wasted.

Modern GPUs, however, are designed for **dense** computation and cannot easily exploit sparsity, even if the near-zero parameters could be found and removed. So the Inception authors looked for an architecture made of ordinary dense layers that would nonetheless approximate the ideal sparse network. Following Arora, they built it layer by layer, clustering correlated units into groups: a 1 × 1 convolution finds correlations across feature maps, while 3 × 3 and 5 × 5 convolutions find correlations over progressively wider regions.

{{fig inception_module}}

Each **Inception module** is a network within a network. Parallel convolutions with different kernel sizes, plus a pooling path, run side by side, and their outputs are **concatenated** into a single larger output. The largest kernel is limited to 5 × 5 because larger kernels greatly increase computation; the pooling path is included because pooling had worked so well elsewhere.

The naive version has a problem: deep in the network a 5 × 5 convolution may face hundreds of input feature maps and become very expensive. GoogLeNet therefore inserts **1 × 1 bottleneck** convolutions ahead of the 3 × 3 and 5 × 5 convolutions to reduce the computation — the same idea as in Lesson 1.3.

{{example inception_block seed=5}}

## GoogLeNet and auxiliary classifiers

{{fig googlenet}}

The Inception network submitted to the ImageNet challenge, **GoogLeNet**, is 22 layers deep — the deepest CNN known at the time. Global average pooling converts the last convolution output to the input of the final fully connected layer, with dropout during training. The network is both deep and wide, the width coming from the parallel paths.

GoogLeNet also used **auxiliary outputs**, used only during training. They help avoid vanishing gradients: with several outputs the network is effectively a superposition of networks of different depths, combined through a joint loss function. They are removed when the network is used for inference.

Overall GoogLeNet used far fewer parameters than AlexNet and performed better, and is much smaller than the VGG networks. Later versions of Inception added batch normalisation (next lesson), replaced the 5 × 5 convolutions with pairs of 3 × 3 convolutions to reduce computation, and used stride-2 convolutions inside the modules to reduce spatial size. They remain competitive with the best image classification architectures, at the cost of more computation.

{{selfcheck 02_02}}
