---
id: "02_02"
title: Architectures for image classification
section: 2
engines: [s2_classification.architectures]
practice: [global_avg_pool, inception_block, conv_param1]
summary: AlexNet and its memory and computational cost, VGG and stacked 3×3 convolutions, the model zoo, network in network, global average pooling, and the Inception architecture with its bottlenecks and auxiliary classifiers.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- describe **AlexNet** and its differences from LeNet-5;
- calculate the **memory and computation** of a network layer by layer, and explain why memory is dominated by dense layers and computation by convolution;
- explain why stacks of 3 × 3 convolutions are used in **VGG**;
- interpret the **model zoo** trade-off between accuracy, computation and parameters;
- explain the **network-in-network** concept and **global average pooling**;
- describe the **Inception module**, the need for **1 × 1 bottlenecks**, and the role of **auxiliary classifiers** in GoogLeNet.

## AlexNet

AlexNet was the first neural network to win the ImageNet challenge. Its success demonstrated that, given large amounts of training data, features could be learned from images by a large neural network that surpassed all of the hand-engineered computer vision approaches.

{{fig alexnet_diagram}}

The structure is similar to LeNet-5, with 5 convolution, 3 pooling and 3 fully connected layers, and softmax on the last layer. As it was trained on ImageNet, 1000 class probabilities are output. The key difference is **scale**. The input is much larger, so large 11 × 11 kernels are used in the first convolution, and there are 96 kernels rather than 6. Deeper in the network the number of feature maps increases to 384 while their spatial size is reduced by pooling, and the kernels become smaller. As the number of channels increases the spatial size decreases, which balances the number of parameters per layer.

The original AlexNet was split across two GPUs because of the memory limits of the hardware available at the time; the single-GPU equivalent is shown here. Several features that are now standard were also introduced: **ReLU** activations, which avoid the saturation and vanishing gradients of sigmoid and tanh; **dropout** on the fully connected layers; and **data augmentation** during training.

## Memory and computation

The **memory** requirement is determined by the number of parameters. For a convolution layer this is the sum of the parameters in all of its kernels, and for a fully connected layer it is the number of inputs multiplied by the number of neurons.

{{fig alexnet_cost}}

The convolution layers contain tens to hundreds of thousands of parameters each, while the fully connected layers contain tens of millions. AlexNet has approximately 60 million parameters in total, which at 16-bit precision is approximately 120 MB, more than can be held by many mobile and embedded devices. In addition, the computation rate that can be achieved by a GPU on fully connected layers is limited by **memory bandwidth** rather than by arithmetic speed, which slows training.

The **computation**, however, is dominated by the convolution layers. As the kernel is stepped across the input, every weight is used in many multiplications, so convolution is computationally expensive. However, convolution is not limited by bandwidth, and the full speed of the GPU can be used. Even so, the training of a complex network on a large dataset can take days on a single GPU.

{{example conv_param1 seed=8}}

## VGG networks

The VGG family, of which VGG-16 is the best known, follows the same overall structure as AlexNet, with a key difference in the convolutions.

{{fig vgg_stacking}}

Each block of VGG-16 is a cascade of **3 × 3** convolutions, giving 13 convolution layers in total rather than 5, with up to 512 channels. It was observed by the authors that consecutive small convolutions have the same receptive field as a single larger convolution, but are **less computationally expensive** than the 11 × 11 and 5 × 5 kernels of AlexNet.

The large fully connected layers are retained at the end of VGG, and as with AlexNet they are the dominant source of parameters. The additional parameters can cause overfitting, but they also increase accuracy on ImageNet.

## The model zoo

{{fig model_zoo}}

When accuracy is plotted against computation, with the number of parameters shown as the bubble size, the trade-offs between architectures can be compared. AlexNet requires little computation and is not very accurate, but has a large number of parameters. VGG is more accurate but requires far more computation and has an extremely large number of parameters, which makes it unsuitable for mobile devices. In both cases the large number of parameters is due to the large fully connected output layers.

These concerns led to research into architectures in which **the large fully connected layers are eliminated**. For this, more processing must be performed in the convolution layers, so that representations are learned that can separate the classes without a large classifier at the output. In the remainder of this lesson this approach is followed through to the Inception architecture; residual networks are described in the next lesson.

## Network in network

In the network-in-network structure, proposed by Lin, a different approach to convolution is taken.

{{fig network_in_network}}

The convolution kernel is replaced by a small **multilayer perceptron**, a small fully connected network, which is still applied across the input in a convolutional manner. The rationale is that convolution is a **linear** operation, so the representational power of the features produced may be low. Such features are difficult to separate, and this leads to the need for large fully connected layers at the output of networks such as AlexNet and VGG. When small fully connected layers are added *inside* the convolution, the representational power is increased locally, and the classes are more easily discriminated.

The output layers can then be replaced by a single **global average pooling** layer. In the Inception architecture, the small fully connected layers are implemented as equivalent **1 × 1 convolutions**, which is much more efficient.

## Global average pooling

{{fig global_avg_pool}}

In global average pooling, a feature map of any size is pooled to a single value, giving a 1D array with one element per feature map. This is typically followed by one dense layer with softmax to produce the class probabilities.

Dense layers contribute greatly to the number of parameters and are prone to overfitting, and regularisation such as dropout is required to train them properly. Both of these problems are avoided by global average pooling. However, dense layers should only be replaced in this way when sufficiently abstract features can be extracted by the convolution layers, or when the network is deep enough for this to be possible.

{{example global_avg_pool seed=3}}

## Inception

The Inception architecture departs from the sequential convolution-and-pooling structure of AlexNet and VGG. It is based on a theoretical result by Arora and colleagues, in which it was shown that a data distribution can be represented by a large **sparse** network. A sparse network has fewer connections, and therefore fewer parameters and less computation. The networks that are trained in practice are dense, but many of the learned parameters are close to zero, and the computation performed with them is wasted.

Modern GPUs, however, are designed for **dense** computation, and sparsity cannot easily be exploited, even if the near-zero parameters could be identified and removed. The Inception authors therefore sought an architecture composed of ordinary dense layers that would approximate the ideal sparse network. Following Arora, the network was constructed layer by layer, with correlated units clustered into groups: correlations across feature maps are found by a 1 × 1 convolution, while correlations over progressively wider regions are found by 3 × 3 and 5 × 5 convolutions.

{{fig inception_module}}

Each **Inception module** is a network within a network. Parallel convolutions with different kernel sizes, together with a pooling path, are applied side by side, and their outputs are **concatenated** into a single larger output. The largest kernel is limited to 5 × 5, as larger kernels greatly increase the computation, and the pooling path is included because pooling had been shown to be effective in other networks.

In the naive version of the module, a 5 × 5 convolution deep in the network may be applied to hundreds of input feature maps, which is very expensive. In GoogLeNet, **1 × 1 bottleneck** convolutions are therefore inserted before the 3 × 3 and 5 × 5 convolutions to reduce the computation, as described in Lesson 1.3.

{{example inception_block seed=5}}

## GoogLeNet and auxiliary classifiers

{{fig googlenet}}

The Inception network submitted to the ImageNet challenge, **GoogLeNet**, is 22 layers deep, and was the deepest CNN known at the time. Global average pooling is used to convert the output of the last convolution to the input of the final fully connected layer, with dropout applied during training. The network is both deep and wide, with the width provided by the parallel paths.

**Auxiliary outputs** were also used in GoogLeNet during training only. They help to avoid vanishing gradients: with several outputs, the network is effectively a superposition of networks of different depths, which are combined through a joint loss function. The auxiliary outputs are removed when the network is used for inference.

Overall, far fewer parameters were used in GoogLeNet than in AlexNet, with better performance, and the network is much smaller than the VGG networks. In later versions of Inception, batch normalisation (described in the next lesson) was added, the 5 × 5 convolutions were replaced by pairs of 3 × 3 convolutions to reduce computation, and stride-2 convolutions were used inside the modules to reduce the spatial size. These versions remain competitive with the best image classification architectures, at the cost of additional computation.

{{selfcheck 02_02}}
