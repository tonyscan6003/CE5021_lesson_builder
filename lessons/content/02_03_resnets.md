---
id: "02_03"
title: Residual networks
section: 2
engines: [s2_classification.resnets]
practice: [bat_norm1, bat_norm2]
summary: The difficulty of training very deep plain networks, residual learning and the residual block, ResNet as an ensemble of paths, the ResNet family, batch normalisation, squeeze-and-excite and DenseNet.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain the **degradation problem** and why it is not overfitting;
- describe **residual learning** and draw a **residual block**;
- explain the **unravelled view** of a ResNet as an ensemble of paths;
- describe the structure of the **ResNet family**, including the **bottleneck block**;
- carry out **batch normalisation** in training and at inference, and explain its benefits;
- describe the **squeeze-and-excite** block and **DenseNet**.

## Depth and training error

ResNet is one of the most important advances in neural network design since AlexNet, as it allowed very deep networks, eventually of more than 1000 layers, to be trained.

The work began with an experiment. VGG-19 was extended with additional layers, and it was found that the deeper network had a **higher training error** than the shallower network for the same number of training iterations. Very deep plain networks converged so slowly that they were impractical to train.

{{fig degradation}}

It should be noted that this is a failure on the *training* set. A deeper network has more capacity, as it could in principle reproduce the shallow network with the additional layers performing no operation, so the problem is not overfitting. Instead, the deeper network **cannot be optimised** effectively, and this is the problem addressed by residual connections.

## Residual learning

{{fig residual_block}}

The basic residual block has a main path of convolution and activation layers, and a shortcut from the input, which is **added** to the main path before the final activation.

When a network without shortcuts is trained, a block can be considered to converge to a desired mapping $H(x)$. With residual learning, the main path instead learns

$$F(x) = H(x) - x, \tag{1}$$

and the original mapping is recovered as $H(x) = F(x) + x$, with $x$ supplied by the shortcut. These residual mappings have been found to be much easier to train than the original mappings.

## The ensemble of paths

An explanation was proposed by Veit and colleagues. When a ResNet is unravelled, it can be seen that each block offers two routes, either through its layers or around them by the shortcut, so the network is an **ensemble** of many paths of different lengths.

{{fig unravelled_paths}}

The ensemble view explains the observation that a residual block can be removed from a trained ResNet without a large effect on its performance, whereas in a plain network this would cause the network to fail.

Many of the paths through the ensemble are short. During training it is observed that the **larger-magnitude gradients are carried by the short paths**, and these determine the weight updates. It was concluded that deep networks are enabled by residual connections because the effective paths are shortened and the vanishing gradient problem is avoided.

This explanation is in part contradicted by the original ResNet authors, who considered it unlikely that the difficulties of deep plain networks were caused by vanishing gradients. The exact reason that training is improved by shortcut connections remains an open research question, but their effectiveness is well established.

## The ResNet family

{{fig resnet_family}}

A complete ResNet is a stack of many residual blocks with a regular, repeating structure. After an initial 7 × 7 convolution and max pooling, each depth of ResNet is constructed from repeating units arranged in four stages.

Blocks of two 3 × 3 convolutions are used in ResNet-18 and -34. **Bottleneck** blocks are used in ResNet-50, -101 and -152, in which the number of channels is reduced and then restored by 1 × 1 convolutions around the 3 × 3 convolution, in order to reduce computation, as described in Lesson 1.3.

## Batch normalisation

**Batch normalisation** layers are also used in ResNet to improve training speed. Batch normalisation is now used in most modern networks, and it also has a regularising effect. In the original ResNet a batch normalisation layer follows each convolution.

{{fig batch_norm}}

For each channel, the mean and variance of the input are calculated over the mini-batch:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2. \tag{2}$$

The data is normalised, and then scaled and shifted by two learned parameters, $\gamma$ and $\beta$:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta. \tag{3}$$

The parameters $\gamma$ and $\beta$ are learned independently for each batch normalisation layer, and as only these two parameters are learned per channel, very few parameters are added to the network.

In the original batch normalisation paper, the benefit was described as a reduction in **internal covariate shift**, that is, the changes at the input of each layer caused by weight updates in the preceding layers. With normalisation, the input statistics remain more consistent during training, and convergence is improved.

As with residual connections, the reason for the effectiveness of batch normalisation is still an open question, and the covariate-shift explanation has been disputed in later work. It has been shown experimentally that networks without batch normalisation can only be trained with low learning rates and converge slowly, whereas much higher learning rates can be used with batch normalisation. The same work showed that without batch normalisation the internal states of deep layers **explode**: in a 110-layer ResNet, the variance of one channel in layer 80 reached $10^{16}$. With batch normalisation, the activations of each channel are corrected and are prevented from accumulating through the network, as reproduced in panel (b) of the figure.

{{example bat_norm1 seed=4}}

## Batch normalisation at inference

{{fig batch_norm_running}}

At inference there may be only a single image, so there is no batch from which statistics can be calculated. During training, a **running** (global) mean and variance are therefore also accumulated, with a decay parameter $d$:

$$\mu \leftarrow d\,\mu + (1 - d)\,\mu_B, \qquad \sigma^2 \leftarrow d\,\sigma^2 + (1 - d)\,\sigma_B^2. \tag{4}$$

At inference, $\mu_B$ and $\sigma_B^2$ in equation (3) are replaced by these stored values.

{{example bat_norm2 seed=4}}

## Squeeze-and-excite networks

Squeeze-and-excite (SE) networks are created by the addition of SE blocks to Inception or ResNet. In a ResNet, the SE block is inserted between the last convolution of the main path and the shortcut addition.

{{fig squeeze_excite}}

The input is a volume of height × width with C channels, which is reduced to 1 × 1 × C by global average pooling. This is passed through a cascade of fully connected layers with activations, the last of which is a sigmoid, so that the values are limited to between 0 and 1. These values are then used as scale factors, one for each channel of the input. In this way **interdependencies between channels** can be modelled and exploited. The block is computationally lightweight, and when it is added to ResNet-50 the ImageNet error is reduced by approximately one tenth.

## DenseNet

In DenseNet the concept of ResNet is extended with greatly increased connectivity.

{{fig densenet}}

As in ResNet, DenseNet begins with a large-kernel convolution and max pooling. The remainder of the network is a series of **dense blocks** separated by simple **transition** blocks of 1 × 1 convolution and average pooling, and the network ends with global average pooling and softmax.

Inside a dense block, each composite function (batch normalisation, ReLU and convolution, in that pre-activation order) is similar to a ResNet layer, except that the feature maps are **concatenated** rather than the shortcut being **summed** with the main path. The outputs of all previous stages, including the input of the block, are therefore received by every composite function, which results in dense connectivity. If the input has $k_0$ channels and $k$ new feature maps are added by each function, there are $k_0 + nk$ channels after $n$ functions.

The benefit is similar to that of ResNet: there are many short paths, and problems with exploding or vanishing gradients are avoided. Despite the much greater connectivity, a DenseNet of equivalent performance has many fewer parameters, and can be considered a less redundant representation of the data. DenseNet is **narrow but deep**: each 3 × 3 convolution typically has only $k = 32$ kernels, compared with 64 to 256 channels in the blocks of ResNet, and the computation is again reduced by 1 × 1 bottleneck layers.

{{selfcheck 02_03}}
