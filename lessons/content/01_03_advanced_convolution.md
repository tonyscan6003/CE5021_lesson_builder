---
id: "01_03"
title: Advanced convolution
section: 1
engines: [s1_convnets.advanced_convolution]
practice: [mult_kernel_conv, conv_param1, conv_param2, bottleneck_conv]
summary: The convolution operations found in real CNNs — multiple input channels, multiple parallel kernels, the volume notation, the number of parameters and multiplications, and the 1×1 bottleneck.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- perform a convolution with a **multi-channel** input and a multi-channel kernel;
- explain how the number of output feature maps is set by **multiple parallel kernels**;
- interpret a **volume** diagram and determine the shapes and computational cost it represents;
- calculate the **number of parameters** and the **number of multiplications** in a convolution layer;
- explain the operation of a **$1 \times 1$ convolution** and why no padding is required;
- calculate the computational saving from a **bottleneck** convolution.

## Introduction

Convolutional layers are located at the front of a deep network and perform its key function, the detection of image features, both simple and complex. In a modern CNN the convolution operations are more complex than the 2D convolution described in Lesson 1.1. In this lesson, convolutions with multiple input channels and multiple parallel kernels are described, the calculation of their memory and computational requirements is covered, and $1 \times 1$ convolutions and bottlenecks are introduced.

## Multiple channels

In Lesson 1.1 a two-dimensional single-channel input was convolved with a single kernel. In a modern CNN the input is usually a colour RGB image with 3 channels, and deeper in the network a feature map may have hundreds of channels.

To convolve such an input, **the kernel must have the same number of channels as the input**. Each channel of the kernel is convolved with the corresponding channel of the input, and the results are summed to produce **one** output channel.

{{fig multi_channel_conv}}

The output of a convolution in a CNN is usually called a **feature map**, as it shows where the features to which the kernel responds are located in the input.

The spatial size of the output is calculated with the same arithmetic as in Lesson 1.1. The 2D convolution between one input channel and one kernel channel is considered, and the relationships from the table in Lesson 1.1 are applied. The number of channels has no effect on the spatial calculation.

{{example mult_kernel_conv seed=5}}

## Multiple kernels

**Parallel kernels** are used in deep CNNs to increase the number of parameters. This allows many complex image features to be learned and improves the ability of the network to classify or detect objects in real images.

{{fig multi_kernel_conv}}

The relationship between the shapes is summarised as follows:

- the input has $C$ channels, so **every kernel must have $C$ channels**;
- each kernel is convolved with the whole input as described above, producing **one** 2D feature map;
- with $N$ kernels there are therefore **$N$ parallel feature maps** in the output.

The depth of the kernel is therefore set by the number of input channels, and the depth of the output is set by the number of kernels. The output spatial size is determined by the same relationships as for 2D convolution, and valid, same or full padding may be used as required.

## Volumes

When a convolution involves hundreds of feature maps, it is impractical to draw each one. In architecture diagrams a compact representation is used instead, in which a stack of feature maps is drawn as a single **volume**.

The orientation used throughout this module is the standard one in the literature: the **horizontal axis is the channel axis**, and the two spatial dimensions form the square **end cap**. A volume with more channels is therefore drawn as a *longer* box, and a volume with a larger spatial size has a *larger end cap*. In a typical network the volumes become longer and their end caps become smaller with depth.

The convolution is usually indicated by a dashed cube drawn **inside** the input volume. The cube extends over the full length of the volume, as a kernel must span every input channel, while covering only a $k \times k$ region spatially.

{{fig conv_volumes}}

No information is lost in this representation, as the computational details can be inferred from a volume diagram. Kernels of size $k \times k$ operating on $C$ input channels must have $C$ channels, and an output volume with $N$ channels indicates that there are $N$ individual kernels. The layer therefore contains $N$ kernels of size $k \times k \times C$.

## Parameters and multiplications

In a 2D convolution with a $3 \times 3$ kernel there are 9 parameters. In a more complex convolution there are multiple kernels and multiple channels, and the number of parameters is greatly increased. It can usually be determined by inspection:

$$\text{parameters} = N \cdot k \cdot k \cdot C + N, \tag{1}$$

where the final $N$ corresponds to one bias parameter per output feature map.

The number of **multiplications** is also of interest, as multiplication is more computationally expensive than addition. This depends on the number of times the kernels are applied. Rather than working back through the stride and padding, a shortcut can be used: **each spatial position in the output corresponds to one position of the kernel in the input**, so the number of kernel positions is $m' \times n'$. Therefore

$$\text{multiplications} = (N \cdot k \cdot k \cdot C) \cdot m' \cdot n'. \tag{2}$$

The bias is excluded, as it is added rather than multiplied.

{{fig conv_cost}}

Convolution can involve a very large number of multiplications, but it remains computationally efficient, as the same parameters are reused at every position. It should also be noted that the cost increases with $k^2$, which is why small kernels are preferred in CNNs.

{{example conv_param1 seed=6}}

{{example conv_param2 seed=6}}

## $1 \times 1$ convolution

The kernel can be reduced to a minimum spatial size of $1 \times 1$, which gives the smallest possible number of parameters. A $1 \times 1$ convolution is computed in the same way as any other convolution. The kernel still spans the full depth of the input, so each kernel is $1 \times 1 \times C$, and a weighted sum **across channels** is computed independently at every pixel.

{{fig one_by_one_conv}}

The key properties of the $1 \times 1$ convolution are that no padding is required, the spatial size is unchanged, and the number of parameters is the smallest of any kernel size. The number of channels, however, can be changed, and this is why the $1 \times 1$ convolution is used as a bottleneck.

## Bottleneck convolution

In a **bottleneck**, a $1 \times 1$ convolution is inserted before a larger convolution, in order to reduce the number of channels processed by the more expensive kernels.

{{fig bottleneck_conv}}

As an example, a $32 \times 32$ input with $C = 24$ channels is convolved with $N = 12$ kernels of size $3 \times 3$, with same padding and stride 1, so the output is also $32 \times 32$. The number of multiplications is

$$N \cdot k \cdot k \cdot C \cdot m' \cdot n' = 12 \cdot 3 \cdot 3 \cdot 24 \cdot 32 \cdot 32 = 2{,}654{,}208.$$

A $1 \times 1$ convolution is now inserted to reduce the number of channels from $C = 24$ to $P = 12$. All of the feature maps have the same spatial size, as the $1 \times 1$ convolution does not change the spatial size and same padding is used in the $3 \times 3$ stage. The two stages are considered separately:

- $1 \times 1$ stage: $C \cdot 1 \cdot 1 \cdot P = 24 \cdot 12 = 288$ parameters, so $288 \cdot 32 \cdot 32 = 294{,}912$ multiplications;
- $3 \times 3$ stage: $P \cdot 3 \cdot 3 \cdot N = 12 \cdot 9 \cdot 12 = 1{,}296$ parameters, so $1{,}296 \cdot 32 \cdot 32 = 1{,}327{,}104$ multiplications.

The total is $1{,}622{,}016$ multiplications, a saving of approximately 39%. The overall computation is reduced because the inexpensive $1 \times 1$ convolution is used to reduce the number of channels processed by the more expensive $3 \times 3$ convolution.

A saving is only obtained when $P$ is sufficiently smaller than $C$. In the interactive version of the figure above, $P$ is varied, and the value at which the bottleneck no longer reduces the computation can be seen.

{{example bottleneck_conv seed=8}}

Bottlenecks are used in many of the architectures described in Section 2, including the Inception block and the deeper variants of ResNet, where depth is required without a corresponding increase in computational cost.

{{selfcheck 01_03}}
