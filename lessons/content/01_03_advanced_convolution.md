---
id: "01_03"
title: Advanced convolution
section: 1
engines: [s1_convnets.advanced_convolution]
practice: [mult_kernel_conv, conv_param1, conv_param2, bottleneck_conv]
summary: The convolution operations found in real CNNs — multiple input channels, multiple parallel kernels, the volume notation, counting parameters and multiplications, and the 1×1 bottleneck.
---

## Learning outcomes

After this lesson you should be able to:

- perform a convolution with a **multi-channel** input and a multi-channel kernel;
- explain how **multiple parallel kernels** set the number of output feature maps;
- read a **volume** diagram and recover the shapes and cost it implies;
- calculate the **number of parameters** and the **number of multiplications** in a convolution layer;
- explain what a **$1 \times 1$ convolution** does and why it needs no padding;
- calculate the computational saving from a **bottleneck** convolution.

## Introduction

Convolutional layers sit at the front of a deep network and perform its key function: detecting features, both simple and complex. In a modern CNN the convolution operations are more complex than the plain 2D convolution of Lesson 1.1. We will meet convolutions with multiple input channels and multiple kernels in parallel, learn to calculate their memory and computational requirements, and look at $1 \times 1$ convolutions and bottlenecks.

## Multiple channels

Lesson 1.1 used a two-dimensional single-channel input with a single kernel. In a modern CNN the input is usually a colour RGB image with 3 channels, and deeper in the network a feature map may have hundreds.

To convolve such an input, **the kernel must have the same number of channels as the input**. Each channel of the kernel is convolved with the corresponding channel of the input, and the results are summed to produce **one** output channel.

{{fig multi_channel_conv}}

The output of a convolution in a CNN is usually called a **feature map**, because it shows where in the input the features that the kernel responds to are located.

Note that the spatial size of the output follows exactly the same arithmetic as in Lesson 1.1 — consider the 2D convolution between one input channel and one kernel channel, and apply the relationships from the table there. The channel count plays no part in the spatial calculation.

{{example mult_kernel_conv seed=5}}

## Multiple kernels

Deep CNNs use **parallel kernels** to increase the number of parameters, which allows many complex image features to be learned and improves the network's ability to classify or detect objects in real images.

{{fig multi_kernel_conv}}

The pattern is worth stating precisely, because it is the source of most confusion about CNN shapes:

- the input has $C$ channels, so **every kernel must have $C$ channels**;
- each kernel is convolved with the whole input as described above, producing **one** 2D feature map;
- with $N$ kernels there are therefore **$N$ parallel feature maps** in the output.

So the input's channel count sets the kernel's depth, and the number of kernels sets the output's depth. The output spatial size is governed by the same relationships as plain 2D convolution, and the padding may be valid, same or full as required.

## Volumes

When a convolution involves hundreds of feature maps, drawing each one is impractical. Architecture diagrams instead use a compact representation in which a whole stack of feature maps is a single **volume**.

Note the orientation, which is the standard one in the literature and the one used throughout this module: the **horizontal axis is the channel axis**, and the two spatial dimensions form the square **end cap**. A volume with more channels is therefore a *longer* box, and one with a larger spatial size has a *bigger end cap*. As you go deeper into a typical network the volumes get longer and their end caps shrink.

The convolution itself is usually marked by a dashed cube drawn **inside** the input volume. That cube runs the full length of the volume — because a kernel must span every input channel — while covering only a $k \times k$ patch spatially.

{{fig conv_volumes}}

Nothing is lost by this. Looking at a volume diagram, we can infer the computational details: kernels of size $k \times k$ operating on $C$ input channels must themselves have $C$ channels, and an output volume with $N$ channels means there must be $N$ individual kernels. So the layer holds $N$ kernels of size $k \times k \times C$.

## Parameters and multiplications

In a simple 2D convolution with a $3 \times 3$ kernel there are 9 parameters. In a complex convolution there are multiple kernels and multiple channels, which greatly increases the count. Usually we can determine it by inspection:

$$\text{parameters} = N \cdot k \cdot k \cdot C + N, \tag{1}$$

the final $N$ being one bias parameter per output feature map.

We also want the number of **multiplications**, since multiplication is more computationally expensive than addition. This depends on how many times the kernels are applied. Rather than working back through the stride and the padding, there is a shortcut: **each spatial position in the output corresponds to one position the kernel occupied in the input**, so the number of kernel positions is just $m' \times n'$. Therefore

$$\text{multiplications} = (N \cdot k \cdot k \cdot C) \cdot m' \cdot n'. \tag{2}$$

The bias is excluded, since it is added rather than multiplied.

{{fig conv_cost}}

Convolution can involve a very large number of multiplications, but it remains computationally efficient because the same parameters are reused at every position. Note also that the cost grows with $k^2$: this is why small kernels are preferred in CNNs.

{{example conv_param1 seed=6}}

{{example conv_param2 seed=6}}

## $1 \times 1$ convolution

The kernel can be reduced to a minimum spatial size of $1 \times 1$, which gives the smallest possible number of parameters. A $1 \times 1$ convolution is computed exactly like any other: the kernel still spans the full depth of the input, so each kernel is $1 \times 1 \times C$ and computes a weighted sum **across channels** at every pixel independently.

{{fig one_by_one_conv}}

Its key properties are that it requires no padding and leaves the spatial size unchanged, and that it has the smallest parameter count of any kernel. What it *does* change is the number of channels — and that is what makes it useful as a bottleneck.

## Bottleneck convolution

A **bottleneck** inserts a $1 \times 1$ convolution before a larger convolution in order to reduce the channel count that the expensive kernels have to process.

{{fig bottleneck_conv}}

Worked through: a $32 \times 32$ input with $C = 24$ channels convolved with $N = 12$ kernels of $3 \times 3$, using same padding and stride 1, so the output is also $32 \times 32$. The multiplications are

$$N \cdot k \cdot k \cdot C \cdot m' \cdot n' = 12 \cdot 3 \cdot 3 \cdot 24 \cdot 32 \cdot 32 = 2{,}654{,}208.$$

Now insert a $1 \times 1$ convolution that reduces the channels from $C = 24$ to $P = 12$. All the feature maps keep the same spatial size, since $1 \times 1$ convolution does not change it and the $3 \times 3$ stage uses same padding. Taking the two stages separately:

- $1 \times 1$ stage: $C \cdot 1 \cdot 1 \cdot P = 24 \cdot 12 = 288$ parameters, so $288 \cdot 32 \cdot 32 = 294{,}912$ multiplications;
- $3 \times 3$ stage: $P \cdot 3 \cdot 3 \cdot N = 12 \cdot 9 \cdot 12 = 1{,}296$ parameters, so $1{,}296 \cdot 32 \cdot 32 = 1{,}327{,}104$ multiplications.

The total is $1{,}622{,}016$ multiplications, a saving of about 39%. Using the cheap $1 \times 1$ convolution to reduce the number of channels seen by the more expensive $3 \times 3$ convolution reduces the overall computation.

The saving is not automatic — it depends on $P$ being meaningfully smaller than $C$. The interactive version of the figure above sweeps $P$ so you can see where the bottleneck stops paying for itself.

{{example bottleneck_conv seed=8}}

Bottlenecks appear throughout the architectures in Section 2: in the Inception block, in ResNet's deeper variants, and wherever a network needs depth without the computational cost that depth would otherwise imply.

{{selfcheck 01_03}}
