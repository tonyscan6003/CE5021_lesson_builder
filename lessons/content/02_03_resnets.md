---
id: "02_03"
title: Residual networks
section: 2
engines: [s2_classification.resnets]
practice: [bat_norm1, bat_norm2]
summary: Why very deep plain networks fail to train, residual learning and the residual block, ResNet as an ensemble of paths, the ResNet family, batch normalisation, squeeze-and-excite and DenseNet.
---

## Learning outcomes

After this lesson you should be able to:

- explain the **degradation problem** and why it is not overfitting;
- describe **residual learning** and draw a **residual block**;
- explain the **unravelled view** of a ResNet as an ensemble of paths;
- describe the structure of the **ResNet family**, including the **bottleneck block**;
- carry out **batch normalisation** in training and at inference, and explain why it helps;
- describe the **squeeze-and-excite** block and **DenseNet**.

## Why deeper is not automatically better

ResNet is one of the most important breakthroughs in neural network design since AlexNet: it made it possible to train very deep networks, eventually more than 1000 layers.

Its authors began with an experiment. They extended VGG-19 with additional layers and found that the deeper network had a **higher training error** than the shallower one for the same number of training iterations. Very deep plain networks converged so slowly as to be impractical to train.

{{fig degradation}}

Notice that this is a failure on the *training* set. A deeper network has strictly more capacity — it could in principle copy the shallow network and make the extra layers do nothing — so the problem is not that it overfits. The problem is that it **cannot be optimised**. Residual connections fix exactly that.

## Residual learning

{{fig residual_block}}

The basic residual block has a main path of convolution and activation layers and a shortcut from the input, which is **added** to the main path before the final activation.

Training a network without shortcuts, we can think of a block as converging to some desired mapping $H(x)$. With residual learning, the main path instead learns

$$F(x) = H(x) - x, \tag{1}$$

and the original mapping is recovered as $H(x) = F(x) + x$, with $x$ supplied by the shortcut. These residual mappings turn out to be much easier to train than the original mappings.

## Why it works: an ensemble of paths

A useful explanation comes from Veit and colleagues. Unravelling a ResNet shows that every block offers two routes — through its layers, or round them by the shortcut — so the network is really an **ensemble** of many paths of different lengths.

{{fig unravelled_paths}}

The ensemble view explains a striking observation: a residual block can be removed from a trained ResNet without greatly affecting its performance, which would be catastrophic in a plain network.

Many of the paths through the ensemble are short. During training, it is observed that the **short paths carry the larger-magnitude gradients**, and so determine the weight updates. The conclusion is that residual connections enable deep networks by shortening the effective paths, avoiding the vanishing gradient problem.

That explanation is somewhat contradicted by the original ResNet authors, who considered it unlikely that the difficulties of deep plain networks were caused by vanishing gradients. The exact reason shortcut connections help training remains an open research question — but there is no doubt about their effectiveness.

## The ResNet family

{{fig resnet_family}}

A complete ResNet is a stack of many residual blocks with a regular, repeating structure. After an initial 7 × 7 convolution and max pooling, each depth of ResNet is built from repeating units arranged in four stages.

ResNet-18 and -34 use blocks of two 3 × 3 convolutions. ResNet-50, -101 and -152 use **bottleneck** blocks, in which 1 × 1 convolutions reduce and then restore the channel count around the 3 × 3 convolution, to reduce computation — exactly the bottleneck idea of Lesson 1.3.

## Batch normalisation

ResNet also uses **batch normalisation** layers to improve training speed. Batch norm is now used in most modern networks, and it also has a regularising effect. In the original ResNet a batch norm layer follows each convolution.

{{fig batch_norm}}

For each channel, the mean and variance of the input are computed over the mini-batch:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i, \qquad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu_B)^2. \tag{2}$$

The data is normalised, then scaled and shifted by two learned parameters, $\gamma$ and $\beta$:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \qquad y_i = \gamma\hat{x}_i + \beta. \tag{3}$$

$\gamma$ and $\beta$ are learned independently for each batch norm layer, and since each layer learns only these two per channel, batch norm adds very little to the parameter count.

The original batch norm paper described its benefit as reducing **internal covariate shift** — the changes at each layer's input caused by weight updates in the layers before it. Normalising keeps those input statistics more consistent during training and so helps convergence.

Why batch norm really helps is, like residual connections, still an open question, and later work argues against the covariate-shift explanation. Experiments show that networks without batch norm can only train with low learning rates and converge slowly, while networks with it tolerate much higher learning rates. The same work shows that without batch norm, the internal states of deep layers **explode** — in a 110-layer ResNet, the variance of one channel in layer 80 reached $10^{16}$. Batch norm corrects each channel's activations and stops them accumulating through the network, which is the effect panel (b) of the figure reproduces.

{{example bat_norm1 seed=4}}

## Batch norm at inference

{{fig batch_norm_running}}

At inference there may be only a single image, so there is no batch to compute statistics from. During training, therefore, a **running** (global) mean and variance are also accumulated, with a decay parameter $d$:

$$\mu \leftarrow d\,\mu + (1 - d)\,\mu_B, \qquad \sigma^2 \leftarrow d\,\sigma^2 + (1 - d)\,\sigma_B^2. \tag{4}$$

At inference these stored values replace $\mu_B$ and $\sigma_B^2$ in equation (3).

{{example bat_norm2 seed=4}}

## Squeeze-and-excite networks

The squeeze-and-excite (SE) networks are created by adding SE blocks to Inception or ResNet. In a ResNet the SE block is inserted between the last convolution of the main path and the shortcut addition.

{{fig squeeze_excite}}

The input is a volume of height × width with C channels. Global average pooling reduces it to 1 × 1 × C. This is passed through a cascade of fully connected layers with activations, the last a sigmoid, which limits the values to between 0 and 1. These values are then used as scale factors, one per channel of the input. The block thereby lets **interdependencies between channels** be modelled and exploited. It is computationally lightweight, and adding it to ResNet-50 reduces ImageNet error by roughly a tenth.

## DenseNet

DenseNet takes the concept of ResNet and greatly extends the connectivity.

{{fig densenet}}

Like ResNet, DenseNet begins with a large-kernel convolution and max pooling. The rest of the network is a series of **dense blocks** separated by simple **transition** blocks of 1 × 1 convolution and average pooling, and it ends with global average pooling and softmax.

Inside a dense block, each composite function — batch norm, ReLU and convolution, in that pre-activation order — is similar to a ResNet layer, except that instead of **summing** the shortcut with the main path, DenseNet **concatenates** the feature maps. Every composite function therefore receives the outputs of all previous stages, including the block's input, which creates dense connectivity. If the input has $k_0$ channels and each function adds $k$ new feature maps, there are $k_0 + nk$ channels after $n$ functions.

The benefit is similar to ResNet's: many short paths, with no problems from exploding or vanishing gradients. Despite having far more connectivity, a DenseNet of equivalent performance has many fewer parameters — it can be seen as a less redundant representation of the data. DenseNet is **narrow but deep**: each 3 × 3 convolution typically has only $k = 32$ kernels, against 64 to 256 channels in ResNet's blocks, and bottleneck 1 × 1 layers again reduce the computation.

{{selfcheck 02_03}}
