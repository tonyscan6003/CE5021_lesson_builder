---
id: "01_02"
title: Fully connected layers and classifiers
section: 1
engines: [s1_convnets.fc_layers]
practice: [single_neu, softmax_act, fc_param]
summary: Neural network basics — the artificial neuron, activation functions, fully connected layers, binary and multi-class classifiers, the softmax output, and the number of parameters in a layer.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- describe the structure of a **single artificial neuron** and calculate its output;
- name the common **activation functions**, explain why non-linearity is necessary, and describe the effect of saturation;
- explain what is meant by a **fully connected** layer;
- describe how a network is used as a **binary** and as a **multi-class** classifier;
- calculate the **softmax** output of a layer and check the result;
- calculate the **number of parameters** in a fully connected layer.

## The single neuron

In an artificial neuron, each input is multiplied by a weight unique to that input, so $x_1$ is multiplied by $w_1$, $x_2$ by $w_2$, and so on. The products are summed and an additional **bias** $b$ is added. Finally a function $\sigma$, the **activation function**, is applied to the sum:

$$y = \sigma(z) = \sigma\!\left(\sum_{i=1}^{n} x_i w_i + b\right). \tag{1}$$

{{fig single_neuron}}

The weights and the bias are the **parameters** of the neuron, and their values are adjusted during training. A neuron with $n$ inputs has $n + 1$ parameters.

{{example single_neu seed=3}}

## Activation functions

The activation function is normally non-linear, and its purpose is to add non-linearity to the overall network function. Without a non-linear activation, a stack of layers would reduce to a single linear mapping regardless of the number of layers, and only a linear decision boundary could be produced. Non-linear activations allow the complex functions needed for tasks such as image recognition to be learned.

{{fig activation_functions}}

In early neural networks the **sigmoid** and **tanh** functions were used almost exclusively. They are still used, mainly where a bounded output is required, for example a sigmoid at the output of a binary classifier. These functions have two disadvantages: they are slow to compute, and they **saturate**. For inputs far from zero the curve flattens and its derivative is close to zero. As the gradient is multiplied by this derivative during backpropagation, very little gradient reaches the earlier layers and learning is slowed or stopped.

The **rectified linear unit (ReLU)** and **leaky ReLU** are the default activation functions in modern deep learning. They are fast to compute and do not saturate for positive inputs. In leaky ReLU a small gradient is retained for negative inputs, so that a unit cannot become permanently inactive.

## Fully connected layers

The simplest network is the one-hidden-layer network, which consists of an **input layer** (the inputs themselves), a **hidden layer** and an **output layer**. The hidden and output layers both contain neurons, and each neuron operates as described above.

{{fig fc_network}}

This type of network is described as **fully connected**, as every neuron in a layer is connected to every neuron or input in the previous layer. The network can be extended to any number of hidden layers, any number of neurons per layer, and any number of inputs or outputs.

A neural network can be interpreted as a mathematical function that maps the inputs $x_i$ to the output. With sufficient layers and neurons, an internal model of the data presented at the inputs can be formed, and the network can act as a **probabilistic classifier**: the *probability* that an input belongs to a class is output, rather than a class label being assigned directly. A fully connected network with hidden layers is capable of simple computer vision tasks such as handwritten digit recognition.

## Binary and multi-class classifiers

For a **binary** task, such as distinguishing pedestrians from background in a driving scene, a sigmoid activation is used at the output neuron, so that the result is between 0 and 1. A value close to 1 indicates high confidence that the image contains a pedestrian, and a value close to 0 indicates high confidence that it does not.

For a **multi-class** task, the last layer must have as many neurons as there are classes. An ordinary activation function is **not** used for these neurons. Instead, the sums $z_i$ from each neuron are converted into normalised probabilities by the **softmax** function, and the probabilities always sum to 1. The class with the highest probability is taken as the class contained in the image.

{{fig nn_classifiers}}

## Softmax activation

The softmax function converts a vector of sums into a probability distribution:

$$p_i = \sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}. \tag{2}$$

The exponential of each $z_i$ is calculated, and each is divided by the sum of all of the exponentials.

{{fig softmax_activation}}

As the probabilities must sum to 1, the sum provides a quick check on the arithmetic. It should also be noted that differences are exaggerated by the exponential: a logit that is one unit larger than another results in a probability approximately $e \approx 2.7$ times greater.

{{example softmax_act seed=7}}

## Counting parameters

When a network is being designed, it is useful to be able to calculate the number of parameters required by a fully connected layer. Each neuron is connected to all of its inputs, with one weight per connection, and each neuron has its own bias.

Fully connected layers are frequently connected to the output of convolution layers. For this, the feature map must first be **flattened** into a 1D vector: a feature map of spatial size $n \times m$ with $C$ channels becomes a vector of length $n \times m \times C$. When this vector is connected to a layer of $N$ neurons, the number of parameters is

$$\text{parameters} = (n \cdot m \cdot C) \cdot N + N, \tag{3}$$

where the first term is the number of weights and the second the number of biases.

{{fig fc_param_count}}

Even with relatively few neurons, a large number of parameters is accumulated in fully connected layers, and these layers dominate the memory requirements of a network. This observation is the starting point of the next lesson, in which the use of fully connected layers to process images is examined.

{{example fc_param seed=9}}

{{selfcheck 01_02}}
