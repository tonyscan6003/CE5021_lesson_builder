---
id: "01_02"
title: Fully connected layers and classifiers
section: 1
engines: [s1_convnets.fc_layers]
practice: [single_neu, softmax_act, fc_param]
summary: Neural network basics — the artificial neuron, activation functions, fully connected layers, binary and multi-class classifiers, the softmax output, and how to count a layer's parameters.
---

## Learning outcomes

After this lesson you should be able to:

- describe the structure of a **single artificial neuron** and compute its output;
- name the common **activation functions**, explain why non-linearity is necessary, and say what saturation costs;
- explain what makes a layer **fully connected**;
- describe how a network is used as a **binary** and as a **multi-class** classifier;
- compute the **softmax** output of a layer and check the result;
- calculate the **number of parameters** in a fully connected layer.

## The single neuron

An artificial neuron multiplies each of its inputs by a weight unique to that input, so $x_1$ is multiplied by $w_1$, $x_2$ by $w_2$, and so on. The products are summed, and an additional **bias** $b$ is added. Finally a function $\sigma$, the **activation function**, is applied to the sum:

$$y = \sigma(z) = \sigma\!\left(\sum_{i=1}^{n} x_i w_i + b\right). \tag{1}$$

{{fig single_neuron}}

The weights and the bias are the neuron's **parameters** — the numbers that training adjusts. A neuron with $n$ inputs has $n + 1$ of them.

{{example single_neu seed=3}}

## Activation functions

The activation function is normally non-linear, and this is its whole purpose: it adds non-linearity to the overall network function. Without it, a stack of layers would collapse into a single linear map no matter how many layers it had, and the network could only ever produce a linear decision boundary. Non-linear activations are what allow a network to learn the complex functions needed for tasks like image recognition.

{{fig activation_functions}}

In early neural networks the **sigmoid** and **tanh** functions were used almost exclusively. They are still used today, mainly where a bounded output is wanted — a sigmoid on a binary classifier's output, for example. They have two disadvantages: they are slow to compute, and they **saturate**. For inputs far from zero the curve flattens, so its derivative is nearly zero, and since backpropagation multiplies by that derivative, almost no gradient reaches the earlier layers. Learning stalls.

The **rectified linear unit (ReLU)** and **leaky ReLU** are the defaults in modern deep learning. They are fast to compute and do not saturate for positive inputs. Leaky ReLU keeps a small gradient for negative inputs so that a unit cannot become permanently inactive.

## Fully connected layers

The simplest network is the one-hidden-layer network: an **input layer**, which is just the inputs; a **hidden layer**; and an **output layer**. Both the hidden and output layers contain neurons, and each neuron behaves exactly as described above.

{{fig fc_network}}

This type of network is called **fully connected**: every neuron in a layer is connected to every neuron or input in the previous layer. The network can be expanded to any number of hidden layers, any number of neurons per layer, and any number of inputs or outputs.

A neural network can be interpreted as a mathematical function mapping the inputs $x_i$ to the output. Given enough layers and neurons it can build an internal model of the data presented at its inputs, and so act as a **probabilistic classifier** — meaning it outputs the *probability* that an input belongs to a class, rather than assigning a class label directly. A fully connected network with hidden layers is capable of simple computer vision tasks such as handwritten digit recognition.

## Binary and multi-class classifiers

For a **binary** task — distinguishing pedestrians from background in a driving scene, say — the output neuron uses a sigmoid activation, which guarantees a result between 0 and 1. A value close to 1 means the network is confident the image contains a pedestrian; a value close to 0 means it is confident it does not.

For a **multi-class** task the last layer must have as many neurons as there are classes. These neurons do **not** use an ordinary activation function. Instead the sums $z_i$ from each neuron are converted into normalised probabilities by the **softmax** function, and the probabilities always sum to 1. We take the class with the highest probability to be what the image contains.

{{fig nn_classifiers}}

## Softmax activation

Softmax turns a vector of sums into a probability distribution:

$$p_i = \sigma(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{k} e^{z_j}}. \tag{2}$$

The exponential of each $z_i$ is calculated, and each is divided by the sum of all of them.

{{fig softmax_activation}}

Because the probabilities must sum to 1, adding them up is a quick and reliable check on your arithmetic. Note also that the exponential exaggerates differences: a logit one unit larger than another ends up roughly $e \approx 2.7$ times more probable.

{{example softmax_act seed=7}}

## Counting parameters

When constructing a network it is useful to be able to work out how many parameters a fully connected layer will need. Each neuron connects to all of its inputs with one weight per connection, and each neuron has its own bias.

Fully connected layers are frequently attached to the output of convolution layers. To do that, the feature map must first be **flattened** into a 1D vector: a feature map of spatial size $n \times m$ with $C$ channels becomes a vector of length $n \times m \times C$. Connecting that vector to a layer of $N$ neurons gives

$$\text{parameters} = (n \cdot m \cdot C) \cdot N + N, \tag{3}$$

the first term being the weights and the second the biases.

{{fig fc_param_count}}

Even with relatively few neurons, fully connected layers accumulate a large number of parameters and dominate a network's memory requirements. This is worth remembering: it is the observation that the next lesson starts from, when we ask why fully connected layers are a poor way to process images in the first place.

{{example fc_param seed=9}}

{{selfcheck 01_02}}
