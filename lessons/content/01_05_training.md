---
id: "01_05"
title: Training neural networks
section: 1
engines: [s1_convnets.training]
practice: [cross_ent, lr_schedule, lr_schedule_cos]
summary: Supervised learning and the datasets behind it, cross-entropy loss, gradient descent and the learning rate, backpropagation, batch size, momentum, and learning-rate schedules.
---

## Learning outcomes

After this lesson you should be able to:

- describe **supervised learning** and say what distinguishes the deep learning approach from classical computer vision;
- name the standard **datasets** and what they are used for;
- write down and interpret the **binary** and **categorical cross-entropy** losses;
- explain the **gradient descent** update rule and the effect of the **learning rate**;
- explain what **backpropagation** computes and why it is efficient;
- compare **batch**, **stochastic** and **mini-batch** gradient descent;
- explain what **momentum** adds;
- describe common **learning-rate schedules**.

## Supervised learning

Supervised learning is the dominant paradigm in computer vision, used for everything from simple classifiers to state-of-the-art deep learning systems. It is the formalisation of learning from examples, and its defining feature is the presence of **labelled training data**.

{{fig supervised_pipeline}}

In classical computer vision the engineer had to design a feature extractor that converted images into an abstract representation the classifier could learn from. Only the classifier's internal parameters were optimised.

Deep learning differs in two ways. The models are **deep** — large networks with many convolution layers — and the features are learned from the images in an **end-to-end** fashion. The feature learning is automatic, and the fully connected layers forming the classifier are optimised at the same time. The cost is that deep models have a great many parameters, so large amounts of data and computation are needed to train them successfully.

On each training iteration a portion of the training data is passed through the network. The predictions and the true labels are combined in the **loss function**, the optimisation algorithm minimises $J(\theta)$, and the model parameters $\theta$ are updated. Different applications use widely different loss functions, but the optimisation algorithm is much the same in every case.

## Datasets

{{fig datasets_table}}

The number and complexity of public datasets has grown enormously over the last twenty years. Small early sets such as MNIST and CIFAR-10/100 are still very useful for prototyping. Large sets such as ImageNet and MS COCO are what enabled state-of-the-art deep networks to be trained at all. There are also specialised sets for particular tasks, which later sections of the module return to.

## Loss functions

The loss function measures the current error of the model and allows the optimiser to adjust the parameters so that the loss is minimised. Using the correct loss for the task matters: with the wrong one the optimiser may fail to converge and the model will not learn.

### Binary cross-entropy

For a binary problem — telling cats from dogs — take the ground truth labels to be $y_i = 0$ or $y_i = 1$. To constrain the network to output a probability, the final single neuron uses a sigmoid activation. The loss is then

$$L = -\frac{1}{m}\sum_{i=1}^{m}\Big[\, y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \,\Big]. \tag{1}$$

The loss has two components and **only one is active for each sample**, depending on the ground truth. If $y_i = 1$ the $\log(\hat{y}_i)$ term determines the loss, and it falls as the prediction approaches 1. If $y_i = 0$ the $\log(1 - \hat{y}_i)$ term is active, and the loss is minimised when the prediction is close to 0. The result is averaged over the $m$ training examples.

### Categorical cross-entropy

For multi-class classification the final layer uses softmax, producing a probability vector $\hat{\mathbf{y}}$ of length $N$ that sums to 1. The loss is

$$L = -\sum_{k=1}^{N} y_k \log(\hat{y}_k). \tag{2}$$

The ground truth $\mathbf{y}$ is usually given in **one-hot** format: 1 for the correct class and 0 for all others. Every term but one is therefore multiplied by zero, so the loss depends only on the probability the network assigned to the correct class.

{{fig cross_entropy}}

As in the binary case, the logarithm is large when the prediction is wrong and shrinks as it becomes more nearly correct. That is exactly what a training signal needs to be.

{{example cross_ent seed=3}}

## Gradient descent

The key algorithm for training neural networks is **gradient descent**, which minimises the value of the loss function $J(\theta)$. From a random starting value of $\theta$, it moves in the direction of steepest descent, one step at a time, using the update rule

$$\theta \leftarrow \theta - \eta\, \frac{\partial J(\theta)}{\partial \theta}. \tag{3}$$

The other important quantity is $\eta$, the **learning rate**, which determines how large a step is taken on each iteration.

{{fig gradient_descent}}

The learning rate has to be chosen with care:

- **too low** and the algorithm needs a great many updates to find the minimum;
- **too high** and the behaviour becomes divergent — the loss increases rather than decreasing;
- the goal is the rate that finds the minimum in the fewest updates while remaining stable.

## Backpropagation

Gradient descent needs the partial derivative of the loss with respect to each individual parameter. The efficient way to obtain them is **backpropagation**, first published in 1989, which is an application of the chain rule from differentiation.

{{fig backprop_chain}}

Training data is passed forward through the network and a prediction made for each example. Those predictions and the true labels go to the loss function. During the backward pass, gradients for every parameter are obtained in a computationally efficient manner in a single sweep back through the network, and gradient descent then uses them to update the parameters.

The efficiency is the whole point. Estimating each gradient separately, by perturbing one parameter at a time and re-running the network, would require as many forward passes as there are parameters. Backpropagation gets all of them in one backward pass — which is what makes training a network with millions of parameters possible.

## Batch size

How much training data should be used for each update?

{{fig batch_size_noise}}

**Batch** gradient descent applies all the examples in the training set. This is impractical for large datasets: the working memory needed for gradient descent and backpropagation would easily exceed what is available, even on the best GPUs.

**Stochastic** gradient descent applies only one example per iteration. It still works, but converges to the minimum in a noisy manner. That noise can be an advantage — in a complex optimisation problem it can allow the optimiser to escape a local minimum — but updating the network for every single training example is computationally inefficient.

**Mini-batch** gradient descent is the compromise used in practice. A limited number of examples — commonly 16, 32 or 64 — are applied on each iteration. There is still some noise in the process, but processing a batch of images together suits the GPU's parallelism.

## Momentum

Plain SGD can struggle with regions of low gradient and with local minima in the optimisation surface. The simplest upgrade is **SGD with momentum**, which interprets the update as a velocity:

$$v \leftarrow \gamma v + \eta\,\frac{\partial J(\theta)}{\partial \theta}, \qquad \theta \leftarrow \theta - v. \tag{4}$$

{{fig momentum_path}}

Including previous gradients in the update damps the effect of noise in the gradients and preserves the general trend. Momentum builds as the optimiser passes through a region of large gradients; that momentum then increases the step taken through flat regions and prevents the optimiser becoming stuck in a local minimum.

Both $\eta$ and $\gamma$ must be set. Setting $\gamma$ close to 1 gives larger updates but can cause convergence problems, and the momentum value is sometimes increased over the course of training. A great many deep networks are trained with SGD plus momentum.

## Learning-rate schedules

A single fixed learning rate forces a compromise: large enough to make progress early in training, small enough to settle near the minimum at the end. A **schedule** sidesteps this by reducing $\eta$ as training proceeds.

{{fig lr_schedules}}

{{example lr_schedule seed=5}}

{{example lr_schedule_cos seed=5}}

In practice, training a network is largely a matter of getting these choices right: the loss function for the task, the learning rate and its schedule, the batch size, and the optimiser. Frameworks such as TensorFlow and PyTorch automate the mechanics of the forward and backward passes, but these decisions remain the designer's.

{{selfcheck 01_05}}
