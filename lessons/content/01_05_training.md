---
id: "01_05"
title: Training neural networks
section: 1
engines: [s1_convnets.training]
practice: [cross_ent, lr_schedule, lr_schedule_cos]
summary: Supervised learning and the datasets used for it, cross-entropy loss, gradient descent and the learning rate, backpropagation, batch size, momentum, and learning-rate schedules.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- describe **supervised learning** and the differences between the deep learning approach and classical computer vision;
- name the standard **datasets** and their uses;
- write down and interpret the **binary** and **categorical cross-entropy** losses;
- explain the **gradient descent** update rule and the effect of the **learning rate**;
- explain what is computed by **backpropagation** and why it is efficient;
- compare **batch**, **stochastic** and **mini-batch** gradient descent;
- explain the effect of **momentum**;
- describe common **learning-rate schedules**.

## Supervised learning

Supervised learning is the dominant paradigm in computer vision, and is used in applications ranging from simple classifiers to state-of-the-art deep learning systems. It is the formalisation of learning from examples, and its defining feature is the use of **labelled training data**.

{{fig supervised_pipeline}}

In classical computer vision, a feature extractor had to be designed by the engineer to convert images into an abstract representation from which the classifier could learn. Only the internal parameters of the classifier were optimised.

Deep learning differs in two ways. The models are **deep**, that is, large networks with many convolution layers, and the features are learned from the images in an **end-to-end** manner. The features are learned automatically, and the fully connected layers that form the classifier are optimised at the same time. However, deep models have a very large number of parameters, so large amounts of data and computation are required to train them successfully.

On each training iteration, a portion of the training data is passed through the network. The predictions and the true labels are combined in the **loss function**, $J(\theta)$ is minimised by the optimisation algorithm, and the model parameters $\theta$ are updated. Widely different loss functions are used in different applications, but the optimisation algorithm is similar in every case.

## Datasets

{{fig datasets_table}}

The number and complexity of public datasets has increased greatly over the last twenty years. Small early datasets such as MNIST and CIFAR-10/100 are still very useful for prototyping. Large datasets such as ImageNet and MS COCO enabled state-of-the-art deep networks to be trained. There are also specialised datasets for particular tasks, which are described in later sections of the module.

## Loss functions

The loss function is a measure of the current error of the model, and allows the parameters to be adjusted by the optimiser so that the loss is minimised. The correct loss function must be used for the task; if an incorrect loss function is used, the optimiser may fail to converge and the model will not learn.

### Binary cross-entropy

For a binary problem, such as distinguishing cats from dogs, the ground truth labels are taken to be $y_i = 0$ or $y_i = 1$. A sigmoid activation is used at the final single neuron, so that the output is constrained to be a probability. The loss is then

$$L = -\frac{1}{m}\sum_{i=1}^{m}\Big[\, y_i \log(\hat{y}_i) + (1 - y_i)\log(1 - \hat{y}_i) \,\Big]. \tag{1}$$

The loss has two components, and **only one is active for each sample**, depending on the ground truth. If $y_i = 1$, the loss is determined by the $\log(\hat{y}_i)$ term, which decreases as the prediction approaches 1. If $y_i = 0$, the $\log(1 - \hat{y}_i)$ term is active, and the loss is minimised when the prediction is close to 0. The result is averaged over the $m$ training examples.

### Categorical cross-entropy

For multi-class classification, softmax is used in the final layer, and a probability vector $\hat{\mathbf{y}}$ of length $N$ that sums to 1 is produced. The loss is

$$L = -\sum_{k=1}^{N} y_k \log(\hat{y}_k). \tag{2}$$

The ground truth $\mathbf{y}$ is usually given in **one-hot** format, with 1 for the correct class and 0 for all other classes. All terms except one are therefore multiplied by zero, and the loss depends only on the probability assigned by the network to the correct class.

{{fig cross_entropy}}

As in the binary case, the logarithm is large when the prediction is incorrect and decreases as the prediction approaches the correct value, which provides a suitable training signal.

{{example cross_ent seed=3}}

## Gradient descent

The key algorithm used to train neural networks is **gradient descent**, in which the value of the loss function $J(\theta)$ is minimised. From a random starting value of $\theta$, steps are taken in the direction of steepest descent using the update rule

$$\theta \leftarrow \theta - \eta\, \frac{\partial J(\theta)}{\partial \theta}. \tag{3}$$

The other important quantity is $\eta$, the **learning rate**, which determines the size of the step taken on each iteration.

{{fig gradient_descent}}

The learning rate must be chosen with care:

- if it is **too low**, a very large number of updates is required to find the minimum;
- if it is **too high**, the behaviour becomes divergent and the loss increases rather than decreases;
- the aim is to find the rate at which the minimum is reached in the fewest updates while remaining stable.

## Backpropagation

In gradient descent, the partial derivative of the loss with respect to each individual parameter is required. These derivatives are obtained efficiently by **backpropagation**, first published in 1989, which is an application of the chain rule of differentiation.

{{fig backprop_chain}}

Training data is passed forward through the network and a prediction is made for each example. The predictions and the true labels are applied to the loss function. During the backward pass, the gradients for every parameter are obtained in a computationally efficient manner in a single pass back through the network, and the parameters are then updated by gradient descent.

If each gradient were estimated separately, by perturbing one parameter at a time and running the network again, as many forward passes as there are parameters would be required. With backpropagation all of the gradients are obtained in one backward pass, and this efficiency allows networks with millions of parameters to be trained.

## Batch size

The amount of training data used for each update must also be chosen.

{{fig batch_size_noise}}

In **batch** gradient descent, all of the examples in the training set are applied. This is impractical for large datasets, as the working memory required for gradient descent and backpropagation would exceed that available, even on the best GPUs.

In **stochastic** gradient descent, only one example is applied per iteration. Convergence to the minimum still occurs, but in a noisy manner. The noise can be an advantage, as in a complex optimisation problem it may allow the optimiser to escape from a local minimum. However, updating the network for every training example is computationally inefficient.

**Mini-batch** gradient descent is the compromise used in practice. A limited number of examples, commonly 16, 32 or 64, is applied on each iteration. Some noise remains in the process, and the processing of a batch of images together is well suited to the parallelism of the GPU.

## Momentum

Plain SGD can perform poorly in regions of low gradient and in the presence of local minima in the optimisation surface. The simplest improvement is **SGD with momentum**, in which the update is interpreted as a velocity:

$$v \leftarrow \gamma v + \eta\,\frac{\partial J(\theta)}{\partial \theta}, \qquad \theta \leftarrow \theta - v. \tag{4}$$

{{fig momentum_path}}

When previous gradients are included in the update, the effect of noise in the gradients is reduced and the general trend is preserved. Momentum is built up as the optimiser passes through a region of large gradients, and this momentum increases the step taken through flat regions and prevents the optimiser from becoming stuck in a local minimum.

Both $\eta$ and $\gamma$ must be set. A value of $\gamma$ close to 1 gives larger updates but can cause convergence problems, and the momentum value is sometimes increased during training. A large number of deep networks are trained with SGD and momentum.

## Learning-rate schedules

With a single fixed learning rate, a compromise is required: the rate must be large enough for progress to be made early in training, and small enough for the parameters to settle near the minimum at the end. With a **schedule**, $\eta$ is reduced as training proceeds, and this compromise is avoided.

{{fig lr_schedules}}

{{example lr_schedule seed=5}}

{{example lr_schedule_cos seed=5}}

In practice, the successful training of a network depends largely on these choices: the loss function for the task, the learning rate and its schedule, the batch size, and the optimiser. The forward and backward passes are automated in frameworks such as TensorFlow and PyTorch, but these decisions must still be made by the designer.

{{selfcheck 01_05}}
