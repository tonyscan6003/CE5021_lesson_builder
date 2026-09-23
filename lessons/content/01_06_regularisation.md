---
id: "01_06"
title: Overfitting and regularisation
section: 1
engines: [s1_convnets.regularisation]
practice: [dropout]
summary: The bias-variance trade-off, detecting under- and overfitting from training curves, generalisation, the curse of dimensionality, and the regularisers used with neural networks — L1/L2, dropout, spatial dropout and data augmentation.
---

## Learning outcomes

After this lesson you should be able to:

- explain the **bias-variance trade-off** and relate it to under- and overfitting;
- detect under- and overfitting from **training and validation curves**;
- explain **generalisation** and why the test set must not be used during training;
- explain the **curse of dimensionality** and why CNNs are less susceptible to it;
- describe **L1 and L2 regularisation** and the difference between them;
- explain **dropout** and why **spatial dropout** is used in convolutional networks;
- explain how **data augmentation** reduces overfitting.

## Introduction

For this lesson we can treat an entire neural network as a black-box classifier: a mapping from input data to labels, or equivalently the problem of finding a decision boundary between classes. Overfitting is a serious problem for neural networks, and this lesson covers both why it happens and the regularisation techniques that mitigate it.

## The bias-variance trade-off

**Bias** and **variance** are two sources of error in a trained model.

- **Bias** is the error inherent in the model's complexity.
- **Variance** is the amount the result changes with different training data.

The two are linked. Less complex models tend to have **high bias and low variance**; more complex models tend to have **low bias and high variance**.

{{fig bias_variance}}

The total error is a U-shaped curve — high for both low- and high-complexity models, with a minimum at some optimal complexity.

A low-complexity model, such as a linear classifier, can only produce simple decision boundaries, which may not fit the data well. This is **underfitting**. A high-complexity model may fit the training data so closely that the decision boundary becomes a very complicated shape, following noise and outliers rather than the trend. This is **overfitting**, and its signature is that the boundary changes significantly if different training data is used.

In summary: too simple gives underfitting, with high bias and low variance; too complex gives overfitting, with low bias and high variance.

Note one thing carefully in the figure: the **training error falls monotonically** with complexity throughout. That is precisely why it cannot be used to choose a model.

## Detecting it during training

Under- and overfitting can be detected during training by inspecting the error on the training set and on a **validation set**.

{{fig train_val_curves}}

In the **underfitting** case both errors remain high. This is to be expected with high bias: the decision boundary does not fit the data well, and no amount of further training will produce a low error.

In the **overfitting** case the two **diverge**. The training error continues to decrease while the validation error initially decreases and then starts to rise. As the model fits the training data ever more closely, it becomes less able to make good predictions on data it has not seen.

Detecting overfitting is the reason validation data must always be used during training — relying on the training error alone will mislead you. It is also the basis of **early stopping**: stop training at the minimum of the validation curve.

For an **optimal** model, both errors reduce together to a low value.

## Generalisation

**Generalisation** is the ability of the model to perform well on unseen data, and it is the goal of the whole training process. A model that merely overfits the training data is of no value, since it will perform poorly on validation and test data alike.

It is imperative that **test data is not used during training**, or misleading performance results will be obtained. Good generalisation requires that the model is neither underfitting nor overfitting.

## The curse of dimensionality

When an image is applied to a classifier, each pixel can be thought of as a dimension of the data, so even a small $256 \times 256$ image has 65,536 dimensions.

{{fig curse_of_dimensionality}}

Adding dimensions can help discriminate between classes. But with the number of data points held constant, the distance between points generally increases as dimensions are added. This increase is called **sparsity**, and it only grows with more dimensions. Sparse training data makes overfitting likely.

If we try to offset that by adding more training data, we find that the required amount increases **exponentially** with each dimension, just to maintain generalisation. That exponential requirement is the curse of dimensionality.

Neural networks always take raw images at the input, which suggests the curse applies to them. However, because convolution layers at the input operate only locally, and because the network learns to extract features before the final layers classify them, neural networks are **less susceptible** to it than classical machine learning algorithms.

## Overfitting in neural networks

A key feature of neural networks is that they are **over-parameterised**: they contain a very large number of parameters and, without regularisation, will tend to form a complex model of the data and strongly overfit the training set.

There are several remedies, and they are usually combined:

- **early stopping**, as above;
- **regularisation** — adding a penalty to the loss function to prevent large weights dominating;
- **dropout** — randomly ignoring neurons during each training update;
- **batch normalisation**, which mainly accelerates training but also has a regularising effect (Lesson 2.3);
- **more training data**, including data created by augmentation.

## L1 and L2 regularisation

L1 and L2 regularisation are used across machine learning generally. They work by penalising large parameter values in the loss function, which forces the model towards more equally sized parameters and better generalisation. The **L1** term penalises the absolute values of the weights; the **L2** term is proportional to their squares:

$$L_{\text{total}} = L + \lambda \sum_i |w_i| \quad \text{(L1)}, \qquad L_{\text{total}} = L + \lambda \sum_i w_i^2 \quad \text{(L2)}. \tag{1}$$

{{fig l1_l2_regularisation}}

The two behave differently, and the reason is geometric. L1's constraint region is a diamond with corners on the axes, so the optimum often lands exactly on a corner and a weight is driven to **zero** — L1 produces sparse solutions. L2's constraint region is a circle with no corners, so it shrinks weights without eliminating them.

In neural networks these can be applied to the parameters of convolution and dense layers individually. In the TensorFlow/Keras API the regulariser is passed as an argument to the layer, and may be applied to the kernel, the bias or the activation.

## Dropout

**Dropout** randomly removes nodes of the network during training. At any training step a node may be removed with some probability, producing a sparser network at each interval.

{{fig dropout_mask}}

The effect is to force the network not to rely on specific paths or neurons, which provides regularisation similar to an L2 penalty by reducing the weight values.

One detail matters in practice. If a fraction of the units is zeroed, the layer's expected output shrinks — so under **inverted dropout** the surviving activations are divided by $(1 - \text{rate})$, keeping the expected value unchanged and meaning nothing needs to be rescaled at inference time.

In the TensorFlow/Keras API dropout is applied as an independent layer. Using too large a dropout rate can slow down convergence or cause underfitting.

{{example dropout seed=6}}

## Spatial dropout

Dropout was originally designed for fully connected networks. In a convolutional network it works less well, and it is worth understanding why.

{{fig spatial_dropout}}

The convolution kernel is applied across a **region** of the preceding feature map. If dropout removes individual pixels at random from that region, and the pixels in the region are highly correlated — which is very likely in a CNN — then the result of the convolution may barely change, and little regularisation is achieved.

In **spatial dropout**, entire feature maps are dropped instead of single pixels. This has a much greater effect on specific kernel channels and prevents them from learning from particular correlated features during the training interval.

## Data augmentation

The final approach is **data augmentation**. Ideally we would prevent overfitting by training for longer on genuinely new data rather than repeating the same examples, but labelled datasets are finite, and producing new labelled data is expensive and time consuming.

Data augmentation modifies the existing data so that variations of the images can be used as new training examples.

{{fig data_augmentation}}

The simplest approach is to apply **affine transforms** — rotations, translations, scaling and shear. There are more complex augmentations that remove segments of an image or combine parts of several, but the basic affine transformations work well. In the TensorFlow/Keras API these are implemented as layers that can be included at the input to any model.

As with the dropout rate, the amount needs judgement: too much augmentation causes underfitting, so the right level usually has to be found by experiment, alongside the other regularisers, for the task at hand.

{{selfcheck 01_06}}
