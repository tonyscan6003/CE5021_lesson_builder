---
id: "01_06"
title: Overfitting and regularisation
section: 1
engines: [s1_convnets.regularisation]
practice: [dropout]
summary: The bias-variance trade-off, detection of under- and overfitting from training curves, generalisation, the curse of dimensionality, and the regularisation methods used with neural networks — L1/L2, dropout, spatial dropout and data augmentation.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain the **bias-variance trade-off** and relate it to under- and overfitting;
- detect under- and overfitting from **training and validation curves**;
- explain **generalisation** and why the test set must not be used during training;
- explain the **curse of dimensionality** and why CNNs are less susceptible to it;
- describe **L1 and L2 regularisation** and the difference between them;
- explain **dropout** and why **spatial dropout** is used in convolutional networks;
- explain how overfitting is reduced by **data augmentation**.

## Introduction

In this lesson an entire neural network is treated as a black-box classifier: a mapping from input data to labels, or equivalently the problem of finding a decision boundary between classes. Overfitting is a serious problem for neural networks, and in this lesson the causes of overfitting and the regularisation techniques used to reduce it are described.

## The bias-variance trade-off

**Bias** and **variance** are two sources of error in a trained model.

- **Bias** is the error inherent in the complexity of the model.
- **Variance** is the amount by which the result changes when different training data is used.

The two are related. Less complex models tend to have **high bias and low variance**, and more complex models tend to have **low bias and high variance**.

{{fig bias_variance}}

The total error follows a U-shaped curve: it is high for both low- and high-complexity models, with a minimum at an optimal complexity.

Only simple decision boundaries can be produced by a low-complexity model, such as a linear classifier, and these may not fit the data well. This is known as **underfitting**. A high-complexity model may fit the training data so closely that the decision boundary takes a very complicated shape, following noise and outliers rather than the general trend. This is known as **overfitting**, and it is characterised by a significant change in the decision boundary when different training data is used.

In summary, a model that is too simple results in underfitting, with high bias and low variance, and a model that is too complex results in overfitting, with low bias and high variance.

It should be noted in the figure that the **training error decreases monotonically** as complexity increases. For this reason, the training error cannot be used to select a model.

## Detection during training

Under- and overfitting can be detected during training by inspection of the error on the training set and on a **validation set**.

{{fig train_val_curves}}

In the **underfitting** case both errors remain high. This is expected with high bias: the decision boundary does not fit the data well, and a low error is not obtained with further training.

In the **overfitting** case the two errors **diverge**. The training error continues to decrease, while the validation error initially decreases and then starts to increase. As the training data is fitted more and more closely, the model becomes less able to make good predictions on data that it has not seen.

For this reason, validation data must always be used during training, as the training error alone gives a misleading result. This is also the basis of **early stopping**, in which training is stopped at the minimum of the validation curve.

For an **optimal** model, both errors decrease together to a low value.

## Generalisation

**Generalisation** is the ability of the model to perform well on unseen data, and it is the aim of the training process. A model that overfits the training data is of no value, as it will perform poorly on both validation and test data.

It is essential that **test data is not used during training**, or misleading performance results will be obtained. For good generalisation, the model must be neither underfitting nor overfitting.

## The curse of dimensionality

When an image is applied to a classifier, each pixel can be considered as a dimension of the data, so even a small $256 \times 256$ image has 65,536 dimensions.

{{fig curse_of_dimensionality}}

The addition of dimensions can help to discriminate between classes. However, when the number of data points is held constant, the distance between points generally increases as dimensions are added. This increase is called **sparsity**, and it continues to increase with further dimensions. When the training data is sparse, overfitting is likely.

If more training data is added to compensate, the amount required increases **exponentially** with each dimension in order to maintain generalisation. This exponential requirement is known as the curse of dimensionality.

Raw images are always applied at the input of a neural network, which suggests that the curse of dimensionality applies. However, as the convolution layers at the input operate only locally, and as features are extracted by the network before they are classified in the final layers, neural networks are **less susceptible** to the curse of dimensionality than classical machine learning algorithms.

## Overfitting in neural networks

A key feature of neural networks is that they are **over-parameterised**: they contain a very large number of parameters and, without regularisation, a complex model of the data tends to be formed and the training set is strongly overfitted.

Several remedies are available, and they are usually combined:

- **early stopping**, as described above;
- **regularisation**, in which a penalty is added to the loss function to prevent large weights from dominating;
- **dropout**, in which neurons are randomly ignored during each training update;
- **batch normalisation**, which is mainly used to accelerate training but also has a regularising effect (Lesson 2.3);
- **more training data**, including data created by augmentation.

## L1 and L2 regularisation

L1 and L2 regularisation are used throughout machine learning. Large parameter values are penalised in the loss function, so that the model is constrained towards parameters of more equal size and better generalisation is obtained. In **L1** regularisation the absolute values of the weights are penalised, and in **L2** regularisation the penalty is proportional to their squares:

$$L_{\text{total}} = L + \lambda \sum_i |w_i| \quad \text{(L1)}, \qquad L_{\text{total}} = L + \lambda \sum_i w_i^2 \quad \text{(L2)}. \tag{1}$$

{{fig l1_l2_regularisation}}

The behaviour of the two methods differs, and the difference can be explained geometrically. The L1 constraint region is a diamond with corners on the axes, so the optimum is often located exactly at a corner and a weight is driven to **zero**; sparse solutions are therefore produced by L1. The L2 constraint region is a circle without corners, so the weights are reduced but not eliminated.

In neural networks, regularisation is applied to the parameters of the convolution and dense layers. In PyTorch, L2 regularisation is usually applied through the `weight_decay` argument of the optimiser, for example `torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)`, and can be restricted to particular layers, such as the kernels but not the biases, by the use of parameter groups. L1 regularisation is added directly to the loss before the backward pass:

```python
loss = criterion(model(x), y)
loss = loss + lam * sum(p.abs().sum() for p in model.parameters())
loss.backward()
```

## Dropout

In **dropout**, nodes of the network are randomly removed during training. At each training step a node may be removed with a given probability, and a sparser network is produced at each interval.

{{fig dropout_mask}}

As a result, the network is prevented from relying on specific paths or neurons, and a regularising effect similar to an L2 penalty is obtained through a reduction in the weight values.

When a fraction of the units is set to zero, the expected output of the layer is reduced. In **inverted dropout** the remaining activations are therefore divided by $(1 - \text{rate})$, so that the expected value is unchanged and no rescaling is required at inference time.

In PyTorch, dropout is applied as an independent layer, `nn.Dropout(p=0.5)`, where `p` is the dropout rate. Dropout is applied only in training mode, set by `model.train()`; in evaluation mode, set by `model.eval()`, the layer has no effect, and as inverted dropout is used, no rescaling is required. If too large a dropout rate is used, convergence may be slowed or underfitting may occur.

{{example dropout seed=6}}

## Spatial dropout

Dropout was originally designed for fully connected networks, and it is less effective in convolutional networks, for the following reason.

{{fig spatial_dropout}}

The convolution kernel is applied across a **region** of the preceding feature map. If individual pixels are removed at random from that region by dropout, and the pixels in the region are highly correlated, as is very likely in a CNN, the result of the convolution may be changed very little, and little regularisation is achieved.

In **spatial dropout**, entire feature maps are dropped instead of single pixels; in PyTorch this is implemented by `nn.Dropout2d`. This has a much greater effect on specific kernel channels, which are prevented from learning from particular correlated features during the training interval.

## Data augmentation

The final approach is **data augmentation**. Ideally, overfitting would be prevented by training for longer on new data rather than repeating the same examples. However, labelled datasets are finite, and the production of new labelled data is expensive and time consuming.

In data augmentation, the existing data is modified so that variations of the images can be used as new training examples.

{{fig data_augmentation}}

The simplest approach is to apply **affine transforms**: rotation, translation, scaling and shear. More complex augmentations are available, in which segments of an image are removed or parts of several images are combined, but the basic affine transformations are effective. In PyTorch these are implemented as transforms in `torchvision.transforms`, such as `RandomAffine`, `RandomRotation` and `RandomHorizontalFlip`, which are applied to each image as it is loaded during training, so that a different variation is produced every epoch.

As with the dropout rate, the amount of augmentation must be chosen with care, as too much augmentation causes underfitting. The appropriate level is usually found by experiment, together with the settings of the other regularisation methods, for the task being considered.

{{selfcheck 01_06}}
