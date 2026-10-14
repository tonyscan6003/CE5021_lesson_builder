---
id: "02_04"
title: Transfer learning
section: 2
engines: [s2_classification.transfer]
practice: []
summary: The reuse of a network pre-trained on a large dataset — as a fixed or fine-tuned feature extractor, for classification and for new tasks such as detection — and the evidence on when it is and is not beneficial.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- explain the two problems addressed by transfer learning: **too little data** and **too much computation**;
- describe how a network is divided into a **feature extractor** and a **classifier**, and the difference between a **frozen** and a **fine-tuned** extractor;
- explain how transfer learning is applied when the **task** changes as well as the data;
- summarise the findings of **He et al.** and **Raghu et al.** on the benefits and limitations of pre-training;
- decide, for a given application, **whether and how** transfer learning should be used;
- instantiate a **pre-trained model** without its classifier.

## Motivation

Deep learning has become the dominant approach to computer vision as a result of several developments: large amounts of training data from the big-data revolution, deep architectures together with the algorithms used to train them (gradient descent and backpropagation), and high-performance parallel computing on GPUs.

When a deep learning solution is deployed to a new task or image domain, two problems commonly arise:

- there is **insufficient data** in the new domain to train a model to the required accuracy, which is often the case in specialised domains such as medical imaging;
- even when enough data is available, the **computational cost** of training a deep model can be high, or training can be technically difficult.

Transfer learning has become a very popular solution to both problems, and is used extensively in machine and computer vision.

## Feature extractor and classifier

{{fig transfer_pipeline}}

Transfer learning is based on the idea that a deep network can be divided into a **feature extractor** and a **classifier**. As shown in Lesson 2.5, convolution layers respond to image features: early layers respond to simple features such as lines, edges and blobs, and deeper layers to more specific and complex features. When a CNN is trained on a large dataset such as ImageNet, it is considered to have learned many **generic** image features.

The pre-trained convolution layers can then be reused in a new application. In the simplest case they are used as a **fixed** feature extractor, and only new classifier layers are learned. Usually, however, the feature extractor is also **fine-tuned**, with both the extractor and the new classifier trained together, and this improves performance.

The problem of limited target data is addressed through the generic features of the pre-trained layers. The computation required is also reduced, as the new system converges quickly to its final accuracy.

## Changing the task

In the case described so far, only the data has changed: a classifier is retrained to recognise new classes, but the task is still classification. Transfer learning can also be applied when **both the data and the task change**.

{{fig transfer_detection}}

For object detection, spatial information about the location of objects is retained in the final feature maps of the pre-trained extractor, and this is sufficient for a bounding-box predictor to be trained alongside a class predictor. The location of every object must be recorded in detection data, which limits the amount of data available, and for this reason transfer learning is often necessary for a detector to be trained.

## Limitations of transfer learning

Pre-training on ImageNet classification followed by transfer to the target task has become standard practice. As a result of its popularity, the assumptions behind it have been examined.

Several important observations were made by **He and colleagues (2018)**. As transfer learning was so widely used for tasks such as object detection, it had been assumed by many designers that detection networks *could not* be trained from scratch. It was shown that most detection networks can be trained from scratch, without specialised architectures. It was confirmed that **pre-training results in faster convergence**, but that the accuracy reached is **no better** than that obtained by training from scratch, which requires a longer training time. Training from scratch may fail to match transfer learning when the dataset is very small. In addition, when the pre-training itself is taken into account, transfer learning is not cheaper overall; the benefit comes from computation that has already been performed elsewhere.

{{fig pretrain_convergence}}

Specialised domains, in which transfer learning is very commonly used, were examined by **Raghu and colleagues**. When the target images differ greatly from the pre-training data, as medical images do, transfer learning may be less beneficial than assumed. On a diabetic retinopathy task, high accuracy was reached by ResNet-50 trained by transfer learning, and the same accuracy was reached when the same architecture was trained from scratch. In addition, lightweight architectures with far fewer parameters performed comparably when trained from scratch. These small models perform poorly on ImageNet, for which they lack the capacity, but they are sufficiently complex for the retinopathy task. Large models are often deployed by transfer learning because pre-trained weights are readily available, without consideration of whether a lightweight model would be better suited to the task.

{{fig data_size_sweep}}

## Recommendations

{{fig decision_grid}}

When the use of transfer learning for a new application is being considered, the similarity of the source and target **domains**, the similarity of their **labels**, and the **size** of the source and target datasets should be taken into account:

- Transfer learning may be the **only** possible solution when computation or target data is limited.
- The most favourable case is when the source and target domains are **similar** and the source dataset is **large**. If the classes or objects are also similar, excellent performance can be obtained even without fine-tuning, and with a small target dataset the performance is likely to be much better than that obtained by training from scratch.
- With a **large** target dataset, equivalent performance is obtained by transfer learning and by training from scratch, but training time is saved with transfer learning.
- If the domains are **less similar**, fine-tuning is necessary for good accuracy, and with a small target dataset there is a risk of overfitting to the target domain, which can result in poorer performance than training from scratch.
- If the domains are **dissimilar**, as with medical images, equivalent performance is obtained by training from scratch with a medium-sized dataset, although with a very small target dataset transfer learning still performs slightly better.
- With **limited computational resources**, training from scratch may be impractical, and transfer learning is then a good option regardless; many state-of-the-art object detectors are trained quickly in this way.

## Pre-trained models

{{fig pretrained_models}}

Many standard image recognition models, including ResNet-50, VGG-16, DenseNet and MobileNetV2, are available pre-trained in the `torchvision.models` module, and can be used either directly for classification or as the basis for transfer learning. A pre-trained model can be loaded and adapted to a new task in a few lines:

```python
import torch.nn as nn
from torchvision import models

weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
model = models.mobilenet_v2(weights=weights)          # ImageNet-trained network

for p in model.features.parameters():                 # freeze the feature extractor;
    p.requires_grad = False                           # omit this loop to fine-tune

model.classifier[1] = nn.Linear(model.last_channel, 5)   # new classifier, 5 classes
preprocess = weights.transforms()                     # resizing and normalisation
```

The pre-trained weights are selected by the `weights` argument; with `weights=None` the same architecture is created with random weights, for training from scratch. Each model consists of a feature extractor and a classifier. In MobileNetV2, VGG and DenseNet these are the `features` and `classifier` attributes, and in ResNet the classifier is the single dense layer `fc`. To adapt the model to a new task, the final classification layer is replaced by a new layer with one output per target class, here 1,280 inputs and 5 outputs, giving 6,405 new parameters. When `requires_grad` is set to `False` for the extractor, its weights are frozen and only the new classifier is trained; if the loop is omitted, the entire network is fine-tuned. The images must be prepared in the same way as the pre-training data, and the required resizing and normalisation are provided by `weights.transforms()`.

{{selfcheck 02_04}}
