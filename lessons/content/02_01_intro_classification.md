---
id: "02_01"
title: Introduction to image classification
section: 2
engines: [s2_classification.intro_classification]
practice: [con_mat]
summary: The image classification task and why it is hard, applications, standard datasets and the ImageNet challenge, the metrics used to evaluate classifiers — accuracy, precision, recall, top-1 and top-5 — and robustness, confusion and bias.
---

## Learning outcomes

After this lesson you should be able to:

- describe the **image classification** task and explain why it is difficult;
- explain the advantages and costs of **deep CNNs** for classification;
- name the **standard datasets** and describe the role of **ImageNet**;
- compute **accuracy, precision and recall** from a binary confusion matrix;
- compute them **one class at a time** for a multi-class problem, and explain **macro** and **weighted** averaging;
- explain **top-1** and **top-5** accuracy;
- describe **robustness**, **confusion** and **bias** as failure modes of trained classifiers.

## The classification task

In image classification we label an image according to what is in it.

{{fig classification_task}}

To avoid confusion, images supplied to a classifier need to be unambiguous, with a single subject dominating the image. Classification can be **binary** — a medical application might need only two classes, disease present or absent — or **multi-class**, as in waste classification for recycling.

Image classification was for a long time a very difficult computer vision problem. Deep convolutional neural networks changed that, with a substantial improvement in accuracy over the classical approaches, and by 2015 they had exceeded human-level performance on the 1000-class ImageNet challenge.

## Why it is hard

Real images vary enormously. The same object can be seen from different **viewpoints**, under different **illumination**, at different **scales**. Objects can be **deformable** — a dog appears in many different poses. One class can contain great **intra-class variation**, such as different breeds of dog. And objects can be **occluded** or lost against **background clutter**.

{{fig recognition_challenges}}

Deep network classifiers are capable of learning many complex and robust features from examples of each class, which is what allows them to cope with all of these challenges together.

## Classification with deep CNNs

Deep CNNs, composed of many convolution layers, are now the most widely used approach for most computer vision tasks. Their key property is **end-to-end learning**: simple and complex image features are learned automatically by the convolution layers, and the fully connected layers at the output combine them to classify the image. This is a large advantage over classical computer vision, where algorithms had to be developed by hand both to extract features and to form a representation that a classifier such as a support vector machine could discriminate.

There are disadvantages to weigh. Deep networks need **large amounts of data**, particularly when trained from scratch, and training large networks is **computationally demanding**, taking many hours of GPU time. Transfer learning (Lesson 2.4) mitigates both, by fine-tuning a pre-trained network with a much smaller dataset.

## Applications

High-accuracy classification has allowed visual inspection to be automated in many fields:

- **medical diagnostics** — diabetic retinopathy from retinal images, histopathology, skin lesions, and distinguishing tumours from benign nodules in ultrasound and CT;
- **species recognition** — the Stanford Dogs dataset has 120 breeds; invasive fish species can be identified at underwater monitoring stations;
- **aerial scene classification** — land use and crop types from satellite imagery;
- **agriculture** — disease on plant leaves;
- **disaster response** — classifying images posted to social media to locate and gauge an event;
- **environmental** applications such as sorting waste for recycling;
- and many more, in vehicle safety, manufacturing and logistics.

## Datasets and the ImageNet challenge

To train a deep CNN from scratch we usually start from a public dataset. Early datasets such as MNIST and CIFAR-10 were created before the deep learning era. They use small images and few classes, and networks trained on them do not perform well outside that narrow domain. They remain useful as benchmarks and for prototyping new networks without long training times.

**ImageNet** was the first dataset with enough classes and images to train genuinely general deep classifiers. Networks pre-trained on it are the usual starting point for transfer learning. A further advantage of the well-known datasets is that they are well **balanced**, with good representation of every class — which, as we shall see at the end of this lesson, matters.

ImageNet was not compiled for neural networks, but for the **Large Scale Visual Recognition Challenge**, created to promote new image classification technology.

{{fig ilsvrc_progress}}

## Metrics: the binary case

The most commonly used metrics for classification are **accuracy**, **precision** and **recall**, all computed from the **confusion matrix**.

{{fig binary_metrics}}

For a pedestrian classifier there are four possible outcomes, since there are two predicted labels and two actual labels:

- predicted pedestrian, actually pedestrian — **true positive** (TP);
- predicted pedestrian, actually no pedestrian — **false positive** (FP);
- predicted no pedestrian, actually pedestrian — **false negative** (FN), a missed detection;
- predicted no pedestrian, actually no pedestrian — **true negative** (TN).

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad
\text{precision} = \frac{TP}{TP + FP}, \qquad
\text{recall} = \frac{TP}{TP + FN}. \tag{1}$$

**Precision** is the proportion of positive predictions that are actually correct: high precision means few false positives, and high confidence in a positive prediction. **Recall** is the proportion of actual positives that were correctly identified: high recall means few missed detections.

## Metrics: the multi-class case

With more than two classes the same equations are used in a **one-versus-rest** manner: we compute $TP_i$, $FP_i$, $FN_i$ and $TN_i$ separately for each class $i$.

{{fig multiclass_metrics}}

Reading the confusion matrix for one class — say *cat* — the true positive is the diagonal cell where predicted and actual are both cat. The other cells in the **cat column** were predicted cat but are actually dog or mouse: those are false positives. The other cells in the **cat row** are actual cats predicted as something else: false negatives. Every other cell is a true negative for this class.

To combine the per-class values into a single figure there are two choices:

- the **macro** average gives every class equal importance, so a poorly performing class — perhaps one with few training examples — pulls the average down;
- the **weighted** average weights each class by the number of examples it has, so the overall score is dominated by the largest class.

For a balanced dataset the two are the same.

{{example con_mat seed=4}}

## Top-1 and top-5 accuracy

Deep network classifiers are probabilistic: they output a normalised probability for every class. Two accuracy metrics are associated with ImageNet.

{{fig top1_top5}}

**Top-1** accuracy is the conventional multi-class accuracy: the highest-probability prediction must match the actual label. Because that is a very stringent criterion for a 1000-class dataset with many similar classes, the less stringent **top-5** accuracy was introduced, which records a true positive if the actual label is among the five highest-probability predictions, in any order. Any classifier scores at least as well on top-5 as on top-1.

## Robustness, confusion and bias

Neural networks are trained on specific datasets for a given application, and three issues follow from that.

{{fig imbalance_and_shift}}

**Robustness.** A network can perform exceptionally well on test images from the same domain as its training set and poorly on images from a different one. A network trained only on daytime driving scenes may not work at night.

**Confusion** is a mislabelling error between similar classes — objects or scenes that share similar image features, such as a cello labelled as a violin, or a library labelled as a bookshop. Under-represented classes are more prone to it, because the network has had less opportunity to learn what distinguishes them.

**Bias** is a matter of great concern whenever a network is deployed. The network learns whatever biases are present in its training data. In one documented example, a network trained to classify the gender of people in images mislabelled women playing sport, because the training images of sports activities were heavily skewed towards men. The figure shows the underlying mechanism directly: the less a class is represented in training, the worse the network serves it.

{{selfcheck 02_01}}
