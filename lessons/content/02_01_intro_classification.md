---
id: "02_01"
title: Introduction to image classification
section: 2
engines: [s2_classification.intro_classification]
practice: [con_mat]
summary: The image classification task and its difficulties, applications, standard datasets and the ImageNet challenge, the metrics used to evaluate classifiers — accuracy, precision, recall, top-1 and top-5 — and robustness, confusion and bias.
---

## Learning outcomes

On completion of this lesson the student should be able to:

- describe the **image classification** task and explain why it is difficult;
- explain the advantages and disadvantages of **deep CNNs** for classification;
- name the **standard datasets** and describe the role of **ImageNet**;
- calculate **accuracy, precision and recall** from a binary confusion matrix;
- calculate these metrics **one class at a time** for a multi-class problem, and explain **macro** and **weighted** averaging;
- explain **top-1** and **top-5** accuracy;
- describe **robustness**, **confusion** and **bias** as failure modes of trained classifiers.

## The classification task

In image classification, an image is labelled according to its content.

{{fig classification_task}}

To avoid confusion, the images supplied to a classifier should be unambiguous, with a single subject dominating the image. Classification can be **binary**, for example a medical application in which only two classes are required (disease present or absent), or **multi-class**, as in the classification of waste for recycling.

For a long time image classification was a very difficult computer vision problem. A substantial improvement in accuracy over the classical approaches was achieved with deep convolutional neural networks, and by 2015 human-level performance on the 1000-class ImageNet challenge had been exceeded.

## Difficulties

Real images vary greatly. The same object can be seen from different **viewpoints**, under different **illumination** and at different **scales**. Objects can be **deformable**; for example, a dog can appear in many different poses. A single class can contain large **intra-class variation**, such as different breeds of dog, and objects can be **occluded** or obscured by **background clutter**.

{{fig recognition_challenges}}

Many complex and robust features can be learned by deep network classifiers from examples of each class, and this allows these challenges to be addressed together.

## Classification with deep CNNs

Deep CNNs, composed of many convolution layers, are now the most widely used approach for most computer vision tasks. Their key property is **end-to-end learning**: simple and complex image features are learned automatically by the convolution layers, and these features are combined by the fully connected layers at the output to classify the image. This is a significant advantage over classical computer vision, in which algorithms had to be developed by hand both to extract features and to form a representation that could be discriminated by a classifier such as a support vector machine.

There are also disadvantages. **Large amounts of data** are required to train deep networks, particularly when they are trained from scratch, and the training of large networks is **computationally demanding**, requiring many hours of GPU time. Both of these disadvantages are reduced by transfer learning (Lesson 2.4), in which a pre-trained network is fine-tuned with a much smaller dataset.

## Applications

High-accuracy classification has allowed visual inspection to be automated in many fields:

- **medical diagnostics**, including diabetic retinopathy from retinal images, histopathology, skin lesions, and the distinction of tumours from benign nodules in ultrasound and CT;
- **species recognition**, for example the 120 breeds of the Stanford Dogs dataset, or the identification of invasive fish species at underwater monitoring stations;
- **aerial scene classification**, such as land use and crop types from satellite imagery;
- **agriculture**, such as disease on plant leaves;
- **disaster response**, in which images posted to social media are classified to locate and assess an event;
- **environmental** applications such as the sorting of waste for recycling;
- many others in vehicle safety, manufacturing and logistics.

## Datasets and the ImageNet challenge

When a deep CNN is trained from scratch, a public dataset is usually used. Early datasets such as MNIST and CIFAR-10 were created before the deep learning era. They contain small images and few classes, and networks trained on them do not perform well outside that narrow domain. They remain useful as benchmarks and for the prototyping of new networks without long training times.

**ImageNet** was the first dataset with enough classes and images for general deep classifiers to be trained. Networks pre-trained on ImageNet are the usual starting point for transfer learning. A further advantage of the well-known datasets is that they are well **balanced**, with good representation of every class; the importance of this is discussed at the end of this lesson.

ImageNet was not compiled for neural networks, but for the **Large Scale Visual Recognition Challenge**, which was created to promote new image classification technology.

{{fig ilsvrc_progress}}

## Metrics: the binary case

The most commonly used metrics for classification are **accuracy**, **precision** and **recall**, all of which are calculated from the **confusion matrix**.

{{fig binary_metrics}}

For a pedestrian classifier there are four possible outcomes, as there are two predicted labels and two actual labels:

- predicted pedestrian, actually pedestrian: **true positive** (TP);
- predicted pedestrian, actually no pedestrian: **false positive** (FP);
- predicted no pedestrian, actually pedestrian: **false negative** (FN), a missed detection;
- predicted no pedestrian, actually no pedestrian: **true negative** (TN).

$$\text{accuracy} = \frac{TP + TN}{TP + TN + FP + FN}, \qquad
\text{precision} = \frac{TP}{TP + FP}, \qquad
\text{recall} = \frac{TP}{TP + FN}. \tag{1}$$

**Precision** is the proportion of positive predictions that are correct; a high precision indicates few false positives and high confidence in a positive prediction. **Recall** is the proportion of actual positives that are correctly identified; a high recall indicates few missed detections.

## Metrics: the multi-class case

When there are more than two classes, the same equations are applied in a **one-versus-rest** manner, and $TP_i$, $FP_i$, $FN_i$ and $TN_i$ are calculated separately for each class $i$.

{{fig multiclass_metrics}}

When the confusion matrix is read for one class, for example *cat*, the true positive is the diagonal cell in which the predicted and actual labels are both cat. The other cells in the **cat column** were predicted as cat but are actually dog or mouse, and are false positives. The other cells in the **cat row** are actual cats that were predicted as another class, and are false negatives. All remaining cells are true negatives for this class.

The per-class values can be combined into a single value in two ways:

- in the **macro** average every class is given equal importance, so the average is reduced by a poorly performing class, for example one with few training examples;
- in the **weighted** average each class is weighted by its number of examples, so the overall score is dominated by the largest class.

For a balanced dataset the two averages are the same.

{{example con_mat seed=4}}

## Top-1 and top-5 accuracy

Deep network classifiers are probabilistic, and a normalised probability is output for every class. Two accuracy metrics are associated with ImageNet.

{{fig top1_top5}}

**Top-1** accuracy is the conventional multi-class accuracy, in which the highest-probability prediction must match the actual label. As this is a very stringent criterion for a 1000-class dataset with many similar classes, the less stringent **top-5** accuracy was introduced, in which a true positive is recorded if the actual label is among the five highest-probability predictions, in any order. The top-5 accuracy of a classifier is always at least as high as its top-1 accuracy.

## Robustness, confusion and bias

Neural networks are trained on specific datasets for a given application, and three issues arise as a result.

{{fig imbalance_and_shift}}

**Robustness.** A network may perform exceptionally well on test images from the same domain as the training set, and poorly on images from a different domain. For example, a network trained only on daytime driving scenes may not work at night.

**Confusion** is a mislabelling error between similar classes, that is, objects or scenes with similar image features, such as a cello labelled as a violin, or a library labelled as a bookshop. Under-represented classes are more prone to confusion, as there has been less opportunity for the features that distinguish them to be learned.

**Bias** is of great concern when a network is deployed, as any bias present in the training data is learned by the network. In one documented example, women playing sport were mislabelled by a network trained to classify the gender of people in images, as the training images of sports activities were heavily skewed towards men. The underlying mechanism is shown in the figure: the less a class is represented in training, the poorer the performance of the network on that class.

{{selfcheck 02_01}}
