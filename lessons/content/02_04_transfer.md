---
id: "02_04"
title: Transfer learning
section: 2
engines: [s2_classification.transfer]
practice: []
summary: Reusing a network pre-trained on a large dataset — as a fixed or fine-tuned feature extractor, for classification and for new tasks such as detection — and the evidence on when it helps and when it does not.
---

## Learning outcomes

After this lesson you should be able to:

- explain the two problems transfer learning addresses: **too little data** and **too much computation**;
- describe how a network is split into a **feature extractor** and a **classifier**, and the difference between a **frozen** and a **fine-tuned** extractor;
- explain how transfer learning applies when the **task** changes as well as the data;
- summarise the evidence of **He et al.** and **Raghu et al.** on what pre-training does and does not achieve;
- decide, for a given application, **whether and how** to use transfer learning;
- instantiate a **pre-trained model** without its classifier.

## The problem transfer learning solves

Deep learning has become the dominant approach to computer vision through a confluence of developments: large amounts of training data from the big-data revolution, deep architectures together with the algorithmic tools to train them — gradient descent and backpropagation — and high-performance parallel computing in GPUs.

Deploying a deep learning solution to a new task or image domain commonly runs into two problems:

- there is **insufficient data** in the new domain to train a model to the required accuracy, which often happens in specialised domains such as medical imaging;
- even with enough data, the **computational cost** of training a deep model can be expensive or technically difficult.

Transfer learning has become a very popular solution to both, and is used extensively across machine and computer vision.

## Feature extractor and classifier

{{fig transfer_pipeline}}

Transfer learning rests on the idea that a deep network can be partitioned into a **feature extractor** and a **classifier**. As Lesson 2.5 shows, convolution layers respond to image features: early layers to simple ones such as lines, edges and blobs, deeper layers to more specific and complex ones. Trained on a large dataset such as ImageNet, a CNN is considered to have learned many **generic** image features.

Those pre-trained convolution layers can then be reused in a new application. In the simplest case they are used as a **fixed** feature extractor and only new classifier layers are learned. Usually, though, the feature extractor is **fine-tuned** as well, training both it and the new classifier together, which helps performance.

Transfer learning solves the problem of scarce target data by drawing on the generic features of the pre-trained layers. It also reduces the computation required, because the new system converges quickly to its final accuracy.

## Changing the task

So far only the data has changed: a classifier is retrained to recognise new classes, but the task is still classification. Transfer learning also works when **both the data and the task change**.

{{fig transfer_detection}}

For object detection, the pre-trained extractor's final feature maps carry spatial information about where objects are, which is enough to train a bounding-box predictor alongside a class predictor. Detection data must record the location of every object, which limits how much is available — and that limitation often makes transfer learning necessary to train a detector at all.

## Should we always use transfer learning?

Pre-training on ImageNet classification and transferring to the target task has become standard practice. Its popularity has prompted researchers to check the assumptions behind it.

**He and colleagues (2018)** made several important observations. Because transfer learning was so popular for tasks such as object detection, many designers had assumed that detection networks *could not* be trained from scratch. The authors showed that most can, without specialised architectures. They confirmed that **pre-training converges faster** — but that the accuracy it reaches is **no better** than training from scratch, which simply takes longer to get there. Training from scratch can fail to match transfer learning when the dataset is very small. And, counting the pre-training itself, transfer learning is not cheaper overall — we benefit from computation someone else has already paid for.

{{fig pretrain_convergence}}

**Raghu and colleagues** looked at specialised domains, where transfer learning is very commonly used. When the target images differ greatly from the pre-training data — as medical images do — transfer learning may not help as much as assumed. On a diabetic retinopathy task, ResNet-50 trained by transfer learning reached high accuracy, and the same architecture trained from scratch reached the same accuracy. More strikingly, lightweight architectures with far fewer parameters performed comparably when trained from scratch. Those small models do poorly on ImageNet, which they lack the capacity for, but they are complex enough for the retinopathy task. Designers often deploy a large model by transfer learning simply because pre-trained weights are readily available, without asking whether a lightweight model would suit the task better.

{{fig data_size_sweep}}

## Recommendations

{{fig decision_grid}}

When deciding whether to use transfer learning for a new application, consider the similarity of the source and target **domains**, the similarity of their **labels**, and the **size** of the source and target datasets:

- Transfer learning may be the **only** possible solution when computation or target data is limited.
- The most favourable case is **similar** source and target domains with a **large** source dataset. If the classes or objects are similar too, excellent performance can be had even without fine-tuning, and with a small target dataset it will likely be much better than training from scratch.
- With a **large** target dataset, transfer learning and training from scratch reach equivalent performance, but transfer learning saves training time.
- If the domains are **less similar**, fine-tuning is necessary for good accuracy, and a small target dataset risks overfitting the target domain, which can give worse performance than training from scratch.
- If the domains are **dissimilar**, as with medical images, training from scratch with a medium-sized dataset performs as well, though with a very small target dataset transfer learning still does slightly better.
- With **limited computational resources**, training from scratch may be impractical, which makes transfer learning a good option regardless — many state-of-the-art object detectors are trained quickly this way.

## Pre-trained models

{{fig pretrained_models}}

Keras's applications module contains many standard image recognition models — ResNet50, VGG16, DenseNet, MobileNetV2 and others — pre-trained and ready to use, either for classification as they are or as the basis for transfer learning. Instantiating one takes a single line, as in TensorFlow's transfer learning tutorial:

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                               include_top=False,
                                               weights="imagenet")
base_model.trainable = False      # freeze the extractor; set True to fine-tune
```

The `input_shape` argument lets the model's tensor sizes be calculated. `include_top=False` removes the fully connected classification part, leaving only the feature extractor. We can then freeze the extractor and train only new classification layers, or fine-tune the entire network.

{{selfcheck 02_04}}
