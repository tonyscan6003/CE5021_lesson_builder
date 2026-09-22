---
id: "06_01"
title: The bag of features representation
section: 6
engines: [s6_bof.bag_of_features]
summary: "An orderless, high-level description of an image: the bag of words and tf-idf weighting, then building a visual vocabulary by clustering descriptors, assigning terms and forming the term histogram."
---

## Learning outcomes

After this lesson you should be able to:

- explain what makes the bag of features an **orderless** representation, and what that costs;
- compute a **tf-idf** weight and say why common words must be down-weighted;
- state the three stages of the bag of features algorithm;
- compare **keypoint**, **random** and **dense** feature sampling for building a vocabulary;
- describe how **clustering** produces visual words, and what the **frequency problem** is.

## Introduction

The **bag of features** is a high-level representation of an image that can be used for image **classification** or image **retrieval**. It is based on the same principle as the bag of words representation used for matching documents.

It is an **orderless** representation. Large-scale spatial information, relative scale and orientation information are all discarded in constructing it. Despite that loss the representation still works very well for image classification and scene recognition. It is **not** suited to object detection and localisation, because a representation that has thrown away where things are cannot say where an object is.

## Bag of words

Using the bag of words representation, similar documents are matched by comparing the frequency of the words used in each one. The bag is a list of the words that appeared in a document together with how often they appeared. It is unordered, so sentence structure is not captured, only word frequency.

Very common words such as "the", "and" and "it" have a high frequency in every document. Their frequency must be penalised or reduced so that the more discriminative words stand out.

{{fig bag_of_words}}

### tf-idf

A weighting is applied to each word to reduce the effect of the common ones. The usual choice is **tf-idf**, term frequency-inverse document frequency. A document $d$ is represented by a vector $V_d = (t_1, \ldots, t_i, \ldots, t_k)$ of weighted word frequencies, where

$$t_i = \frac{n_{id}}{n_d} \log \frac{N}{n_i}. \tag{1}$$

The expression has two parts:

- the **term frequency** $n_{id}/n_d$, the number of occurrences of word $i$ in document $d$ divided by the total number of words in that document;
- the **inverse document frequency** $\log(N/n_i)$, where $N$ is the total number of documents in the database and $n_i$ is the number of them that contain word $i$.

When a word occurs in a large fraction of the database the second term is small, so that word is down-weighted. A word that occurs in **every** document has $n_i = N$ and therefore a weight of exactly zero.

{{fig tf_idf}}

## The bag of features algorithm

The algorithm has three stages.

1. **Build a visual vocabulary.** The visual vocabulary is a list of visually similar image patches drawn from all the images in a database. These **visual words**, or **codewords**, are the equivalent of the words in a bag of words. The vocabulary is built by detecting or sampling image features across the whole database, describing them with a descriptor such as SIFT, and then applying a clustering algorithm to group similar descriptors together. Each cluster centre is one visual word.
2. **Assign terms.** Features are extracted and described from a new image, and each descriptor is matched to the nearest visual word in the vocabulary. The image becomes a list of codewords.
3. **Generate the term vector.** The counts of each individual visual word in the image form a **histogram**. As with the bag of words, a weighting can be applied to reduce the effect of very common visual words.

{{fig bof_algorithm}}

## Building the visual vocabulary

### Which features?

Image patches around **keypoints** are the obvious first choice. The keypoints correspond to blob regions or corners depending on the detector. They are sparse and well localised, but they are also a disadvantage: when the pose, viewpoint or illumination of an object changes, the detector fires in different places, so the same object yields a **different set of visual words**.

Features can instead be extracted **randomly** or by **dense sampling** on a uniform grid. Both generate many more features than a keypoint detector does, and dense sampling puts them in positions that do not depend on the image content at all. That increases the likelihood of finding a common set of visual words for an object photographed in a different pose or under different light.

Whichever way the features are chosen, the patch around each one is converted into a descriptor. The **SIFT** descriptor has been found to be a better input to the clustering stage than simpler descriptors or raw pixels.

{{fig keypoint_vs_dense}}

### Clustering

After describing the patches, a clustering algorithm finds cluster centres that correspond to visually similar image features. These are the visual words. Clustering is what **limits** the number of distinct features in the representation to something manageable.

**k-means** is the default choice. As Jurie and Triggs showed, it is not necessarily the best, particularly when features are densely sampled: the feature space then contains a large number of very common features, and less frequent but more discriminative features may fail to form distinct clusters and be lost from the representation.

{{fig descriptor_clustering}}

### The final vocabulary

With a large dataset of training images a vocabulary of visual words can be created. It generally contains a wide variety of edges, lines and stripes together with some more complex features.

The frequency problem of the bag of words occurs here too. Common features such as plain edges can dominate the vocabulary while carrying almost no discriminative information, because they are seen in every image. Ideally we want visual words of **intermediate frequency**: common enough to recur, rare enough to discriminate.

Note that the vocabulary is formed by clustering, which is **unsupervised learning**. No labelling or annotation of the training images is needed to create it.

{{fig visual_vocabulary}}

## Assigning terms and the final histogram

A bag of features can now be determined for any image using the vocabulary. Features are detected or sampled, descriptors are computed for the regions around them, and each descriptor vector is matched to the vocabulary to find which cluster it belongs to. The frequency of each visual word in the image forms a histogram.

Every image is therefore represented by a histogram, and with a large vocabulary that histogram is **sparse**: many words have a count of zero. The histogram can be structured in three ways:

- **raw counts** of each visual word;
- **tf-idf** weighting, exactly as for documents;
- a fully **binary** representation, used particularly for large sparse vectors, where the presence of a word is more informative than its actual count.

{{fig bof_histogram}}

The next lesson uses this representation for classification and for large-scale image retrieval.

{{selfcheck 06_01}}
