---
id: "06_02"
title: Classification and image retrieval with bag of features
section: 6
engines: [s6_bof.classification_retrieval]
summary: "Training an SVM on term vectors, putting spatial information back with the spatial pyramid, and large-scale retrieval with inverted file indexes and vocabulary trees."
---

## Learning outcomes

After this lesson you should be able to:

- describe how a bag of features feeds an **SVM** classifier, and what overhead the representation adds;
- explain the **spatial pyramid** and why it improves classification;
- distinguish bag-of-features retrieval from **content-based** retrieval;
- explain what an **inverted file index** is for and how a query uses it;
- explain how a **vocabulary tree** makes retrieval scale to a million images;
- summarise the strengths and weaknesses of the representation.

## Classification

A **support vector machine** can be trained to perform binary image classification using the bag of features representation. An SVM with a **radial basis function** kernel is typical, because it allows a more complex decision boundary to be learnt in the histogram space.

Using the bag of features carries the overhead of having to build the **visual vocabulary** before any classification can be done. A large dataset of images is gathered, features are detected and described, k-means or another clustering algorithm is applied to all the descriptor vectors, and the cluster centres are found. The labelled training images are then converted into their bag of features representation using that vocabulary, and those term vectors are sent to the SVM, which learns to separate the binary classes.

The bag of features was a very successful approach to building image classification systems, and it was not surpassed in performance until deep neural networks arrived in 2012.

{{fig bof_classification}}

## Bag of features with a spatial pyramid

The bag of features is orderless and therefore ignores spatial information. As Lazebnik and Schmid demonstrated, re-introducing spatial information with the **spatial pyramid** method improves classification performance.

The image is divided into a series of spatial grids. A bag of features histogram is formed for **each grid cell**, and a much longer term vector than usual is formed by concatenating and weighting the histograms at each level of the pyramid. Finer levels are weighted more heavily, since a match in a small cell is a stronger piece of evidence than a match anywhere in the whole image.

With this spatial information the **pyramid match kernel** is used in the SVM instead of the radial basis function. The spatial pyramid is an important concept that is still used in some vision networks today.

{{fig spatial_pyramid}}

## Image retrieval

The bag of features also has a powerful use in large-scale image **search and retrieval**. The goal is to find images in a database that are similar to a query image. Bag-of-features retrieval returns similar images based on the similarity of their final term vectors.

This is an **unsupervised** approach, so it can be used on any dataset without labelling it, although there may be problems with generalisation when querying with images from outside the training distribution. The main alternative is **content-based** image retrieval, which classifies the images and objects directly and then looks for images with the same label. That is a **supervised** approach, because the training data has to be labelled in advance.

{{fig retrieval_task}}

### The inverted file index

The key difficulty is matching the query term vector against the database. For very large databases a standard algorithm such as k nearest neighbours quickly becomes far too expensive.

The standard solution is to build an **inverted file index**. When each image is converted to its bag of features representation we have a histogram giving the number of occurrences of each visual word. The inverted file index presents the same information the other way round: a table showing, for each visual word, which images in the database contain it.

To match a query image, convert it to its bag of features representation and look up its words in the table. Images that share many visual words with the query are potential matches. The match can then be refined by applying k nearest neighbours to just that set of **candidate** images, rather than to the whole database.

{{fig inverted_index}}

### Video Google

The best-known example of image retrieval with the bag of features is the **Video Google** paper by Sivic and Zisserman. There the database is a series of frames from a film, and the goal is to find occurrences of a particular static object in specific frames.

A visual vocabulary is created from the frames of the film, using image regions around keypoints, and all frames are converted to their bag of features representation. To find a specific object, the visual words in the object region of a query image are found, the inverted file index is used to gather relevant frames containing similar visual words, and within that candidate set the full word counts are used for refinement. Video Google adds one further step that improves object recognition: a **spatial consistency** check that the matching visual words lie close together in the frame and therefore correspond to an object.

### Scaling up

Video Google used 4,000 images drawn from video frames and a vocabulary of 10,000 visual words, so it is not really a very large-scale retrieval system. Nister and Stewenius introduced **hierarchical clustering** to represent the vocabulary as a **tree**. This allows much larger vocabularies, and only a small percentage of the database needs to be searched when the database is large. The approach achieves sub-second retrieval over a million database images.

The database of descriptors is first clustered by k-means into $k$ clusters. Each of those clusters is then broken into $k$ more, and the process repeats until a stopping level is reached or the cluster size becomes one. During retrieval a query image is directed quickly through the tree by finding the best match with the cluster centres at each level, so the number of comparisons is proportional to $k$ times the depth rather than to the size of the vocabulary. A similarity score is computed as the hierarchy is traversed, using the inverted files, and that score ranks the possible matches.

{{fig vocab_tree}}

## Summary

The bag of features provides a compact summary of image content. It is flexible with respect to viewpoint changes, achieves good results in practice, and was the state of the art in image recognition before deep learning.

It has disadvantages. There is **no semantic meaning** for the vocabulary: the system does not know what the individual visual words represent. It also **ignores spatial information**, although the spatial pyramid recovers some of it and improves accuracy for classification.

It is worth noting the similarity between the bag of features and the first generation of neural networks. The early stages of a convolutional neural network learn simple filters that are visually similar to the visual words in a bag of features vocabulary.

{{fig bof_summary}}

{{selfcheck 06_02}}
