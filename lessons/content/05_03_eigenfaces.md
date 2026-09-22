---
id: "05_03"
title: Eigenfaces
section: 5
engines: [s5_faces.eigenfaces]
practice: [pca_project]
summary: "Applying PCA to a database of faces: why raw pixels fail, what eigenfaces look like, how a face becomes a short weight vector, and how recognition becomes a nearest-neighbour search in eigenface space."
---

## Learning outcomes

After this lesson you should be able to:

- explain why a **naive** raw-pixel approach to facial recognition fails;
- describe what **eigenfaces** are and how a face is represented as a **weight vector**;
- state the **enrolment** and **query** algorithms of the Eigenface method;
- give the advantages of the method and its main limitation.

## Naive facial recognition

In object detection and classification we train a classifier with many examples of each class. Facial recognition has the opposite problem: many classes, one per person, and often very few examples of each, possibly only one. The recognition stage therefore often uses no classifier at all, relying instead on **brute-force matching** of the kind used to match keypoints between images.

The naive approach uses **all the pixels** of the detected face as a vector and compares it against the equivalent vectors of every face in the database, declaring a match when the Euclidean distance is smallest. On a database as small and as tightly constrained as Olivetti this still classifies most faces correctly, but it is a poor representation. The dimensionality is very high, there are many classes and very few samples per class, which is exactly the curse of dimensionality, and the approach breaks down as the database grows and the imaging conditions loosen. The naive approach is also the **slowest** possible, since the vectors are long. We cannot reduce the number of faces in the database, but we can shorten the vector that represents each one: the shorter it is, the faster the distances are computed and the less storage the database needs.

{{fig naive_fr}}

## Eigenfaces

The **Eigenface** representation reduces the dimensionality of a database of facial images. Eigenfaces are the set of image vectors $V$ obtained from a **principal component analysis** of the database of faces, so they are the eigenvectors of the sample covariance matrix of the face vectors. Each has the same length as a face vector, so it can be rearranged and displayed as an image.

The compression property of PCA is used to represent all the faces in the database with a reduced set of eigenvectors, and the representation is used **directly for recognition**, either by comparing distances between vectors or with a classifier. The advantages are a reduction in database storage, since compressed faces are stored, and a reduction in computation compared with using image pixels.

{{fig eigenfaces_gallery}}

### Representing faces in a database

Because the eigenfaces are the principal components of the faces in the database, each face vector can be transformed into a compressed representation by multiplying it by the set of eigenfaces. That principal-component transformation can be interpreted as building each face from a **weighted combination of eigenfaces**: each face vector $x_i$ is converted by multiplication with the $k$-eigenface matrix $V_k^{T}$ into a one-dimensional vector

$$w_i = V_k^{T}(x_i - \mu), \tag{1}$$

which is a set of coordinates in the $k$-dimensional eigenface feature space. Each value in $w_i$ is the position along one eigenface dimension, and the vector as a whole is the set of **weights** for that face. The face itself is approximated by

$$x_i \approx \mu + \sum_{j=1}^{k} w_{ij}\, v_j. \tag{2}$$

{{fig face_as_weights}}

{{fig face_reconstruction}}

{{example pca_project seed=4}}

## The Eigenface database algorithm

To convert a database of $N$ faces to its Eigenface representation:

1. each image in the database is a vector, for example 64 × 64 = 4096 elements long;
2. as in regular PCA, the **mean** is subtracted from each sample vector; in this case the mean is the **average face** of the database;
3. the sample covariance matrix is obtained from the $N$ face vectors;
4. its eigenvalues and eigenvectors are determined; the eigenvectors are the **eigenfaces**. There are at most $N - 1$ of them, and only $K$ are retained;
5. each image in the database is multiplied by the matrix of $K$ eigenvectors, yielding a weight vector $w$. The database of $N$ weight vectors is the compressed representation of the original images.

To recognise a query image against such a database, the query is converted to a vector representation by multiplication with the $K$ stored eigenfaces, and the transformed query vector is compared against the $N$ database vectors. The **minimum distance** is taken to be a match and a label is output. Unknown faces can be rejected with a criterion like the one used in feature matching: if the difference in distance between the closest and the next closest vector is too small, the match is rejected.

{{fig weight_space}}

{{fig eigenface_database}}

## Summary

The Eigenface method is one of the most basic approaches to facial recognition. Because it is based on principal component analysis, which is a method of **compression**, it may be considered optimal for **reconstruction** but genuinely non-optimal for **discrimination** between faces, a point the next two lessons address. As with many facial-recognition methods, it requires the faces in the training and test databases to be centred and the same size. The method can be extended to be more robust to pose by learning representations for different angles, but this requires a detailed database. Interestingly it can also be used for facial **expression** recognition.

{{selfcheck 05_03}}
