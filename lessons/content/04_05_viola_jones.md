---
id: "04_05"
title: The Viola-Jones algorithm
section: 4
engines: [s4_detection.viola_jones]
practice: [int_img, haar_feature]
summary: "Fast face detection from Haar-like rectangle features computed with the integral image, selected and combined by AdaBoost, and arranged in an attention cascade that rejects background windows early."
---

## Learning outcomes

After this lesson you should be able to:

- describe **Haar-like features** and how many arise inside a detector window;
- construct an **integral image** and use it to sum any rectangle with four lookups;
- explain how **AdaBoost** selects and combines weak classifiers;
- explain the **attention cascade** and why the algorithm is fast at detection but slow to train.

## The Viola-Jones algorithm

Viola-Jones is an extremely well-known object detector, associated above all with face detection in early digital cameras and phones, though it can be trained for other categories such as road signs. It is a feature-engineering approach with four key ideas: simple **Haar-like features** for discrimination, the **integral image** for fast feature computation, a **boosted classifier** that combines many simple classifiers, and a **cascade** of classifiers that speeds up detection. It is computationally efficient enough for embedded devices.

## Haar-like features

The detector window is usually 24 × 24 pixels and is slid across the image in an exhaustive search. Within it a large number of Haar-like features are evaluated. Each feature is a pattern of adjacent black and white rectangles: the feature value is the sum of the pixels under the white rectangles minus the sum under the black ones, and depending on the pattern it responds to an edge, a line or a diagonal contrast. Every feature type can be scaled and shifted anywhere in the window; applying all positions and scalings of the seven basic types to a 24 × 24 window gives over **180,000** candidate features. No single feature can detect a face; only a combination of very many can, which means a lot of computation.

{{fig haar_features}}

{{example haar_feature seed=3}}

## The integral image

Each feature needs the sum of the pixels under each of its rectangles: 4 additions for a 2 × 2 rectangle, 9 for 3 × 3, 16 for 4 × 4, and so on. To avoid this cost the authors introduced the **integral image**, also called the summed-area table. It is computed **once** per window in two passes: accumulate the sum down each column, then accumulate along each row of the result. Each entry then holds the sum of all pixels above and to its left, which is easy to check by direct summation. The sum of any rectangle, whatever its size, is obtained from just **four** entries at its corners,

$$\text{sum} = A - B - C + D, \tag{1}$$

so larger features cost no more than small ones.

{{fig integral_image}}

{{example int_img seed=6}}

## Classification with AdaBoost

Individually a rectangle feature is a **very weak binary classifier**: it merely detects a contrast difference somewhere in the window. During training only the best-performing weak classifiers, and hence features, are retained and combined using the **AdaBoost** algorithm. AdaBoost trains weak classifiers in sequence, each concentrating on the examples the previous ones misclassified, and combines their weighted votes into an accurate overall classifier. In Viola-Jones the pool of 180,000 features is reduced to about **6,000** selected ones, still a large number.

{{fig adaboost}}

### Interpreting the selected features

The rectangle features AdaBoost selects first are easily interpreted. The first exploits the fact that the eye region is darker than the top of the nose and the cheeks; the second that the eyes are darker than the bridge of the nose between them.

{{fig feature_on_face}}

## The attention cascade

Even after selection, evaluating all 6,000 features in every window is expensive, and most windows of an image contain no face at all. The remedy is a **cascade** of classifiers grouped into stages. In the original detector there are 38 stages, with 1, 10, 25 and 50 weak classifiers in the first four. Training is modified to build the cascade so that the classifiers with the **lowest false-negative rate**, those least likely to miss a face, come first. A window is rejected as soon as any stage rejects it, so empty and background regions are discarded after one or a few feature evaluations and no time is spent on the remaining stages.

{{fig cascade}}

## Training and use

Viola-Jones is **slow to train** but, because of its structure, **fast to run**. Training needs a labelled set of face and non-face patches; the complete set of features is evaluated on them to produce a very large set of weak classifiers; the modified AdaBoost selects a reduced set of the best and arranges them into a cascade. At test time the cascade is applied as a sliding-window detector, with an image pyramid for scale.

{{fig face_detection}}

{{selfcheck 04_05}}
