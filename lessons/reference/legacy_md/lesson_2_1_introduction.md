# lesson_2_1_introduction

Deep Learning for Computer vision 

Introduction to Image Classification with Deep Neural Networks Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Image Classification Task 

- We would like to be able to label each image according to what is actually present in the image. 

   - Each Image is assigned a “Class” label. 

   - We can have binary or multiple classes. 

- This is a very challenging Computer Vision problem! 

   - Deep Convolutional Neural Networks achieve Human level performance. 

“Cat”<sup>(i)</sup> 

“Ship”<sup>(iii)</sup> 

“Dog”<sup>(ii)</sup> 

“Car”<sup>(iv)</sup> 

   - (a) Example of Image Classification 

- (i)  “Agathe <u>the cat”, Author: Pierre Tribhou, License:</u> **CC BY-SA 4.0** 

- (ii)  “Standing Yellow Labrador Retriever dog”, Author: Djmirko, License: **CC BY-SA 3.0** 

- (iii)  “Atlas Maritimes’s Mitera Marigo Aframax Oil Tanker”, Author: PennyKall, License: **CC BY-SA 3.0** 

- (iv) “2001 Chrysler Voyager”, Author: Karrmann, License: **CC BY-SA 3.0** 

©Dr. Tony Scanlan 

# Image Recognition Challenges 

- Multi Views 

- Illumination 

- Scale 

- Occlusion 

- Background Clutter 

- Deformable Objects 

- Intra Class Variation 

- (a) Two Views of the Statue of Liberty demonstrating changes in Scale, Illumination and view point (Original Images (v) & (vi)). 

- (v) “Statue of Liberty NY”, Author: William Warby, License: **CC BY 2.0** 

- (vi) “ <u>The Statue of Liberty at night, seen from the Staten Island Ferry”, Author: 0x010C, License:</u> **CC BY-SA 4.0** (vii) “Breeds of dog from The Yuzhakov's Bolshaya Enc. (1904) ”, Author: Unknown, License: Public Domain 

- (b) Example Image<sup>(vii)</sup> Demonstrating several Image Recognition Challenges! 

©Dr. Tony Scanlan 

# Image Classification with CNNs 

- <u>Learns End to End.</u> 

   - Many Simple & also Complex Features are . 

   - <u>automatically learned</u> 

- Requires large amounts of data to train Networks. 

- Computationally Demanding 

- Transfer Learning 

- (a) Convolutional Neural Network 

- (b) Features Learned by early and deep layers in a Deep Neural Network [1]. 

[1] <u>Mahendran, A. and Vedaldi, A., 2016. Visualizing deep convolutional neural networks using natural pre-images.</u> _<u>International Journal of Computer Vision</u>_ <u>,</u> _<u>120</u>_ <u>(3), pp.233-255.</u> 

©Dr. Tony Scanlan 

# Applications 

- Image Classification has many varied Impactful Applications 

- (a) Diabetic Retinopathy Detection (x) 

(b) Dog Breed Classification (xi) 

- Medical Diagnostics 

      - Retinopathy, Histopathology, Radiology 

   - Species Classification 

   - Aerial Scene Classification 

- (x) Diabetic Retinopathy Detection (Kaggle) 

(xi) Stanford Dogs Dataset 

[3]     Xia, G.S., Hu, J., Hu, F., Shi, B., Bai, X., Zhong, Y., Zhang, L. and Lu, X., 2017. AID: A benchmark data set for performance evaluation of aerial scene classification. _IEEE Transactions on Geoscience and Remote Sensing_ , _55_ (7), pp.3965-3981. 

(c) Aerial Scene Classification [3] 

©Dr. Tony Scanlan 

# More Applications 

- Agriculture 

   - Plant Disease Detection 

- Social Media 

   - Disaster Type Detection 

- Environmental 

   - Waste Classification 

[4] Paymode, A.S. and Malode, V.B., 2022. Transfer learning for multicrop leaf disease image classification using convolutional neural network VGG. _Artificial Intelligence in Agriculture_ , _6_ , pp.23-33. [5] Alam, F., Ofli, F., Imran, M., Alam, T. and Qazi, U., 2020, December. Deep learning benchmarks and datasets for social media image classification for disaster response. In _2020 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM)_ (pp. 151-158). IEEE. 

[6] Zhang, Q., Yang, Q., Zhang, X., Bao, Q., Su, J. and Liu, X., 2021. Waste image classification based on transfer learning and convolutional neural network. _Waste Management_ , _135_ , pp.150-157. 

- (a) Leaf Disease Detection [4] 

(b) Disaster Type Detection [5] 

(c) Waste Classification [6] 

©Dr. Tony Scanlan 

# Public Datasets 

- Used as Benchmarks 

   - MNIST 

   - CIFAR-10/100 

   - Oxford_iiit_pet 

(a) MNIST Handwritten Digit Dataset<sup>(xii)</sup> 

   - (b) CIFAR-10 Dataset<sup>(xiii)</sup> 

- ImageNet 

- ImageNet Used to train High Accuracy Neural Networks • Huge number of Classes with wide variety of images. 

   - Transfer Learning to fine-tune for new applications 

(xii) MNIST Dataset 

- (c) ImageNet Dataset Samples from 4 Classes [7]. 

(xiii) CIFAR-10 & CIFAR 100 Datasets 

[7] ImageNet: A large-scale hierarchical image database 

©Dr. Tony Scanlan 

# ImageNet LSVRC 

- Large Scale Image recognition (1000 Classes). 

- Driver of Development of Novel CNN Architectures [8,9]. 

   - Evolution of Deep Convolution Neural Networks (2013-2017). 

   - Out Perform Classical Approaches – Achieve Human Level Performance 

[8] <u>Rawat, W. and Wang, Z., 2017. Deep convolutional neural networks for image classification: A comprehensive review.</u> _<u>Neural computation</u>_ <u>,</u> _<u>29</u>_ <u>(9), pp.2352-2449.</u> [9] Chen, L., Li, S., Bai, Q., Yang, J., Jiang, S. and Miao, Y., 2021. Review of image <u>classification algorithms based on convolutional neural networks.</u> _Remote Sensing_ , _13_ (22), p.4712. 

- (a) ImageNet: Large Scale Visual Recognition Challenge. 

- (b) Performance of Networks on ImageNet 2012-2017<sup>(xiv)</sup> 

(xiv) New Vision Technologies For Real-World Applications 

©Dr. Tony Scanlan 

# Metrics (Binary Classification) 

• Accuracy (Standard measurement). 𝑁𝑢𝑚𝑏𝑒𝑟𝑜𝑓 𝑐𝑜𝑟𝑟𝑒𝑐𝑡𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 𝑇𝑃+ 𝑇𝑁 = **(1)** 𝑇𝑜𝑡𝑎𝑙𝑁𝑜. 𝑃𝑟𝑒𝑑𝑖𝑐𝑡𝑖𝑜𝑛𝑠 𝑇𝑃+ 𝑇𝑁+ 𝐹𝑃+ 𝐹𝑁 

- Precision and Recall 

• Precision: Proportion of Positive Predictions actually correct. 𝑇𝑃 **(2)** 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑃+ 𝐹𝑃 

• Recall: Proportion of Positives correctly identified. 𝑇𝑃 𝑅𝑒𝑐𝑎𝑙𝑙= **(3)** 𝑇𝑃+ 𝐹𝑁 

(a) Confusion Matrix Binary Classification (Pedestrian Detection as example) 

©Dr. Tony Scanlan 

# Metrics (Multi–Class) 

• Metrics Calculated as “One vs Rest” for each ( 𝑖<sup>𝑡ℎ</sup> ) Class 𝑇𝑃𝑖+𝑇𝑁𝑖 **(1)** • 𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦𝑖 = 𝑇𝑃𝑖+𝑇𝑁𝑖+𝐹𝑃𝑖+𝐹𝑁𝑖 𝑇𝑃𝑖 • 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛𝑖 = **(2)** 𝑇𝑃𝑖+𝐹𝑃𝑖 𝑇𝑃𝑖 • 𝑅𝑒𝑐𝑎𝑙𝑙𝑖 = **(3)** 𝑇𝑃𝑖+𝐹𝑁𝑖 Weighting Term • Calculate Overall Averages • Macro Average 1𝑙<sup>σ</sup> 𝑙<sup>𝑖=1</sup> 𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦𝑖 **(4)** 1 𝑙 • Weighted Average 𝑙<sup>σ𝑖=1</sup> 𝜇𝑖𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦𝑖 **(5)** 

- Macro Average score affected by poor performing classes, where have few examples (Class Imbalance) 

- Weighed Average dominated by Largest Classes 

[10] Sokolova, M. and Lapalme, G., 2009. A systematic analysis of performance measures for <u>classification tasks.</u> _Information processing & management_ , _45_ (4), pp.427-437. 

<!-- Start of picture text -->
(a)<br><!-- End of picture text -->

**(b)** 

Multi-Class Confusion Matrix. Example of determining TP/FP/FN/TN for (a) the “Cat” Class, (b) the dog class. 

©Dr. Tony Scanlan 

# Top 1 and Top 5 Accuracy 

- Associated Accuracy Metrics with ImageNet (Multi-class) 

- Top 1: Conventional Accuracy 

   - TP: Highest probability prediction must match ground truth 

- Top 5: Consider 5 highest probability predictions • TP: Any of 5 highest probability predictions match ground truth 

(a) Comparison of two classifiers with Test image. Classifier A correctly classifiers the image for both Top 1 and Top 5 cases. Classifier B, correctly classifies using Top accuracy metric.<sup>(XV)</sup> 

(xv) Bengal Tiger, Author: Jon & Karen Hollingsworth, License Public Domain 

©Dr. Tony Scanlan 

# Robustness, Confusion & Bias 

- Exceptional Performance on images from training Domain 

   - May not be robust under different Conditions (E.g. Illumination) 

   - (a) Two Driving Scenes, demonstrating widely different illumination. (Original Images (viii) & (ix)). 

- Confusion 

   - Network may mislabel objects with similar features 

   - Underrepresented Classes more affected 

- Bias 

   - (b) Confusion in Classifications (Cello labelled as Violin, Library labelled as bookshop). 

- Networks may learn bias from training data 

(viii) “Cars driving at night”, Author: Pink Sherbert Photography, License: **CC BY 2.0** (ix) “Driving down the highway”, Author: bluesby, License: **CC BY 2.0** [2] Tian, Y., Zhong, Z., Ordonez, V., Kaiser, G. and Ray, B., 2020, June. <u>Testing DNN image classifiers for confusion & bias errors. In</u> _Proceedings of the acm/ieee 42nd international conference on software engineering_ (pp. 1122-1134). 

- (c) Women mislabelled as Men in gender image captioning task due to Network bias [2].
