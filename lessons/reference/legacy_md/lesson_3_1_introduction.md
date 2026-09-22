# lesson_3_1_introduction

Deep Learning for Computer vision 

## Introduction to Object Detection with Deep Learning Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

Object Detection with Deep Learning 

- More Challenging than Image Classification. 

   - Object Localisation required. (Predict Class & Bounding box) 

   - Multiple Objects in one image. 

- Naïve Approach • Use Pre-trained Image Classifier with a sliding window. 

   - => Too computationally expensive & Slow (+ Need image pyramid for scale) 

- (a) Object Detection: Predicting classes and bounding boxes 

- (b) Sliding Window Approach to Object Detection. 

[1] Liu, L., Ouyang, W., Wang, X., Fieguth, P., Chen, J., Liu, X. and Pietikäinen, M., 2020. <u>Deep learning for generic object detection: A survey.</u> _International journal of computer vision_ , _128_ (2), pp.261-318. 

©Dr. Tony Scanlan 

# Object Detection with Region Proposals 

- Sliding window approach leads to too many calls to the Image Classifier. 

   - How can the number of candidate windows be reduced? 

      - (a) Example Image<sup>(i)</sup> with Region Proposals 

- => Region Proposal Algorithm. • Identify regions of image likely to contain an object. 

   - Image Segmentation fundamental to identifying region proposals 

<!-- Start of picture text -->
Classify<br>Region<br>“Koala”<br><!-- End of picture text -->

   - (b) Segmentation aiding object detection (Original Image<sup>(ii)</sup> ) 

- “A commuter cyclist in the London morning rush hour”, Author: ProfDEH, License: **CC BY-SA 3.0** 

“Combining Detection Recognition and Segmentation”, Ira Kemelmacher-Shlizerman, Sharon Alpert, Washington Univerisity 2006. 

(i) (ii) 

# ©Dr. Tony Scanlan Basic DL based Object Detection with Region Proposals 

- Region Proposal Algorithm Characteristics: • Produces less windows to classify. 

   - Finds objects of any shape/scale. 

   - Low number of False Positives. (Precision) 

   - Low no False Negatives. (Recall). 

- Problems. • Region Proposal Trained Independently 

   - Region Proposals Different Sizes 

   - Image Classifier Operates on overlapping regions several times. 

- (a) Basic Deep Learning Object Detection pipeline with Region Proposals and pre-trained Image Classifier. 

©Dr. Tony Scanlan 

# Object Recognition Datasets 

- PASCAL VOC<sup>(iii)</sup> 

   - Object Class recognition/detection (2005-2012) 

   - 20 Categories (11530 Images, 27450 objects ROI) 

- IMAGENET<sup>(iv)</sup> 

   - 2011-2014: Single object localisation (1000 classes, 14M images) 

   - 2014+ Object Detection localising multiple objects in one image (200 Classes, 450K Training). 

- MS COCO<sup>(v)</sup> 

   - Real world scenes (objects not centered) 

   - 91 object categories, 7.3 objects per image. (330K images, 1.5 Million object instances.) 

   - Instance Segmentation as well as object detection. 

- Open Image<sup>(vi)</sup> 

   - 9 million Images, 6000 categories. 

   - Instance segmentation as well as object detection. 

- (a) Object Detection in ImageNet<sup>(iv)</sup> 

- (b) Segmentation Masks in Open Image<sup>(vi)</sup> 

(iii) The Pascale Visual Object Classes Homepage (iv) ImageNet Database & <u>ImageNet large Scale Visual Recognition Challenge</u> (v) COCO Dataset (vi) Open Images DatasetV6 + Extensions 

©Dr. Tony Scanlan 

# Object Detection Metrics 

- Intersection over Union. 𝑎𝑟𝑒𝑎(𝑏∩𝑏𝑔) 

- 𝐼𝑜𝑈= 𝑎𝑟𝑒𝑎(𝑏∪𝑏𝑔) 

𝑏 is predicted bounding box, 𝑏𝑔 is ground truth bounding box. 

- Confidence 

   - Predicted by detector. 

   - Probability of valid object in bounding box. 

- TP/FP/FN Detection Conditions 

   - 𝐶𝑜𝑛𝑓> 𝑇1 

   - 𝐼𝑜𝑈> 𝑇2 

   - Thresholds Affects Precision/Recall 

- (a) Object Detection: Calculating Bounding Box IoU 

- (b) Example of True Positive, False Positive and False Negative Detections, (original image<sup>(vii)</sup> )) 

(vii) ”Sheep in a thistle field” Author: Forester2009, License: **CC BY-SA 3.0** 

©Dr. Tony Scanlan 

# Precision Recall Curve 

- No True Negatives used evaluating in Object detection => use Precision and Recall. 

   - Precision: Proportion of Positive Predictions actually correct. 

𝑇𝑃 𝑃𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛= 𝑇𝑃+ 𝐹𝑃 

- Recall: Proportion of Positives correctly identified. 

𝑇𝑃 𝑅𝑒𝑐𝑎𝑙𝑙= 𝑇𝑃+ 𝐹𝑁 

- Plot Precision vs Recall for various <u>Confidence Threshold</u> 𝑇1 values (fixed 𝑇2) . 

   - High 𝑇1 : High Precision Low Recall 

   - Low 𝑇1 : Low Precision High Recall 

©Dr. Tony Scanlan 

# Interpolating PR Curve 

- Real Precision Recall Curves often Zig-Zag [1]. 

   - Recall monotonically increases as Precision goes up and Down. 

- Ideal Detector: Want Precision to remain high as Recall increases 

   - Measure area under curve as performance metric 

   - Need to interpolate curve to estimate area 

   - Interpolated Precision 𝑝𝑖𝑛𝑡𝑒𝑟𝑝 at recall 𝑟 (where 𝑝 𝑟 precision at recall 𝑟 ) 𝑝𝑖𝑛𝑡𝑒𝑟𝑝 𝑟= max𝑟<sup>′</sup> >𝑟<sup>𝑝</sup> 𝑟<sup>′</sup> 

### (a) Precision Recall Curve for Binary Object Detector<sup>(viii)</sup> 

   - (b) Interpolated Precision Recall Curve for Binary Object Detector<sup>(viii)</sup> 

- [2] Padilla, R., Netto, S.L. and da Silva, E.A., 2020, July. A survey on performance metrics for object-detection algorithms. In _2020 International Conference on Systems, Signals and Image Processing (IWSSIP)_ (pp. 237-242). IEEE. 

(vii) An introduction to Evaluation Metrics for Object Detection 

©Dr. Tony Scanlan 

# Average Precision & Mean Average Precision 

• Average Precision Metric (area under curve) 𝑛−1 𝑟𝑖+1 −𝑟𝑖 𝑝𝑖𝑛𝑡𝑒𝑟𝑝 𝑟𝑖+1 𝐴𝑃𝐼𝑜𝑈>𝑇2 = ෍ 𝑖=1 

- Typically will have multiple object categories 𝐶 . 

• => Mean average Precision 𝐶 σ𝑖=1 𝐴𝑃𝑖 𝑚𝐴𝑃𝐼𝑜𝑈>𝑇2 = 𝐶 

- Note AP, mAP affected by IoU Threshold 𝑇2 

- (a) Interpolated Precision Recall Curve for Binary Object Detector<sup>(viii)</sup> 

(b) Example of AP data for individual Classes on the PASCAL VOC2012 dataset. Data from [2] 

[3] Redmon, J. and Farhadi, A., 2017. YOLO9000: better, faster, stronger. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 7263-7271). 

©Dr. Tony Scanlan 

# Object Detection Errors 

- Root cause Errors impacting AP [4] 

- False Positives 

   - Localisation Error 

   - Similarity Error 

   - Confusion with Background 

- False Negatives 

   - Object Size (small objects) 

   - Occlusion/Truncation 

   - (a) Examples of False Positive Object Detection Errors<sup>[4]</sup> 

- [4] Hoiem, D., Chodpathumwan, Y. and Dai, Q., 2012, October. Diagnosing error in object detectors. In _European conference on computer vision_ (pp. 340-353). Springer, Berlin, Heidelberg. 

©Dr. Tony Scanlan 

# Evaluating Errors with AP 

- COCO API analysis function for failure diagnosis<sup>(ix)</sup> 

   - Series of Precision Recall Curves with gradually more relaxed conditions. 

   - Allows identification of sources of failure in object detector when evaluated on COCO test data. 

      - Test Conditions: C75: IoU = 0.75 C50: IoU = 0.5 

Loc: localisation errors removed 

Sim: FP Errors between similar classes (supercategory) removed. 

Oth: Similar to Sim, all misclassification removed. 

BG: All FP due to background removed FN: All false negatives removed (AP=1) 

(a) Precision Recall curves produced by COCO Analysis<sup>(ix)</sup> 

(ix) COCO Common Objects in Context, Detection Evaluation 

©Dr. Tony Scanlan 

# Data Augmentation for Object Detection 

- Augmentation Operations[10] 

   - Photometric Operations 

      - Equalise, Contrast, Brightness 

   - Geometric Operations 

      - Rotate, Shear, Translate, Flip 

   - Bounding Box Information needs to be modified also 

   - Operations Can be applied to entire image or just with bounding box 

   - Can apply multiple operations to each image. (Optimal Policy) 

- Object Detection specific Augmentation not available in Tensorflow 

   - Many papers use simpler policies (e.g. image scaling, translation & random crops) 

   - (a) Examples of Data Augmentation for Object detection [10] 

- [5] Zoph, B., Cubuk, E.D., Ghiasi, G., Lin, T.Y., Shlens, J. and Le, Q.V., 2020, August. Learning data augmentation strategies <u>for object detection. In</u> _European Conference on Computer Vision_ (pp. 566-583). Springer, Cham.
