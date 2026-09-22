# lesson_3_4_obj_det_partB_single_stage

Deep Learning for Computer vision 

Object Detection (Part B): Single Stage Object Detection Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Principles of Single Stage Detectors 

   - Naïve Approach to use DCNN as sliding window in Image. 

      - Fixed Input window => Image Pyramid 

      - Computationally Inefficient 

   - Feature maps contain spatial & Sematic information [1]. 

      - Coarse Information about object location from feature map 

      - Fine Information about spatial extent encoded in channel information (Feature map depth) 

- [1] ICCV 2015 Tutorial: <u>Convolutional Feature Maps Elements of efficient (and accurate) CNNbased object Detection, Kaiming He, International Conference Computer Vision 2015.</u> 

- (i) “A Street Scene in Trinidad Cuba” Author: Jose Porras Graham Crumb, License: **CC BY-SA 3.0** 

©Dr. Tony Scanlan 

# Object Detection System 

- Prediction Head: CNN to predict Confidence, Bounding box information & Class. 

   - Applied Convolutionally 

   - Usually 3 x 3 Kernel 

      - Receptive field means object extent is encoded (depthwise) at feature map pixel. 

      - Don’t need Larger kernel! 

   - Predicted Bounding box relative to coarse position. 

      - Correct with coarse coordinates to get position in final image. 

### ©Dr. Tony Scanlan 

# Object Detection Output/Training 

- **Output Vector:** 

- Bounding Boxes 

   - Multiple Boxes with Anchors 

- Confidence not always predicted 

   - Use a background class in classifier. (Imbalance) 

- Class 

   - Usually one Class prediction for cell not per bounding box. 

   - => Can get wrong class 

- **Ground Truth Assignment:** 

   - Anchor Based 

      - Additional computation for IoU 

   - Imbalance Problem: Too many Negative Examples (Confidence/class) 

(ii) “Bluetabbypoint <u>and white mitted Ragdoll male” Author: WeeJeeVee, License:</u> **CC BY-SA 4.0** 

©Dr. Tony Scanlan 

# Anchor Boxes 

- Anchor boxes 

   - Used as a Prior 

   - Predict Offset and scale of anchor from Ground truth. 

   - Faster Training Convergence. 

   - Can accommodate a good degree of shape/scale variation. 

      - Anchor boxes must be set A Priori (Bias). 

(iii) 

      - Work well for large objects with wide aspect ratios. 

      - Problem learning small objects 

   - May Predict several anchor boxes at each cell (overlapping objects). 

- All anchor boxes from any cell with IoU > _threshold_ are selected as positive predictions. 

   - Others are negative prediction for Confidence => Imbalance 

(iii) “Lizzie and Murray-Two dogs in a fields” Author: Panyd, License: **CC BY-SA 3.0** 

©Dr. Tony Scanlan 

# Bounding Box Prediction 

- Prediction Head regresses 𝑛 Anchor Boxes for each cell. 

   - Predicts **displacement** of bounding box centre coordinates from anchor box centre point. 

   - Predicts **scaling** of bounding box height and width from anchor box height and width. 

- Predict displacement 𝑡𝑥, 𝑡𝑦 & scaling 𝑡ℎ, 𝑡𝑤 from anchor box [2]. 

𝑡𝑥 = 𝑥 − 𝑥𝑎 Τ𝑤𝑎 (1) 𝑡𝑦 = 𝑦 − 𝑦𝑎 Τℎ𝑎 (2) 𝑡𝑤 = 𝑙𝑜𝑔 𝑤Τ𝑤𝑎 (3) 𝑡ℎ = 𝑙𝑜𝑔 ℎΤℎ𝑎 (4) 

- Train to match displacement of ground truth from anchor box 

[2] Ren, S., He, K., Girshick, R. and Sun, J., 2015. Faster r-cnn: Towards real-time object detection with <u>region proposal networks. In</u> _Advances in neural information processing systems_ (pp. 91-99). 

©Dr. Tony Scanlan 

# Loss Function for Object Detection 

(iii) Object Detection for Dummies Part 3: R-CNN Family Author: Lilian Weng, Dec 2017 

©Dr. Tony Scanlan 

# Focal loss for Training Imbalance 

• Cross Entropy 𝑔𝑡 = 1 𝑝𝑖 𝑖𝑓𝑝𝑖 **(1)** 𝐶𝐸(𝑝𝑖, 𝑝𝑖 −log 1 −𝑝𝑖 , 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒 𝑔𝑡) = ൝−log • Define 𝑝𝑡𝑖 𝑔𝑡 = 1 𝑖𝑓𝑝𝑖 **(2)** 1 −𝑝𝑖 𝑜𝑡ℎ𝑒𝑟𝑤𝑖𝑠𝑒 𝑝𝑡𝑖 = ൝<sup>𝑝𝑖</sup> • Focal Loss 𝐹𝐿 **(3)** 𝑝𝑡𝑖 = − 1 −𝑝𝑡𝑖<sup>𝛾</sup> 𝑙𝑜𝑔 𝑝𝑡𝑖 𝐹𝐿 𝑝𝑖 𝑔𝑡 = − **(4)** 1 − <u>𝑝𝑡𝑖</u><sup>𝛾</sup> <u>𝑝𝑖𝑔𝑡</u> log <u>𝑝𝑖</u> − 1 − <u>𝑝𝑖 𝑝𝑡𝑖</u><sup>𝛾</sup> 𝑙𝑜𝑔 1 − <u>𝑝𝑡𝑖</u> 𝑝𝑖 →1 𝑝𝑖 →0 𝑔𝑡 **= 1** 𝑔𝑡 **= 0** 𝑝𝑖 𝑝𝑖 log 𝑝𝑖 →0 log 1 −𝑝𝑖 →0 1 −𝑝𝑖<sup>𝛾</sup> →0 𝑝𝑖<sup>𝛾</sup> →0 

- (a) Focal Loss [3]: Adds a scale factor 1 −𝑝𝑡𝑖<sup>𝛾</sup> to standard cross entropy. Setting 

- 𝛾> 0 reduces loss for well classified examples, putting focus on hard misclassified examples. 

[3] Lin, T.Y., Goyal, P., Girshick, R., He, K. and Dollár, P., 2017. <u>Focal loss for dense object detection. In</u> _Proceedings of the IEEE international conference on computer vision_ (pp. 2980-2988). 

©Dr. Tony Scanlan 

## Scale Problem in Single Stage Object Detection 

- In basic System Prediction from only one feature map. 

   - Used Coarse feature map with wide receptive field. 

      - Only Larger objects in Image Detected. 

      - Difficulty with smaller objects. 

(iv) 

   - Mitigate with increased Image Size (or pyramid) • Increased Computation 

- Efficient to use feature maps at different scales. • Small objects can be localised in fine feature maps • Different scale anchor boxes for each Feature Map. • Earlier feature maps may lack semantic information. • SSD / Feature Pyramid Network 

(iv) “Dog Playing Ball” Author: pdpics, License: **CC BY-SA 3.0** 

©Dr. Tony Scanlan 

# Multi-Scale Feature maps 

- SSD (Single Stage Object Detector [4]) 

   - Multiscale feature maps from which prediction can be generated. 

      - Poor semantic information in VGG16 fine feature maps 

      - Further Downsamples Conv_5_3 layer 

      - Retains semantic information 

      - Increased image size 

   - Prediction generated by convolution at each feature map location. 

[4] Liu, W., Anguelov, D., Erhan, D., Szegedy, C., Reed, S., Fu, C.Y. and Berg, A.C., 2016, October. <u>Ssd: Single shot multibox detector. In</u> _European conference on computer vision_ (pp. 21-37). Springer, Cham. 

(a) Single Shot (multibox) Detector: Uses VGG-16 as base network, additional convolution layers added to produce multiscale outputs. Bounding boxes and classes predicted from feature map using small 3 x 3 convolutions. 

©Dr. Tony Scanlan 

# Feature Pyramid Networks 

- Use Feature maps at different scales. 

   - (a) Feature Pyramid Network [5] 

- However semantic information may be poor in earlier feature maps. 

- FPN proposes creating a parallel pyramid with increased semantic information. 

   - Yields greater prediction accuracy for smaller objects. 

   - Widely Adopted Approach [5,6] 

      - (b) Example Detector using FPN with ResNet 50 Backbone [6] 

   - Shared Prediction Head. 

- [5] Lin, T.Y., Dollár, P., Girshick, R., He, K., Hariharan, B. and Belongie, S., 2017. <u>Feature pyramid networks for object detection. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 2117-2125). 

- [6] Li, Y. and Ren, F., 2019. Light-weight retinanet for object detection. _arXiv preprint arXiv:1905.10011_ .
