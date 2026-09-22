# lesson_3_3_obj_det_partA_w_reg_Prop

Deep Learning for Computer vision 

Object Detection (Part A): Two Stage Object Detection with Region Proposals Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Object Detection with Region Proposals 

- Discuss RCNN & Variants 

- RCNN Concepts/Problems/Solutions. 

   - C1: Use of Convolutional Features 

   - P2: Region Proposals Different Sizes 

      - C2: Spatial Pyramidal Pooling/Region of Interest Pooling – Fast RCNN 

   - P3: Image Classifier Operates on overlapping regions several times. 

      - C3: ROI Projection – Fast RCNN 

   - P4: Region Proposal Trained Independently 

      - C4: Region Proposal Network – Faster RCNN 

- (a) Basic Deep Learning Object Detection pipeline with Region Proposals and pre-trained Image Classifier. 

©Dr. Tony Scanlan 

# C1: Use of Convolutional Features 

- CNN Initially adopted as Feature Extractor in Classifier portion object detection system. 

   - Learned Features Produced by CNN seen as superior to Engineered Features. 

- SVM used for Classifier Portion. 

- RCNN Problems/Solutions. • Classifier Trained Independently from CNN. Not End-to-End Learning. 

©Dr. Tony Scanlan 

# RCNN (Region CNN). 

- Multi-stage pipeline • Selective search region proposal 

   - Proposals warped for input to CNN 

   - • Alexnet CNN (feature generator) 

   - Finetune training (Transfer learning) 

   - • Class specific SVM (note training examples). • Bound box regressor (uses cnn features). 

- Each pipeline stage must be individually trained 

- • Computationally slower than image classification • CNN features extracted per object proposal • Features stored for bounding box regressor and SVM classifier. (100’s GB) 

[2] Girshick R., Donahue J., Darrell T., Malik J. (2016) Region-based convolutional networks for <u>accurate object detection and segmentation. IEEE TPAMI 38(1):142–158</u> 

©Dr. Tony Scanlan 

## Towards End to End Learning for Object Detectors 

- RCNN + variants are conceptually 2 stage Detectors • 1<sup>st</sup> stage is Region Proposal & convolution features 

   - 2<sup>nd</sup> Stage operates on regions & predicts bounding box & Class 

- Original RCNN uses convolutional features & SVM/Regression stage to predict Classes/bounding boxes. • Warping Region Proposals to get fixed size vector. 

- Improve RCNN: 

   - Directly Predict Bounding boxes & Class in region proposals with Fully Connected Neural Network 

©Dr. Tony Scanlan 

# C2: Spatial Pyramid Pooling 

- Convolution stages can take any size (spatial dimensions only) input. 

   - Output size (spatial dimensions) will vary with input size. 

- Want to connect to FC Stages to Predict Class and Bounding Box 

   - FC stages require fixed input size to operate. 

   - => Warp Input 

- Spatial Pyramid Pooling [3] 

   - Outputs fixed sized vector (independent of input size) 

   - Allows removal of fixed region proposal size constraint (and need to warp). 

[3] He, K., Zhang, X., Ren, S. and Sun, J., 2015. Spatial pyramid pooling in deep convolutional networks <u>for visual recognition.</u> _IEEE transactions on pattern analysis and machine intelligence_ , _37_ (9), pp.19041916. 

(c) Spatial Pyramid Pooling Layer. 

©Dr. Tony Scanlan Fast RCNN[5] 

- Compared to RCNN: 

   - Improving training time, reduced computation and increased speed. 

- Use of Neural Network for directly prediction of Class probabilities & bounding box values of regions. 

   - Enables End to End Training 

- Region of Interest Projection 

   - Reduced Computation (Image Processed once) 

- Use of RoI Pooling (Special Case of SPP) 

   - RoI pooling output is always 7 x 7 

      - Equivalent to one level of SPP. 

   - Multi Task Loss 

- (a) Fast RCNN Object Detection Pipeline 

[5] Girshick, R., 2015. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_ (pp. 1440-1448). 

©Dr. Tony Scanlan 

# Class Activation Maps 

   - Results [6,7] show Image Classification CNNs already have built in indication of object location based on activation (feature) maps. 

   - Class Activation Map: 

      - Visualise spatial activation for input image. 

      - Uses Global Average Pooling for last pooling layer. 

      - Sum of activation maps based on class weights. 

      - Unsupervised object detection while training for image classification. 

         - Can perform region of interest projection Fast RCNN. 

         - Enables Region Proposal Network Faster RCNN 

- [6] Oquab, M., Bottou, L., Laptev, I. and Sivic, J., 2015. Is object localization for free?-weakly-supervised learning <u>with convolutional neural networks. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 685-694). 

- [7] Zhou, B., Khosla, A., Lapedriza, A., Oliva, A. and Torralba, A., 2016. Learning deep features for discriminative <u>localization. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 2921-2929). 

- (a) Class Activation Mapping [7]. 

- (b) Class Activation Maps for 4 classes [7]. 

©Dr. Tony Scanlan 

# C3: Region of Interest Projection 

- Region proposals projected over feature maps after convolution 

   - Selective Search generates region proposals 

      - Projected to output feature maps 

   - Convolution performed just once per image as opposed to per region proposal in RCNN. 

   - Greatly speeds up inference 

- (a) Projection of Region Proposals 

- (b) Fast RCNN Object Detection Pipeline 

[5] Girshick, R., 2015. Fast r-cnn. In _Proceedings of the IEEE international conference on computer vision_ (pp. 1440-1448). 

©Dr. Tony Scanlan 

# C4: Region Proposal Network 

- Faster RCNN: Eliminating external Region Proposals [8] 

   - Object location Information already in CNN no need to externally supply from selective search. 

   - Add Region Proposal Network. 

      - Conv based, can be trained end to end. 

   - Ultimately Shares convolution layers with feature generating CNN 

   - Similar to Single Stage Detector 

- Alternating training scheme 

   - Train RPN (Base network with ImageNet weights) 

   - Train Fast RCNN with RPN proposals 

   - Join Networks 

   - Fix Common conv layers from RCNN and fine tune RPN unique layers. 

   - Fix common conv layers and fine tune Fast RCNN part. 

[8] Ren, S., He, K., Girshick, R. and Sun, J., 2015. Faster r- <u>cnn: Towards real-time object detection with region proposal networks. In</u> _Advances in neural information processing systems_ (pp. 91-99). 

- (a) Faster RCNN Object Detection Pipeline 

©Dr. Tony Scanlan 

# Comparison 2 Stage and Single Stage 

- Region Proposal Network is equivalent to Single Stage Detector. • RPN finds bounding boxes & confidence 

- • Predictor head directly predicts bounding boxes, class & confidence 

- Additional Stage provides more accurate bounding box estimation and classification of objects. 

- • Slower Frame rate. 

©Dr. Tony Scanlan 

# Summary 

- Object Detection with Deep CNN has rapidly advanced in parallel with image classification. 

- Focused on RCNN family. 

   - Use Region Proposals for high accuracy. 

   - Can be considered an Object Detection framework. 

      - There are many published variations with different enhancements and backbone networks. [1] 

   - Key Advances in RCNN family 

      - RCNN 

         - Use of CNN features for object detection. 

      - Fast RCNN 

         - Use of Region of Interest Pooling, CNN classifier/regression for end-toend training and reduction in CNN computation. 

      - Faster RCNN 

         - Region Proposal Network to remove external region proposal generation. 

- Accuracy, Computation, Applications.
