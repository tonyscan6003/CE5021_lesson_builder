# lesson_3_6_sem_seg_partB

Deep Learning for Computer vision 

Semantic Segmentation (Part B): Context and Instance Segmentation Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Context Knowledge 

- CNNs used for pixel level prediction. 

   - Context Information about a scene is reduced by CNN pooling and downsampling. 

   - But Semantic Information Increases. 

- Spatial information (e.g. skip connections, unpooling) improves pixel level accuracy. 

- Context information is generally increased by increasing the receptive field. 

- (a) Input Image 

      - (b) Ground Truth (c) FCN Output 

- In this lecture: 

   - Dilated Convolution 

   - Pyramid Scene pooling 

   - CRF (Conditional Random Fields) 

Figure: Examples of Semantic Segmentation mislabelling due to lack of Context. Pixels in Windowpane are classified as a door [1]. Boat misclassified as a car [2] 

- [1] Zhang, H., Dana, K., Shi, J., Zhang, Z., Wang, X., Tyagi, A. and Agrawal, A., 2018. Context encoding for semantic segmentation. In _Proceedings of the IEEE conference on Computer Vision and Pattern Recognition_ (pp. 7151-7160). 

[2] Zhao, H., Shi, J., Qi, X., Wang, X. and Jia, J., 2017. Pyramid scene parsing network. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 2881-2890). 

©Dr. Tony Scanlan 

# Dilated Convolutions 

- Also known as “Atrous Convolution” 

- Dilated convolutions formed by spacing out inputs. 

   - Number of parameters stays the same. 

   - Cascading dilated convolutions produces exponential growth in receptive field for linear growth in parameters. 

- Suitable for use in semantic segmentation [3]. 

- (a) Receptive field from Dilated Convolution. 

[3] Yu, F. and Koltun, V., 2015. Multi-scale context aggregation by dilated convolutions. _arXiv preprint arXiv:1511.07122_ . 

©Dr. Tony Scanlan 

# Pyramid Scene Parsing [2] 

- Similar to Spatial Pyramid Pooling in object detection. • Retain information about features at different spatial positions. 

- Based on ResNet backbone. 

   - With dilated convolution. 

- A concatenated feature map with upsampled pooling layers & original features. 

   - (a) Pyramid Scene Parsing Network 

- Spatial Information at different scales (in output feature map) is aggregated. 

©Dr. Tony Scanlan 

# Conditional Random fields. 

- Explicitly model relationships between pixels. 

   - Pixel intensity, position predict label. 

   - Fully Connected CRF (graph). 

- Has been used in machine vision for segmentation prior to deep learning [4]. 

   - Is also discriminative model. 

   - Computationally intractable to learn model at pixel level. 

   - Efficient Inference Algorithm: Dense CRF [5] 

## (a) Fully Connected CRF 

- (b) Image, Segmentation with CRF [5], Ground Truth 

[4] Arnab, A., Zheng, S., Jayasumana, S., Romera-Paredes, B., Larsson, M., Kirillov, A., Savchynskyy, B., Rother, C., Kahl, F. and Torr, P.H., 2018. <u>Conditional random fields meet deep neural networks for semantic segmentation: Combining probabilistic graphical models with deep learning for structured prediction.</u> _IEEE Signal Processing Magazine_ , _35_ (1), pp.37-52. 

[5] P. Krahenb ¨ uhl and V. Koltun, “Efficient inference in fully connected crfs with gaussian edge potentials,” in NIPS, 2011 

©Dr. Tony Scanlan 

# DeepLab 

- A State of the art segmentation framework. 

   - Use of Dilated Convolutions in DCNN (VGG/ResNet) 

      - Image reduced x8 

   - Spatial pyramid pooling for scale. 

   - Bi-linear interpolation 

      - No deconvolution Network. 

      - No explicit skip connections. 

- CRF as a postprocessing method • Can be co-trained with DCNN or even cast as RCNN [4]. 

- (a) Deeplab System High level Overview [6] 

[6] Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K. and Yuille, A.L., 2017. Deeplab: Semantic image segmentation with deep <u>convolutional nets, atrous convolution, and fully connected crfs.</u> _IEEE transactions on pattern analysis and machine intelligence_ , _40_ (4), pp.834-848. 

©Dr. Tony Scanlan 

# Instance Segmentation 

- Combining object detection and semantic segmentation. 

   - Goal to identify different instances of same object in a scene. 

   - Segmentation before recognition. 

      - Segmentation (Object) Proposals. 

      - Cut instances from pixel level semantic segmentation [7]. 

   - Mask RCNN [8] 

      - Simultaneous prediction of class labels and binary object mask. 

- (a) Instance segmentation with proposals 

- (b) Instance segmentation with proposals [7] 

   - (c) Binary Mask generation with mask RCNN [8] 

- [7] Arnab, A. and Torr, P.H., 2017. <u>Pixelwise instance segmentation with a dynamically instantiated network. In</u> _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_ (pp. 441-450). 

- [8] He, K., Gkioxari, G., Dollár, P. and Girshick, R., 2017. <u>Mask r-cnn. In</u> _Proceedings of the IEEE international conference on computer vision_ (pp. 2961-2969). 

©Dr. Tony Scanlan 

# Mask RCNN 

- Predict m x m Mask with FCN 

   - Maintains spatial dimensions from Output feature maps. 

      - (a) Mask RCNN 

- Faster RCNN: RoI Pooling 

   - Region proposal quantized to output feature maps. 

   - Pooling of features within bins of RoI 

   - Gives fixed sized output (7 x 7 x C)  to Classifier/Bounding box prediction. 

- Mask RCNN: RoI Align 

   - No quantization (performance key). 

   - Features are interpolated to points in bins of RoI. 

   - Helps preserve spatial correspondence of output features with proposed RoI. 

©Dr. Tony Scanlan 

# Mask RCNN Output 

- Note that output masks are low resolution and projected onto original image. 

   - Limited Resolution due to output size of Mask path. 

   - Output predicts k masks one per class only use mask for predicted class (from class prediction branch). 

(a) Output of Mask RCNN [8]
