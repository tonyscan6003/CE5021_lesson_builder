# lesson_3_5_sem_seg_partA

Deep Learning for Computer vision 

Semantic Segmentation (Part A) Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Segmentation 

- Basic Segmentation is dividing image into regions of similar pixels. 

- Semantic segmentation: 

   - Each pixel is assigned a class. 

- Instance segmentation uses information from object detection. 

   - Multiple instances labelled. 

   - Key task in Challenges such as MS COCO. 

- (a) Sematic Segmentation with Deep Lab 

(b) Instance Segmentation with Mask RCNN 

©Dr. Tony Scanlan 

# Datasets 

   - Pascale VOC<sup>(i)</sup> 

   - Microsoft COCO<sup>(ii)</sup> 

   - ADE20K<sup>(iii)</sup> 

      - Includes Object Parts. 

   - Autonomous driving space: 

      - Cityscapes<sup>(iv)</sup> 

      - Kitti<sup>(v)</sup> 

   - Lots of Datasets! 

      - Descriptions [1] 

      - Comprehensive List<sup>(vi)</sup> 

- [1] Garcia-Garcia, A., Orts-Escolano, S., Oprea, S., Villena-Martinez, V. and Garcia-Rodriguez, J., 2017. A review on deep learning techniques applied to <u>semantic segmentation.</u> 

- (i) <u>PASCAL VOC 2011 Example Segmentations</u> 

- (ii) <u>COCO Dataset</u> 

- (iii) <u>ADE 20K</u> 

- (iv) <u>The Cityscsapes Dataset</u> 

- (v) <u>KITTI Vision Benchmark Suite</u> 

- (a) Instance Segmentation Microsoft COCO<sup>(ii)</sup> 

(b) Segmentation ADE20K, object part Segmentation<sup>(ii)</sup> 

(c) Driving Scene Segmentation Kitti Dataset<sup>(V)</sup> 

(vi) <u>Scenes or Places, Scene Segmentation of Classification</u> 

©Dr. Tony Scanlan 

## Segmentation Network Output 

- Image Classification 

   - Output vector  1 x NClasses 

- Semantic segmentation 

   - Predict at every pixel. 

   - Output tensor H x W x Nclasses 

   - Combine class prediction into false colour map. 

(a) 1 x Nclasses Output Vector from Image Classification, (b) Output tensor H x W x Nclasses Produced by Semantic Segmentation Network 

©Dr. Tony Scanlan 

# Evaluating Semantic Segmentation 

- Intersection over union. 

   - Calculated for each class at a pixel level 𝑇𝑃 

   - • 𝐼𝑜𝑈= 

𝑇𝑃+𝐹𝑃+𝐹𝑁 

- Can obtain a mean value across all classes. 

- Same as metric used for bounding boxes in object detection. 

(a) Ground Truth (b) Predicted Segmentation. (c) True positive, False positive and False Negative predictions for calculation of IoU. 

### ©Dr. Tony Scanlan 

# Fully Convolutional Networks 

- FCN [2] 

   - First end-to-end trained network for Sem. Seg. 

      - Uses pre-trained Alexnet. 

   - Fully convolutional allows arbitrary sized inputs. 

      - Similar to Overfeat/SSD with FC used as convolution. 

   - Output feature map is interpolated to give fine scale output. 

      - Use of transpose convolutions for up-sampling. 

      - Use of feedforward skip connections to improve output quality. (use of early stage fine information) 

[2] Long, J., Shelhamer, E. and Darrell, T., 2015. <u>Fully convolutional networks for semantic segmentation. In</u> _Proceedings of the IEEE_ 

_conference on computer vision and pattern recognition_ (pp. 3431-3440). 

(c) 

(a) Network for Image Classification (b) Fully convolutional Network with up-sampling paths for segmented output. (c) Output of FCN for different paths [2] 

©Dr. Tony Scanlan 

# Transpose convolution I 

- Convolution as matrix multiplication. 

- Spatial Dimension reduced by Convolution. 

- Spatial Dimension can be increased by Transpose convolution. 

(a) 3 x 3 Input matrix, (b) 2 x 2 kernel, (c) Equivalent Multiplication (d) Equivalent Transpose Convolution 

©Dr. Tony Scanlan 

# Transpose Convolution II. 

- Transpose convolution emulated by direct convolution in practice [3]. 

   - Increase output feature map size. 

- Learned weights to perform upsampling. 

- Also known as Fractionally strided convolution. 

   - Note insertion of zeros between samples in case where have equivalent Transpose convolution. 

[3] Dumoulin, V. and Visin, F., 2016. A guide to convolution arithmetic <u>for deep learning.</u> _arXiv preprint arXiv:1603.07285_ . 

(a) Convolution of 4 x 4 image with 3 x 3 kernel, (b) Equivalent Transpose Convolution of (a) requires zero padding x2, (c) Convolution of 5 x 5 image with 3 x 3 kernel stride 2, (d) Equivalent Transpose Convolution with insertion of zeros and convolution of 3 x 3 kernel stride 1 (Note: 1-dimensional equivalent shown) 

©Dr. Tony Scanlan 

## Encoder-Decoder Network 

- Uses a dedicated Decoder Network[4,5] as opposed to minimal decoder used in FCN. 

   - Based on VGG-16 

   - Better at predicting fine detail. 

- Un-pooling Layers used to recover spatial information. 

   - Non-standard layer. 

<!-- Start of picture text -->
Transpose<br>Conv<br><!-- End of picture text -->

[4] Noh, H., Hong, S. and Han, B., 2015. Learning deconvolution network for <u>semantic segmentation. In</u> _Proceedings of the IEEE international conference on computer vision_ (pp. 1520-1528). 

#### (c) 

[5] Badrinarayanan, V., Kendall, A. and Cipolla, R., 2017. Segnet: A deep <u>convolutional encoder-decoder architecture for image segmentation.</u> _IEEE transactions on pattern analysis and machine intelligence_ , _39_ (12), pp.24812495. 

- (a) VGG-16 based Encoder Decoder Network, (b) Max Pooling and Un-pooling operations (c) Sparse output of Unpooling and application of Transpose Convolution for interpolation. 

©Dr. Tony Scanlan 

# U-net 

- Doesn’t use unpooling 

- Extensive use of skip connections 

   - Makes use of spatial information in encoding path. 

   - Concatenation of feature maps. 

- Can be used as a framework with any DNN architecture. 

[6] Ronneberger, O., Fischer, P. and Brox, T., 2015, October. <u>U-net: Convolutional networks for biomedical image segmentation. In</u> _International Conference on Medical image computing and computer-assisted intervention_ (pp. 234-241). Springer, Cham. 

- (a) U-Net Architecture from [6]
