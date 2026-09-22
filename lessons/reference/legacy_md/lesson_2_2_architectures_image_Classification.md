# lesson_2_2_architectures_image_Classification

Deep Learning for Computer vision 

Architectures for Image Classification: Alexnet, VGG & Inception Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

## AlexNet 

- Structure 

   - Originally Split over 2 GPUs. • Recombine paths with Pooling. 

   - 5 Convolution layers 

   - 3 Fully Connected Layers. 

   - Relu activation 

   - Pooling Layers indicated but feature maps not shown in diagram. 

- Overfitting 

   - Dropout Regularisation. 

   - Data Augmentation. 

|**layer​**|**Type​**|**Channels**|**Size​**|**KernelSize**|**Stride​**|**Activation​**|
|---|---|---|---|---|---|---|
|Input​|Image​|3​|224 x 224​|-|-|-|
|1​|Conv​ 1|96|55 x 55​|11x 11​|4​|ReLU|
|2​|Max Pool​|96​|27x27|3 x 3​|2​|-|
|3|Conv​ 2|256|27 x 27​|5 x 5​|1​|ReLU|
|4​|Max Pool​|256|13 x 13|3 x 3|2​|-|
|5|Conv 3|384|13 x 13|3 x 3|1|ReLU|
|6|Conv 4|384|13 x 13|3 x 3|1|ReLU|
|7|Conv 5|256|13 x 13|3 x 3|1|ReLU|
|8|Max Pool|256|6 x 6|3 x 3|2|-|
|9|FC​|-|4096|-|-|ReLU|
|10​|FC​|-|4096|-|-|ReLU|
|11|FC​|-|1000|-|-|Softmax|

(a) (Single GPU) Structure of AlexNet [2] 

[1] Krizhevsky, A., Sutskever, I. and Hinton, G.E., 2012. Imagenet <u>classification with deep convolutional neural networks. In</u> _Advances in neural information processing systems_ (pp. 1097-1105). [Cited by 35554] 

©Dr. Tony Scanlan 

## AlexNet Computation and Memory Requirements 

- FC Layers Dominate Memory. 

- Computation rate limited by GPU memory bandwidth. 

- • Convolution Layers Dominate Computation. 

<!-- Start of picture text -->
Computational Requirement FLOPS<br>105M 112M 149M 112M 74M 38M 17M 2M<br>Conv FLOPS 554M<br>FC FLOPS 59M<br>Conv Mem 2.3M<br>FC Mem 57M<br>35k 307k 884k 663k 442k 38M 17M 4M Memory Requirement (No Parameters)<br>Working Memory Requirement (No. Output<br>290k 187k 64k 65k 43k 4k 4k 1k<br>Map Values)<br>FC Layer 1 FC Layer 2 FC Layer 3<br>Conv Layer 1 Conv Layer 2 Conv Layer 3 Conv Layer 4 Conv Layer 5<br><!-- End of picture text -->

- Computation rate not limited by memory bandwidth. 

- (a) Computational Requirements of AlexNet<sup>(i)</sup> 

(i) A Closer Look at AlexNet – Tutorial 6 

©Dr. Tony Scanlan 

# VGG Networks 

- Structure 

   - Only 3 x 3 Convolutions 

      - Receptive field of back-back conv. equivalent to larger conv. 

      - Computationally efficient. 

   - Deeper (more conv layers). 

      - VGG-16 / VGG-19 

      - Far more model parameters than AlexNet. (Still has Fully Connected Layers) 

      - Increased Computation. 

      - Overfitting 

   - (a) VGG-16 Model Architecture [2] 

- Increased Accuracy. 

[2] Simonyan, K. and Zisserman, A., 2014. Very deep convolutional networks for large-scale image recognition. _arXiv preprint arXiv:1409.1556_ . 

©Dr. Tony Scanlan 

# Model Zoo 

- Computation vs Accuracy tradeoff [3]. 

- AlexNet/VGG Large Number of Parameters (in FC layers) 

   - Too Large for Embedded Devices 

   - Slow Training 

- Newer Deeper models eliminate Fully Connected Layers to reduce Parameter numbers. 

   - Inception Architecture (GoogLeNet) 

   - Residual Structures (ResNets) 

- (a) Accuracy (%) vs Operations (G-FLOPs) for various Networks [3] 

[3] Bianco, S., Cadene, R., Celona, L. and Napoletano, P., 2018. <u>Benchmark Analysis of Representative Deep Neural Network Architectures.</u> _IEEE Access_ , _6_ , pp.64270-64277. 

©Dr. Tony Scanlan 

# Network in Network 

- Small FC Network operates in convolution manner 

- Convolution is a Linear Operation 

   - Abstraction Low 

- More abstraction with MLP (Fully Connected Layers) 

   - Large FC Layers not required to discriminate features 

- Effectively Equivalent to 1 x 1 convolution 

- (a) Convolution Layer [4] 

- (b) Convolution Layer with FC Layers [4] 

[4] Lin, M., Chen, Q. and Yan, S., 2013. Network in network. _arXiv preprint arXiv:1312.4400_ . 

©Dr. Tony Scanlan 

# Global Average Pooling 

- Alternative to FC layer. 

   - Reduces parameter count & overfitting 

   - Conv layers must extract more abstract features or be deeper to compensate 

- Automatically pools each individual feature map down to size 1. 

      - (a) Global Average Pooling<sup>(ii)</sup> 

   - Left with 1d array (same as FC output) 

- (ii) Global Average Pooling Layers for Object Localization 

©Dr. Tony Scanlan 

# Inception Principles 

- Probability Distribution of Data represented by some Sparse Network [5] 

   - Modern GPU not suitable for sparse computation. 

- Motivation to find Dense equivalent of Sparse Model: 

   - Use convolution Layers 

   - “Layer-by layer construction in which one should analyze the correlation statistics of the last layer and cluster them into groups of units with high correlation. These clusters form the units of the next layer and are connected to the units in the previous layer” 

   - 1 x 1 convolution finds correlation across feature maps 

   - 3 x 3 and 5 x 5 can be used over wider regions 

- (a) Dense and Sparse Networks 

- (b) Finding Correlated Features 

[5] Arora, S., Bhaskara, A., Ge, R. and Ma, T., 2014, January. Provable bounds for learning some deep representations. In _International Conference on Machine Learning_ (pp. 584-592). 

©Dr. Tony Scanlan 

# Inception Module 

- Inception Module[5] (Repeating Unit) 

- • Based “Network in Network” [6] 

   - Increased Representative power with embedded network. 

- Different size convolutions to find & cluster correlations. (fig. (a)) • Computationally demanding 

- • Additional 1x1 “bottleneck” convolutions perform dimension reduction. (fig. (b)) 

[6] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V. and Rabinovich, A., 2015. Going deeper with <u>convolutions. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 1-9). 

©Dr. Tony Scanlan 

# Auxillary Classifiers 

- GoogleLeNet: Large 22 (conv.) layer Deep network. 

- 1 FC Layer 

   - Use Average Pooling to Reduce dimensions. 

- Deep and wide for given computational cost. 

- Auxiliary classifiers 

   - Avoid vanishing gradient during training. 

   - Effectively have superposition of networks. 

   - Joint loss during training. 

- Removed for inference. 

(a) GoogLeNet (Last Layers) Inception Architecture [5] (see Fig 3 in [5] for full Network & table) 

©Dr. Tony Scanlan 

# Inception Performance 

- 12x Less Parameters than AlexNet 

   - Increased Performance 

- Enhancements v2,v3v4 [7,8] 

   - Batch Normalisation 

   - 5 x 5 conv replaced by stacked 3 x3 

   - Stride 2 convolution to reduce dimensions 

[7] Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J. and Wojna, Z., 2016. <u>Rethinking the inception architecture for computer vision. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 28182826). 

[8] Szegedy, C., Ioffe, S., Vanhoucke, V. and Alemi, A., 2017, February. <u>Inception-v4, inception-resnet and the impact of residual connections on learning. In</u> _Proceedings of the AAAI conference on artificial intelligence_ (Vol. 31, No. 1). 

- (a) Accuracy (%) vs Operations (G-FLOPs) for various Networks
