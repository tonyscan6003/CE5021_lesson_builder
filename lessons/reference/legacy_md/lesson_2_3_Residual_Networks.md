# lesson_2_3_Residual_Networks

Deep Learning for Computer vision 

Residual Structures Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Introduction 

- ResNets [1] 

   - Next most important breakthrough after AlexNet. 

   - Trial extension of VGG-19 with additional layers 

      - Higher training error than simpler Network. 

      - Exponentially low convergence rates. 

   - Use of Residual Network Structure to allow training of very Deep Networks. (up 1001 layers!) 

   - Less Parameters than VGG (with Bottleneck Blocks). 

[1] He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image <u>recognition. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 770-778). 

- (a) Comparison of VGG-19 Network & Residual Network[1] (see [1] fig (3)) 

©Dr. Tony Scanlan 

## Residual Learning & Residual Block 

- Desired mapping for stacked layer 𝐻 𝑋 

- • Allow stacked layers instead to fit mapping 𝐹 𝑋 , where 𝐹 𝑋= H 𝑋−𝑋 

𝐹 𝑋 

𝐹 𝑋+ X 

- Original mapping H 𝑋= F 𝑋+ 𝑋 

   - Realised with shortcut connections 

- Easier to train residual mapping than stacked layers. 

Fig (a): Residual Blocks output size (56,56): (a) 34-layer (b) Bottleneck Block  ResNet 50+ 

©Dr. Tony Scanlan 

# ResNet Connectivity 

- Unraveled View[6]: 

   - Network comprised paths of different lengths. 

- Ensemble behaviour 

   - Can remove Residual block & not affect 

   - behaviour 

- Short paths carry gradient (magnitude). 

- Enable deep networks by shortening effective paths. 

   - Avoids vanishing gradients. 

[6] Veit, A., Wilber, M.J. and Belongie, S., 2016. Residual networks behave <u>like ensembles of relatively shallow networks. In</u> _Advances in neural information processing systems_ (pp. 550-558). 

Fig (a): Simple Residual Network, (b) Unraveled view. 

©Dr. Tony Scanlan 

# ResNet Variations 

- Table of Variations 

• ResNet composed of regular repeating structures. 

Fig (a): Residual Blocks for output size (56,56): (a) ResNet 34-layer (b) Bottleneck Block  ResNet 50+ 

Table 1: Structure of Various ResNets 18-152 Layer [5]. 

©Dr. Tony Scanlan 

# Batch Normalisation I 

- Batch Normalisation[7] 

   - Very widely used Normalisation layer • Improves training time • Used in ResNet 

   - Original BN paper referrers to “Reducing the Internal Covariate Shift” 

      - Changes at input each layer due to updates during training. 

   - BN Transform 

      - Normalises activations (mean and variance.) 

      - Learns scale and shift parameter. 

**Batch Normalisation Algorithm** [7] Applied to activations 𝑥 over a mini-batch 𝐵= 𝑥1⋯𝑚 

Learn parameters 𝛾, 𝛽 

#### Mini Batch Mean: 

<!-- Start of picture text -->
𝑚<br>1<br>𝜇𝐵 ← 𝑥𝑖<br>𝑚 ෍<br>𝑖=1<br><!-- End of picture text -->

Mini Batch variance: 

<!-- Start of picture text -->
𝑚<br>1<br>𝜎𝐵2 ← 𝑥𝑖 −𝜇𝐵 2<br>𝑚 ෍<br>𝑖=1<br><!-- End of picture text -->

Normalise: 

- 𝑥𝑖 −𝜇𝐵 

- ො𝑥𝑖 ← 2 + 𝜖 

- 𝜎𝐵 

Scale and shift: 𝑦𝑖 ←𝛾ො𝑥𝑖 + β 

[7] Ioffe, S. and Szegedy, C., 2015. <u>Batch normalization: Accelerating deep network training by reducing internal covariate shift.</u> _arXiv preprint arXiv:1502.03167_ . 

©Dr. Tony Scanlan 

# Batch Normalisation II 

- Why does Batch Normalisation Improve Training speed (Allows large learning rates LR)? 

- • Experimental results that show BN prevents internal states growing uncontrollably during training (in deeper network layers with larger LR). 

- (a) Training & Test of a RESNET-110 with and without BN at different learning rates. [8] 

[8] Bjorck, N., Gomes, C.P., Selman, B. and Weinberger, K.Q., 2018. Understanding batch <u>normalization. In</u> _Advances in Neural Information Processing Systems_ (pp. 7694-7705). 

- (b) Heatmap of channel means and variances during a divergent gradient update step. [8] 

©Dr. Tony Scanlan 

# Squeeze Excite Network [9] 

- Squeeze-Excite Block 

   - Global Average pooling 

   - 2 FC layers & activation 

   - Weighing applied to original input. 

- Allows channel interdependencies to be described & utilised. 

- Computationally Lightweight 

- Can be inserted into ResNet/Inception 

   - ~10% Improvement in ImageNet (Top-5) for ResNet-50 

[9] Hu, J., Shen, L. and Sun, G., 2018. Squeeze-and-excitation networks. In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 7132-7141). 

- (a) Squeeze Excite Block 

©Dr. Tony Scanlan 

# DenseNet I 

- Dense Connectivity All layers connected to all subsequent layers [10] (within Dense Bock). 

   - Short paths avoid exploding/vanishing gradients. 

- Fewer parameters than ResNet despite additional connections. 

   - (upto) 90% Less parameters[10] 

   - Less redundancy in representation 

- Narrow (but Deep) architecture 

   - Low number of (kernels) channels in each (CF) block. 

   - Combine shortcut layers by concatenation not summation. 

      - Large number of Output Feature Maps due to concatenation. 

### (a) DenseNet, Dense Block and Dense Connectivity 

[10] Huang, G., Liu, Z., Van Der Maaten, L. and Weinberger, K.Q., 2017. Densely connected <u>convolutional networks. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 4700-4708). 

©Dr. Tony Scanlan 

# DenseNet II 

- Composite Function (DenseNet-B) 

   - Use of Batch Norm 

      - Note Different Order of BN,ReLU Conv [11]. 

   - Concatenation of input and output unlike sum in ResNet. 

   - Narrow architecture. 

      - k is typically 32. 

      - ResNet see 64, 246 channels. 

   - Use of bottleneck layer to reduce computation. 

[11] He, K., Zhang, X., Ren, S. and Sun, J., 2016, October. <u>Identity mappings in deep residual networks. In</u> _European conference on computer vision_ (pp. 630-645). Springer, Cham. 

- (a) DenseNet-B Composite Function
