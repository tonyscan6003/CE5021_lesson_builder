# lesson_1_4_cnns

Introduction to Convolution Neural Networks (CNNs) Instructor: Dr. Tony Scanlan 

©Dr. Tony Scanlan, University of Limerick 

# Convolutional Neural Network (CNN). 

- Convolution layers 

   - Acts as a neural network (with activation) 

   - Learns Image Features during training. 

- Fully Connected Layers also used in Deep CNN 

   - Act as classifier stage 

   - (a) Example of a Deep neural Network with Convolutional and Fully Connected Layers 

- Less Computationally Efficient than Convolution 

## ©Dr. Tony Scanlan, University of Limerick 

# Processing Images with Fully Connected NN 

- Naïve Approach to use Fully Connected NN • Image reshaped as a vector 

   - Each pixel connected to each neuron. 

      - Typical Images 256 x 256 = 65536 pixels 

      - Each connection corresponds to a parameter that must be multiplied against each pixel. 

- Too much memory & computational resources needed. 

- => Impractical to Implement 

©Dr. Tony Scanlan, University of Limerick 

# Why use CNNs 

- Convolution (2D) traditionally used to process images with filters 

- Operates on (n x n) regions of neighbouring pixels (local) 

   - Efficient Operation (same parameters) 

- Convolution is a prior 

   - Knowledge about problem that are coding into the architecture 

   - Kernels are learned 

   - Works well!! 

- (a) Blurring of Image using Gaussian Filter 

(b) Convolution Operates on local regions of pixels (3 x 3 in example shown). 

## ©Dr. Tony Scanlan, University of Limerick 

# Convolution as a Neural Network 

- Simple example with 3 x 3 image and 2 x 2 kernel 

- 4 kernel positions. 

   - Each multiply and accumulate of with kernel with image viewed as part of a neural network. 

   - Include activation + bias 

- Resulting Network 

   - Not Fully Connected (local only) 

   - • Same weights used for each neuron 

- Within 2D convolution don’t have bias. 

   - But do include in CNNs as additional parameter (each feature map) 

(a) Convolution neural network (note weights are the same for each neuron, bias not shown) 

©Dr. Tony Scanlan, University of Limerick 

# Structure of a CNN 

- Convolution Layer • Performs Convolution 

- Activation Layer • Often shown explicitly 

- • Pooling Layer 

- Summarises Features 

- • Reduces spatial size/computation 

- • “Flatten” before Fully Connected 

- Fully Connected Layer • Uses Softmax activation to produce class probabilities 

(a) Example of basic CNN with Common Layers 

## ©Dr. Tony Scanlan, University of Limerick 

# Learned Features 

- Helpful to think of Convolution layers in CNNs performing Cross Correlation with kernels 

- CNN responds to ‘features’ in Images. 

   - Early Convolutions respond to Simple Features 

   - After multiple Layers, respond to complex features (i.e. objects) 

- These Features are Learned! 

- (a) Visualisation of Features From Convolution Layers of VGG-16 [1] 

[1] <u>Mahendran, A. and Vedaldi, A., 2016. Visualizing deep convolutional neural networks using natural pre-images.</u> _<u>International Journal of Computer Vision</u>_ <u>,</u> _<u>120</u>_ <u>(3), pp.233-255.</u> 

©Dr. Tony Scanlan, University of Limerick 

# Receptive Field 

- Receptive field<sup>(i)</sup> : Size of region in the input neuron can view (or responds to). 

- Important for object recognition. • Receptive field large enough for large objects. 

- Receptive Field determined by preceding convolution and pooling layers. 

- Receptive field size increases for deeper layers. 

(i) Computing Receptive Fields of Convolutional Neural Networks 

(a) Simple CNN, (b) Receptive field for neuron B, (c) Receptive Field for neuron C 

©Dr. Tony Scanlan, University of Limerick 

# Translational Invariance 

- Translational Invariance: Same response if image is translated 

- Convolution layers automatically build in translational invariance 

- Max pooling 

   - Achieves partial invariance to small 

   - translations 

   - Max of a region depends only on the single largest element. 

   - Small translation may not change max element in pooling window. 

(Partial) Invariance to Local Translation with Max Pooling 

- => Same Output 

- => (Partial) Invariance to local translation. 

## ©Dr. Tony Scanlan, University of Limerick 

# Simple CNN 

- Le-Net 5. (1998) Early Convolutional Neural Network. 

   - Designed to recognise handwritten characters 

   - <u>MNIST dataset (link)</u> 

   - Only 7 layers. (Not Deep) 

|**layer**|**Type**|**Channels**<br>**(Feature**<br>**Maps)**|**Size**|**Kernel**<br>**Size**|**Stride**|**Activation**|
|---|---|---|---|---|---|---|
|Input|Image|1|32 x 32|-|-|-|
|1|Conv|6|28 x 28|5 x 5|1|Tanh|
|2|Avg. Pool|6|14x14|2 x 2|2|-|
|3|Conv|16|10 x 10|5 x 5|1|Tanh|
|4|Avg. Pool|16|5 x 5|2 x 2|2|-|
|5|FC|-|120|-|-|Tanh|
|6|FC|-|84|-|-|Tanh|
|7|FC|-|10|-|-|Softmax|

(a) Sample of MNIST Dataset 

(b) Le-Net 5 layers & Table
