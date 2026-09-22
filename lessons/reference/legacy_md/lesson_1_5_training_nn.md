# lesson_1_5_training_nn

Deep Learning for Computer vision 

Training Neural Networks Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Supervised Learning 

- Most common training paradigm in computer vision. 

- • Formalisation of Learning from Examples. 

   - Learn from labelled examples in <u>Training</u> set. 

   - • Classifier (model) parameters updated during training iterations. 

   - Predict unlabelled examples in test set (after training). 

- Key Components 

   - Dataset. 

   - • Representation. • Classifiers. 

©Dr. Tony Scanlan 

# Deep Learning 

- Deep Learning 

   - Use of Deep Convolutional Neural Networks in Vision. 

   - Deep implies many Convolution layers. 

- <u>Learns End to End.</u> 

   - Many Simple & also Complex Features are . 

   - <u>automatically learned</u> 

      - (a) Example of a Deep neural Network with Convolutional and Fully Connected Layers 

   - Classifier also learns simultaneously. 

- Needs lots of data/computation. 

©Dr. Tony Scanlan 

# Supervised Learning for Deep NN 

- With Neural Network Models. 

   - Widely Different Loss function for different Tasks. 

   - Most Optimisation Algorithms for NN are variations of gradient descent. 

   - Deep Learning Frameworks help automate training. 

<!-- Start of picture text -->
𝐽 𝜃<br><!-- End of picture text -->

©Dr. Tony Scanlan 

# Datasets 

- Image Datasets needed for training ML Classifiers. 

   - Demand Driven by data hungry Deep Learning! 

- Many Datasets available in vision space. 

   - Small 

      - MNIST (Handwritten digits) 

      - • CIFAR 10/100 (Image Classes) 

   - Large 

      - ImageNet<sup>(i)</sup> . (Classification/Obj. Det) 

      - • Microsoft Coco<sup>(ii)</sup> . (Object Det/Seg) 

   - Specialised 

      - Cityscapes<sup>(iii)</sup> (Driving segmentation) 

      - CelebA<sup>(iv)</sup> (Facial Attributes) 

- (a) Images from ImageNet LSVRC<sup>(i)</sup> 

   - (b) Microsoft Common Objects in Context<sup>(ii)</sup> 

- (i) ImageNet Large Scale Visual Recognition Challenge (ii) COCO Dataset 

(iii) The Cityscapes Dataset (iv) CelebFaces <u>Attributes Dataset</u> 

©Dr. Tony Scanlan 

# Binary Cross Entropy Loss 

- Example Simple Image Classification “Cats” vs “Dogs” 

   - Ground truth label supplied as binary value 𝑦𝑖 = 0,1 

- Sigmoid Function 

• Choose this as final activation to get range 0->1 1 ො𝑦𝑖 = 𝜎 𝑧𝑖 = 1 + 𝑒<sup>−𝑧𝑖</sup> e.g. ො𝑦𝑖 = 0.1 

**(1)** 

• Loss: Binary Cross-Entropy Loss 𝑚 1 𝐽 𝜃= − 𝑦𝑖 log ො𝑦𝑖 + 1 −𝑦𝑖 log 1 −ො𝑦𝑖 𝑚<sup>෍</sup> 𝑖=1 

**(2)** 

Where 𝑧𝑖 is the output of the last neuron for the 𝑖<sup>𝑡ℎ</sup> sample, 𝜎 𝑧𝑖 is the corresponding activation value, ො𝑦𝑖 is the predicted label, 𝑦𝑖 is the ground truth label, 𝑚 is the number of samples in the dataset, 𝜃 is the parameters of the model. 

- (a) Deep Neural Network Model for Binary Classification 

(b) Range of possible loss values given 𝑦𝑖 =1 (image source: ml-cheatsheet) 

©Dr. Tony Scanlan 

# Categorical Cross Entropy 

- Example Image Classification (multi Class). 

• Softmax Output (N-classes) 𝑒<sup>𝑧𝑘</sup> ො𝑦𝑖,𝑘 = 𝜎 𝑧𝑘 = σ𝑁𝑗=1 𝑒<sup>𝑧𝑗</sup> ො𝑦𝑖 = 0.1,0.05,0.1,0.6,0.05,0.1 

**(1) (2)** Inner Product 

• Loss: Categorical Cross Entropy 𝑚 1 𝐽 𝜃= − 𝑦𝑖 ∙𝑙𝑜𝑔 ො𝑦𝑖 𝑚<sup>෍</sup> 𝑖 

- Ground truth label supplied as “One Hot” vector 𝑦𝑖 = 0,0,0,1,0,0 

©Dr. Tony Scanlan 

# Gradient Descent (GD) Algorithm 

- GD Algorithm: 

   - Minimise Loss function 𝐽 𝜃 . 

   - Move in direction of steepest gradient. 

      - Update Network Parameters: 𝜕𝐽 𝜃 

      - 𝜃𝑗 ←𝜃𝑗 −𝜂 𝛿𝜃𝑗 

**(1)** 

      - (a) Gradient Descent with simple single parameter (weight) model<sup>(i)</sup> . (J(θ) is the loss function that we want to mimimise) 

- Learning Rate 𝜂 must be sufficiently low for convergence. 

   - Too high and minimum can be missed. 

   - Too Low and training will take too long. 

(i) Image (a) (ii) Image (b) 

- (b) Training Accurcy vs Gradient Descent iterations with Different Learning Rates<sup>(ii)</sup> . 

time 

©Dr. Tony Scanlan 

# Gradient Descent with Backpropagation 

- GD with BP Algorithm: 

   - Pass Training data Forward through NN 

      - Obtain predicted labels. 

   - Compute Loss function 𝐽 𝜃 

      - Difference between NN predicted output (labels) and True output (labels). 

      - Minimised by Gradient Descent during training 

   - Perform Back-propagation 

      - Efficiently Finds gradient values of parameters within network wrt Loss function. 

   - Parameter Update 

      - Update parameters. 

- All these steps are automated within Deep learning Frameworks e.g. Tensorflow/Keras or pytorch 

- Computationally Demanding! 

- (a) Supervised Learning with Gradient Descent Algorithm & Backpropagation. 

More detail: “What is Backpropagation really <u>doing?</u> _YouTube_ video, Accessed 12/01 2022. 

©Dr. Tony Scanlan 

# Batch size in Gradient Descent 

- Batch size. 

   - How much training data to give the network during each update of the Gradient descent Algorithm<sup>(ii)</sup> 

- Batch Gradient Descent: 

   - ALL Examples! 

   - Slow to run for large datasets. (potential huge memory requirement) 

- Stochastic Gradient Descent: 

   - 1 Different Example per GD iteration 

   - Noisy convergence. (Escapes local minimum) 

   - Frequent model updates computationally inefficient. 

- Mini Batch Gradient Descent: 

   - Compromise Approach. 

   - Takes advantage of GPU parallel computing during training. 

- (a) Supervised Learning with Gradient Descent Algorithm 

<!-- Start of picture text -->
Stochastic<br>Mini Batch GD<br>GD<br>Batch<br>GD<br><!-- End of picture text -->

- (b) Comparision of Covergence of Batch GD, Stocastic GD and Mini Batch GD. (2D Example) 

(iii) <u>A gentle Introduction to Mini-Batch gradient Descent and How to Configure Batch Size</u> 

©Dr. Tony Scanlan 

# SGD with Momentum 

- Helps to increase speed of (mini-batch) SGD 

   - Dampens effect of fluctuations in gradient. 

   - Helps to escape local minima or flat regions. 

• Change gradient descent update rule 𝜕𝐽 𝜃 • SGD: 𝜃𝑗 ←𝜃𝑗 −𝜂 **(1)** 𝛿𝜃𝑗 

   - Now have “velocity” as weight update in SGD with momentum 𝛾 . [1] 𝜕𝐽 𝜃 

   - • 𝑣𝑡 = 𝛾𝑣𝑡−1 + 𝜂 , 𝜃𝑗 ←𝜃𝑗 −𝑣𝑡 **(2)** 𝛿𝜃𝑗 

- Parameters still need tuning: 

(a) SGD with Momentum (1d example) 

- Setting momentum 𝛾 close to 1 leads to larger updates but may have convergence issues. 

- Increase momentum during training with policy. 

- Many DNN trained using SGD with Momentum. 

[1] Qian, N., 1999. <u>On the momentum term in gradient descent learning algorithms.</u> _Neural networks_ , _12_ (1), pp.145-151.
