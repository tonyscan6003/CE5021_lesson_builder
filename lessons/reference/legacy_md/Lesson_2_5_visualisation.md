# Lesson_2_5_visualisation

Deep Learning for Computer vision 

Deep Neural Network Failure, Visualisation & Adversarial Attack Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Introduction 

      - (a) Reconstruction of Monkey Image from VGG-VD [1] 

- What do Convolution Neural Networks Learn? 

   - Explainable Neural Networks 

- Reconstruction (Inversion) 

- Visualisation 

   - Activation Maximisation 

- Style Transfer 

   - Produce new composite images 

   - Content from photograph In Style of Artwork 

- Deep Dreams 

   - Visualise Found Features on Image 

(i) Deep Dream Tensorflow tutorials 

- (b) Visualisation of learned Features VGG-16 [1] 

- (c) Examples of Style transfer B between content image A artwork (inset) [2] 

- (d) Example Deep Dreams<sup>(i)</sup> 

©Dr. Tony Scanlan 

# Reconstruction (Inversion) 

- Determine information about image retained by Trained Network. 

- Algorithm 

   - Given Trained Network, target image. 

   - Obtain & store features of image at layer X 

   - Apply random noise image to network and obtain features at same layer. 

   - Obtain Loss between 2 sets of features. 

   - Apply gradient descent to update noise image until convergence 

   - (a) Reconstruction Algorithm for Conv 3 layer 

- Apply regularisation to get “natural” image 

- Early Layers enable accurate reconstruction 

- Later layers form a composition of parts similar (but not identical to original image) 

[1] Mahendran, A. and Vedaldi, A., 2016. <u>Visualizing deep convolutional neural networks using natural pre-images.</u> _International Journal of Computer Vision_ , _120_ , pp.233-255. 

(b) Reconstructions from layers of VVG-16 [1] 

©Dr. Tony Scanlan 

# Visualisation with Activation Maximisation 

- Algorithm 

   - Given trained Neural Network 

   - Start with random noise input. 

   - Gradient Ascent Algorithm maximises output of neuron (or Layers/Channels) by adjusting pixels of image at input. 

   - Additional Regularisation to get a “Natural” looking image. 

- Demonstrates the Learned Features. 

   - Early layers learn simple Features 

   - Features in later layers are extremely complex. 

- (a) Activation Maximisation method for visualisation of Layers (or Neurons/Channels/Layers) of Deep Neural Network. 

Note: These Features are per Neuron. Sizes relate to receptive field 

(b) Activation Maximisation for Convolution Layers of VGG-16 [1] 

©Dr. Tony Scanlan 

# Feature Visualisation 

- Visualising the Convolution Layers. • Helps understand exactly what the network is looking for in input images. 

      - Very Important. 

   - Can compare examples from the training dataset that also activate the same point in the feature map. 

- (a) Dataset Examples & Activation Maximisation for specific Convolution Layers of GoogLeNet<sup>(ii)</sup> 

(ii) Feature Visualisation 

©Dr. Tony Scanlan 

## Style Transfer Concepts 

- Insight in to representations learned by CNNs 

   - Reconstructions lower layers produce exact pixel values 

   - • Higher layers content (objects/relationship). 

- Style from Textures 

   - Inner product feature maps give correlation information (Texture information). 

[2] Gatys, L.A., Ecker, A.S. and Bethge, M., 2016. Image style transfer using <u>convolutional neural networks. In</u> _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 2414-2423). 

Figure (a). Content and Style Information in a Neural Network 

©Dr. Tony Scanlan 

# Training Style Transfer 

- Algorithm 

   - Start with Trained VGG-16 Network, Required Style & Content Images and Random Noise Image. 

   - Obtain feature map outputs for Style & Content Images and store. 

      - Get Inner Product for Style Features to obtain texture features 

   - Apply Gradient Descent with Back Propagation to Network with Noise Applied. 

      - Obtain Style and Content Losses 

      - Update Noise Image values only on each iterations 

   - Converge to Final Image 

(a) Training Process For Style transfer (Note: Network shown 3 times, however only one network is used. Content and Style Images are passed through Individually and values of features maps stored and reused on each iteration to form Loss functions.) 

©Dr. Tony Scanlan 

# Deep Dreams 

- Similar to Activation Maximisation 

   - Now use a content Image 

   - Force Network to amplify Features for target layers. 

- Specific Images bias network towards specific Features 

- Apply Algorithm Iteratively 

   - Zoom after each iteration 

   - Can start on random noise. 

### (a) Training Deep Dreams 

   - (c) Images produced by Iterative Applications of Deep Dream Alogrithm<sup>(iii)</sup> 

- (b) Demonstrating Feature Bias for specific Inputs<sup>(iii)</sup> 

(iii) Inceptionism <u>Going Deeper into Neural Networks, Alexander Mordvintsev 2015</u> 

©Dr. Tony Scanlan 

# Adversarial Attack 

- Neural Networks can be easily Tricked into misclassifying inputs. 

- Force the “wrong” Neuron to be activated. 

   - Disturbingly only a small amount of noise needs to be added to cause misclassification. 

   - Due to structure of neural network many small changes to the input result in large changes to the output. 

   - (a) Adversarial Example by adding noise to the input [3] 

- Adversarial attacks that affect one network affect others aswell 

[3] Goodfellow, I.J., Shlens, J. and Szegedy, C., 2014. Explaining and harnessing adversarial examples. _arXiv preprint arXiv:1412.6572_ . 

©Dr. Tony Scanlan 

# Real Adversarial Attack 

- Adversarial Attacks have been demonstrated in the real world! 

   - Attack robust to angle and distance. 

- => Implications for safety critical applications [5] 

[4] Eykholt, K., Evtimov, I., Fernandes, E., Li, B., Rahmati, A., Xiao, C., Prakash, A., Kohno, T. and Song, D., 2018. Robust <u>physical-world attacks on deep learning visual classification.</u> In _Proceedings of the IEEE conference on computer vision and pattern recognition_ (pp. 1625-1634).[5] “A Tiny piece of <u>tape tricked Teslas into speeding up to 50 MPH” Brian Barrett, 22nd Feb 2020, Wired.com.</u> 

- (a) Example of Adversarial Attack in real world. (Stop sign misclassified as Speed Limit Sign 80% of time) [4] 

©Dr. Tony Scanlan 

# Lack of Generalisation 

- Neural Networks can be easily fooled by changes in pose of objects [6]. 

- Rotations and changes in pose of Images that are not in Training Dataset. 

   - Network fails to generalise to these novel situations. 

- Don’t even need adversarial examples to get adverse behaviour from neural networks. 

[6] Alcorn, M.A., Li, Q., Gong, Z., Wang, C., Mai, L., Ku, W.S. and Nguyen, A., 2019. Strike <u>(with) a pose: Neural networks are easily fooled by strange poses of familiar objects.</u> In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ (pp. 4845-4854). 

Google Inception Network Image Classification (a) Correct Label produced with object in expected pose. (c-d) Fails to recognise objects in different poses.
