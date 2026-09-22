# lesson_1_7_overfitting_regularisation

Deep Learning for Computer vision 

Overfitting and Regularisation Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Introduction 

- Review key ML theory about classifiers 

   - Bias-Variance Trade Off 

   - Underfitting/Overfitting 

   - Generalisation 

- Classification: 

   - Want to map from input data to labels. 

   - Model can be interpreted as finding a decision boundary between classes. 

- Viewpoint of Neural network as a classifier 

   - Overfitting & Regularisation 

- (a) Classifier Decision boundary for 2D data with 2 classes. 

©Dr. Tony Scanlan 

# Bias Variance Trade off 

- Types Classifier Error: • Bias: Inherent Error due to model (e.g. linear classifier) 

   - Variance: How much classifier changes with different training data. 

- If the Classifier model is too simple get “Underfitting”. • High Bias, Low Variance 

- If the classifier model is too complex get “Overfitting”. • Low Bias, High Variance 

(a) Classifier Error vs Model Complexity, showing Bias vs Variance Trade Off and Under and Over fitting regions. 

©Dr. Tony Scanlan 

# Model Training 

- Detect under/Overfitting During Training. 

- Mitigate “Underfitting” 

   - Increase model Complexity. 

- Mitigate “Overfitting”. 

   - Large Dataset Sizes 

   - Decrease model complexity. 

   - Regularise the parameters. 

- (a) Error observed during training for Under/Over fitting models. 

(vi) Overfitting vs. Underfitting: A Complete Example 

©Dr. Tony Scanlan 

# Generalisation 

• Generalisation: Ability of Classifier System to perform well on unseen test data. 

   - Test data never used during training. 

- Good Generalisation • If model is not underfitting or strongly overfitting. 

©Dr. Tony Scanlan 

# The curse of dimensionality 

- Image Data has large number of dimensions (e.g. 256 x 256 = 65536) • Data is separable • Data points become “sparse” with more data dimensions. 

- • Likely to have overfitting. 

- Curse of dimensionality 

   - Total amount Training Data must increase exponentially with each dimension to keep generalisation constant 

- Implication for Neural Networks • These have large input dimensionality. • Convolution operates locally • Feature extracted before classification • Effectively reduces dimensionality 

©Dr. Tony Scanlan 

# Neural Network Overfitting 

- Neural Networks are over parameterised 

   - Will tend to strongly overfit during training 

- Solutions 

   - Early stopping 

      - Stop training once validation set performance stops improving. 

   - Regularisation 

      - Add penalty to loss function reduce model complexity. 

   - Dropout: 

      - Remove connections during training to reduce reliance on single neurons. (regularisation) 

   - Batch Normalisation 

      - Additional layer in many Deep Neural Networks that improves training time. (regularisation) 

   - More Training Data 

      - Use data augmentation techniques to increase training data pool size. 

## (a)Early Stopping<sup>(iv)</sup> 

(b)Dropout 

(iv) Deep neural networks: preventing overfitting 

©Dr. Tony Scanlan 

# L1 & L2 Regularisation 

- Penalises large values of parameters 

   - Large parameter values associated with overfitting 

- L1 regularisation 

   - Cost proportional to Absolute value of weights 

- 𝑀 

- Loss = Loss + 𝑙1 × σ𝑖=0 𝑤𝑖 

      - **(1)** 

- L2 regularisation 

   - Cost proportional to Square of weights 

Loss = Loss + 𝑙2 × σ𝑀𝑖=0 𝑤𝑖<sup>2</sup> 

**(2)** 

- Can be applied to Kernel, Bias or activation 

   - Convolution or Dense layers 

Value of regularisation 

layers.Dense(512, activation=‘relu', kernel_regularizer=regularizers.l2(0.001)) 

©Dr. Tony Scanlan 

# Dropout 

- Random Removal of neural network Nodes during training 

   - Regularises weights as network cannot rely on single paths. 

   - Nodes removed with probability 0->1 

- Dropout layer 

   - Applied to output of dense layer layers.Dense(512, activation=‘relu'), layers.Dropout(0.5) 

Dropout rate 

[1] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R., 2014. <u>Dropout: a simple way to prevent neural networks from overfitting.</u> _The journal of machine learning research_ , _15_ (1), pp.1929-1958. 

©Dr. Tony Scanlan 

# Spatial Dropout 

- <u>Recommended for Convolution</u> 

   - Regular dropout won’t have any effect as removes correlated pixels 

Kernels Regular dropout removed pixels 

- Spatial Dropout removes entire feature maps 

Kernels 

Spatial 2D dropout (Feature map removed) 

©Dr. Tony Scanlan 

# Data Augmentation 

- Regulation achieved with more data 

   - Labelled Datasets finite 

   - Expensive to label new data 

- Data Augmentation 

   - Modify existing images to increase number of examples 

   - Simplest approach to use affine transforms. 

      - Perform well with smaller datasets 

   - More complex augmentations exist. 

- Keras implementation of data augmentation. 

- (a) Simple data Augmentations: Rotations, translation, flipping, scaling, shear transforms<sup>(ii)</sup> 

def aug_model(input_shape): 

inputs = Input(shape=input_shape) x= tf.keras.layers.RandomFlip(mode="horizontal")(inputs) x= tf.keras.layers.RandomZoom((-0.2,0.2))(x) 

x= tf.keras.layers.RandomTranslation(height_factor=0.2, width_factor= 0.2,fill_mode='nearest')(x) 

model = Model(inputs=inputs, outputs=x) return model 

(b) Use of Keras Preprocessing layers to perform data augmentation. 

(ii) <u>Data Augmentation in Deep Learning, Author: Valentina Alto</u> (iii) <u>Tensorflow/Keras Data Augmentation Tutorial</u>
