# lesson_1_2_FC_layers

Deep Learning for Computer vision 

Fully Connected Layers/Classifiers Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Single Neuron 

• Single Neuron Function 𝑛 𝑦𝑖 = 𝜎 𝑥𝑖𝑤𝑖 + 𝑏 ෍ 𝑖=1 

   - 𝑥𝑖 are the inputs, 𝑤𝑖 are the weights, 𝑏 is the bias, 𝜎 is the activation function. 

- Activation function 𝜎 adds nonlinearity to overall network function => allows complex models. 

   - (a) One Neuron = Weights, Sum and Activation. 

- E.g. Image Classification. 

©Dr. Tony Scanlan 

# Activation Functions 

- Activation functions. 

   - Sigmoid/tanh 

• Still used (binary outputs), Slow to compute, Saturates. 

- ReLU/Leaky ReLU 

   - Rectified Linear Unit 

   - Default activations in Deep Learning. 

- (a) Common Activation functions<sup>(i)</sup> . 

(i) Introducing Activation Functions for Deep Learning, Author: Shuti Jadon 

©Dr. Tony Scanlan 

# Fully Connected Layer 

- One Hidden Layer Neural Network (NN): 

   - Input layer (just inputs to network.) 

   - Hidden layer and output layer contain neurons. 

   - Can have any number of neurons, hidden layers. 

- Interpret neural network as a mathematical function 𝑓 mapping input 𝑥𝑖 to the output 𝑦𝑜𝑢𝑡 𝑦𝑜𝑢𝑡 = 𝑓 𝑥1, 𝑥2, 𝑥3 

- NN has an internal “model” of the data. 

- Can use as a (probabilistic) classifier. 

- (a) One Hidden Layer Neural Network. 

©Dr. Tony Scanlan 

# NN Binary Classifier 

- Neural Networks as a binary Classifier. • Using the sigmoid activation function at the output we can create a binary classifier. 

   - Output from last neuron is interpreted as probability value between 0 and 1. 

(a) Binary Classification with Neural Networks 

©Dr. Tony Scanlan 

# NN Multi-Class Classifier 

- Multi Class NN Classifiers. 

   - Last layer has same number neurons as classes. 

   - Add Softmax activation to last layer network. 

      - =>Outputs a probability value for each class. 

         - <u>Probabilities sum to 1.</u> 

         - Assume class with highest probability is what is identified in image. 

- (a) Multi-Class Classification with Softmax Activation 

©Dr. Tony Scanlan 

# Softmax Activation 

Given the inputs to the softmax layer are: Q: 𝑧1 = 4 , 𝑧2 = 3 , 𝑧3 = 3 , 𝑧4 = 2 What are the output probabilities 𝜎 𝑧 𝑖 ? 

## • Softmax Activation 

𝑒<sup>𝑧𝑖</sup> 

𝑒<sup>𝑧𝑖</sup> Exponential each input • 𝜎 𝑧 𝑖 = 𝐾 𝑒<sup><u>𝑧𝑗</u></sup> σ𝑗=1 Sum Exponential each input for 𝑖= 1, . . 𝐾 and 𝒛= 𝑧1, 𝑧2 … 𝑧𝑘 

A: Work out the exponential of each input: 𝑒<sup>𝑧1</sup> = 54.6 , 𝑒<sup>𝑧2</sup> = 20.1 , 𝑒<sup>𝑧3</sup> = 20.1 , 𝑒<sup>𝑧4</sup> = 7.4 Get the sum of the exponentials : 𝐾 𝑒<sup>𝑧𝑗</sup> = 54.6 + 20.1 + 20.1 + 7.4 = 102.2 ෍ 𝑗=1 

Work out the output probabilities 

𝜎 𝑧 1 = 54.6/102.2= 0.53, 𝜎 𝑧 2 = 20.1/102.2= 0.20 𝜎 𝑧 3 = 20.1/102.2= 0.20, 𝜎 𝑧 4 = 20.1/102.2= 0.07 Check: Probabilities sum to 1 

(a) Multi-Class Classification with Softmax output (showing Probability values) 

©Dr. Tony Scanlan 

# Calculating Parameter Numbers 

- Each neuron in layer is connected to all Inputs. 

   - Bias for each neuron. 

- If connecting to output of a convolution layer 

   - => must “Flatten” to 1D vector 

- Example Fig (b) 

   - Output vector size N (1D) 

      - N neurons 

   - Fully Connected (connects to inputs) ⇒ <u>n × 𝑚</u> × 𝐶 × 𝑁 + 𝑁 

- (a) Flattening of Feature maps to a 1D vector. 

- (b) Flattened Layer applied to Fully Connected Layer with N outputs. 

Weights 

For Bias
