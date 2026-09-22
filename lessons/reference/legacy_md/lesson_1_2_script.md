# lesson_1_2_script

# Fully Connected Layers/Classifiers. 

Instructor: Dr Tony Scanlan. 

## Slide 1: Title Slide 

Hello and welcome to this video. In this lesson, we will discuss some neural network basics including neurons, activation functions, fully connected networks and neural network classifiers. 

## Slide 2: Single Neuron 

To begin, we will discuss the structure of a single artificial neuron shown in figure (a). The neuron multiplies each of its inputs with a unique weight corresponding to that input. Therefore, the input 𝑥1 will be multiplied by 𝑤1 and 𝑥2 by 𝑤2 etc. These multiplications are summed as shown in the corresponding equation for the neuron. We also have an  additional bias term 𝑏 which is added to the sum. After summation a mathematical function sigma known as the activation function is applied to the result, this function is normally non-linear, we will see examples of activation functions in the next slide. The purpose of the activation function is to add non-linearity to the overall network function. The addition of activation functions to the neurons allows a neural network to learn complex functions or models for challenging tasks, like image recognition. 

## Slide 3: Activation Functions 

We will now discuss the non-linear activation functions that are commonly used in neural networks, in early neural networks the sigmoid and tanh functions were mostly used. These are still used today predominately if a binary output is required from a network. They have the disadvantage of being slow to compute and can saturate which leads to problems during training. The Rectified Linear units and leaky rectified units are the default activations used today in deep learning. They are fast to compute and don't saturate, during training. 

## Slide 4: Fully Connected Layers 

On this slide we will discuss a very simple neural network, the one hidden layer neural network shown in figure (a). This consists of 3 layers. An input layer, which are just the inputs to the network, a hidden layer, and an output layer. Both the hidden layer and output layer contain neurons. Each neuron multiplies each of its inputs by a weight, sums the result and applies an activation function to the sum. The arrows show the connections between the inputs or neurons in each of the layers. This type of network is known as fully connected, that is each neuron in a layer is connected to every other neuron or input from the previous layer. We can expand the network to have any number of hidden layers, neurons in each layer and inputs or outputs. We can interpret a neural network as a mathematical function mapping the inputs 𝑥𝑖 to the output 𝑦𝑜𝑢𝑡. The neural network can build an internal model of data applied at the inputs and can therefore act as a probabilistic classifier. The term probabilistic classifier means that neural network outputs the probability of an input belonging to a class rather than directly assigning each input a class label. We can use a Fully connected type network with hidden layers to perform simple computer vision tasks, such as handwritten digit recognition. 

## Slide 5: NN Binary Classifier 

As we have already mentioned, the neural network can be used as a classifier, given a sufficiently complex network with enough layers and neurons the neural network can learn the mathematical function mapping the input image to each class. In the example shown in figure (a) we consider using the two hidden layer classifier as a binary classifier for example distinguishing between pedestrians or background in driving scenes. The output neuron of this network uses the sigmoid activation function, this ensures we will get a result between 0 and 1. If the neural network is confident the image contains a pedestrian we will get a value close to 1, if it is not a pedestrian the value will be close to zero. 

## Slide 6: NN Multi-Class Classifier 

We can extend the simple neural network classifier to multi-classification applications by changing the output of the network. The last layer must have the same number of neurons as the number of classes in our dataset. The neurons in the last layer do not use the usual activation functions (e.g. sigmoid or ReLU) but instead the sum of the weights 𝑤𝑖 times the inputs 𝑥𝑖 from each of the neurons is converted into a normalised probability value by the softmax activation layer. The probabilities always sum to 1. We will assume the class with the highest probability is what is identified in the image. 

## Slide 7: Softmax Activation 

On this slide, we will discuss the Softmax activation function in more detail. As shown in the figure and mentioned in the last slide the Softmax activation outputs a normalised probability for each possible class that sums to 1. The equation for calculating each of the probabilities is shown on the slide. The 𝑧𝑖 are the sum of the weights 𝑤𝑖 times the inputs 𝑥𝑖 from each of the neurons in the last layer. The probability is calculated by dividing the exponential of each 𝑧𝑖 by the sum of the exponential of all the 𝑧𝑖 's. A sample question and answer of how to calculate the probabilities given the input to the Softmax function, is shown on the slide. The exponential of the input values are calculated first, this works out at 54.6 for exp(4), 20.1 for exp(3) which is the same for two of the inputs, and 7.4 for exp(2). The exponentials is summed giving 102.2, we can then easily work out the probabilities by dividing the exponentials by the sum, this give, 0.53, 0.2,0.2 and 0.07. We can check the answer by seeing that the final probabilities sum to one. 

## Slide 8: Calculating Parameter Numbers 

When constructing neural networks it is useful to be able to calculate how many parameters are going to be required by fully connected layers. With a fully connected layer each neuron is connected to all inputs with a parameter or weight associated with each connection and there is also a separate bias for each neuron. Even with relatively few neurons fully connected layers can have a large number of parameters increasing the memory requirements of a large network. Fully connected layers in neural networks are frequently connected to the output of convolution layers, in order to do this we must first “Flatten” the convolution layer to a 1D vector. As shown in figure (a), if we have an image or feature map with a spatial size of n x m and C channels this is formed into a vector of size n x m x C. If we connect this to fully connected layer with N neurons as shown in figure (b), then each of the inputs is connected to each neuron leading to n x m x C x N parameters or weights, we also add a further N parameters for the bias for each neuron.
