<u>View subject</u> | <u>Edit subject</u>

# Image Classification with Deep CNNs

Slides

# **Exercises**

# <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107568

**QUESTION** 

The Top 1 and Top 5 accuracy as Metrics used in the ImageNet Large Scale Visual Recognition Challenge.

Determine if the statements below relating to these metrics are True or False.

In Top 1 accuracy the highest probability prediction must match the ground truth

| ○ True  |   |
|---------|---|
|         |   |
| ○ False | × |

Select one answer. Upon selection it will submit your answer immediately.

| ✓ Feedback for answer: 1                                                                       | 100% of points |
|------------------------------------------------------------------------------------------------|----------------|
| Correct!                                                                                       |                |
| 区 Feedback for answer: 2                                                                       | 0% of points   |
| Incorrect. In Top 1 accuracy, the highest probability predictio indeed match the ground truth. | n must         |

QUESTION

# In Top 5 accuracy the highest probability prediction must match the ground truth.

| ○ True                                                                                                                                               | ×            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|
| ○ False                                                                                                                                              | <b>✓</b>     |
| Select one answer. Upon selection it will submit your answer immediately.                                                                            |              |
| ➤ Feedback for answer: 1                                                                                                                             | 0% of points |
| Incorrect. In Top 5 accuracy, any of the 5 highest probability predictions must match the ground truth, not just the highest probability prediction. |              |
| ✓ Feedback for answer: 2                                                                                                                             |              |
| Correct!                                                                                                                                             |              |
|                                                                                                                                                      |              |

QUESTION

# In Top 5 accuracy the order of the probabilities is important.

| ○ True                                                                                                                                                                            | ×              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| ○ False                                                                                                                                                                           | ✓              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                         |                |
| 区 Feedback for answer: 1                                                                                                                                                          | 0% of points   |
| Incorrect. The order of the probabilities is not important in Topaccuracy. As long as the correct label appears in any of the Topprobabilities, it is considered a True Positive. |                |
| Feedback for answer: 2                                                                                                                                                            | 100% of points |
| Correct!                                                                                                                                                                          |                |

# <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107571

**QUESTION** 

Explain Confusion and Bias in the context of Neural Network based Image Classifiers.

Confusion occurs in deep neural network image classifiers when (select the correct option)

| 0 | The network mislabels similar objects     | <b>✓</b> |
|---|-------------------------------------------|----------|
|   |                                           |          |
| 0 | The network correctly labels all objects. | ×        |
|   |                                           |          |
|   | The network has no training data.         | ×        |

Select one answer. Upon selection it will submit your answer immediately.

✓ Feedback for answer: 1 100% of points

Correct!

☑ Feedback for answer: 2

Incorrect. Confusion occurs when the network mislabels similar objects.

Incorrect. Confusion is related to mislabeling, not the absence of training data.

0% of points

QUESTION

# Bias occurs in neural networks due to (select the correct answer from the options below)

|   | Overrepresentation of certain classes in the training data                                       | ×              |
|---|--------------------------------------------------------------------------------------------------|----------------|
|   | <ul> <li>Underrepresentation of genders or ethnic groups in training images</li> </ul>           | ✓              |
|   | Random initialization of weights.                                                                | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                        |                |
| × | Feedback for answer: 1                                                                           | 0% of points   |
|   | Incorrect. Bias typically occurs due to underrepresentation of cogroups, not overrepresentation. | ertain         |
| V | Feedback for answer: 2                                                                           | 100% of points |
|   | Correct!                                                                                         |                |
| × | Feedback for answer: 3                                                                           | 0% of points   |
|   | Incorrect. Bias is related to the training data, not the initialization weights.                 | n of           |

### <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107573

**QUESTION** 

These questions are in relation to network structure & computation

In a Deep Convolutional Neural Network (such as AlexNet) What layer type dominates the computational requirements? (select from the options below)

Incorrect. Pooling layers do not dominate the computational requirements. (Pooling just operates across each individual feature map, in contrast convolution operations in a CNN usually have multiple kernels operating across all the feature maps.)

QUESTION

# In a Deep Convolutional Neural Network (such as VGG-19) What layer type dominates the memory requirements?

| ○ Convolution layers                                                                                                                                                        | ×              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| Fully Connected Layers                                                                                                                                                      | ✓              |
| O Pooling Layers                                                                                                                                                            | ×              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                   |                |
| 区 Feedback for answer: 1                                                                                                                                                    | 0% of points   |
| Incorrect. Fully connected layers dominate the memory require (Convolution layers effectively share weights as the same convokerel is applied across the input feature map) |                |
| ☑ Feedback for answer: 2                                                                                                                                                    | 100% of points |
| Correct!                                                                                                                                                                    |                |
| 区 Feedback for answer: 3                                                                                                                                                    | 0% of points   |
| Incorrect: Pooling Layers have no parameters associated with t                                                                                                              | hem!           |
|                                                                                                                                                                             |                |

QUESTION

# Why does the VGG family of networks have such a large number of parameters? (select the correct response from the options)

|   | Use of large fully connected layers at the output.                                                                                                                          | ✓              |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | ○ Use of small 1 × 1 convolutions.                                                                                                                                          |                |
|   | O'Se of Small 1 × 1 convolutions.                                                                                                                                           |                |
|   | Absence of pooling layers.                                                                                                                                                  | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                                                                   |                |
| V | Feedback for answer: 1                                                                                                                                                      | 100% of points |
|   | Correct! The very large fully connected layers in the VGG family contribute to the parameter count.                                                                         | greatly        |
| × | Feedback for answer: 2                                                                                                                                                      | 0% of points   |
|   | Incorrect. The large number of parameters is due to the use of I fully connected layers. (Small 1 $\times$ 1 convolutions are a very efficiway to reduce parameter counts.) |                |
| × | Feedback for answer: 3                                                                                                                                                      | 0% of points   |
|   | Incorrect: Pooling layers are used in VGG, in any case they don't contribute to the parameter count.                                                                        |                |
|   |                                                                                                                                                                             |                |

QUESTION

# Why are stacked layers of 3 × 3 convolutions computationally efficient?

|   | 0      | They produce larger receptive fields equivalent to much larger convolutions. | ✓              |
|---|--------|------------------------------------------------------------------------------|----------------|
|   | 0      | They reduce the number of parameters.                                        | <b>V</b>       |
|   |        | They eliminate the need for activation functions.                            | ×              |
|   | Select | one answer. Upon selection it will submit your answer immediately.           |                |
| ~ | Feedba | ck for answer: 1                                                             | 100% of points |
|   | Corr   | ect!                                                                         |                |
| ~ | Feedba | ck for answer: 2                                                             | 100% of points |
|   | Corr   | ect. Stacked 3 × 3 convolutions also help reduce computa                     | tion.          |
| × | Feedba | ck for answer: 3                                                             | 0% of points   |
|   | Inco   | rrect. Activation functions are still needed.                                |                |
| _ |        |                                                                              |                |

QUESTION

# What is the reasoning behind the network in network concept?

| To increase the representational power of the network  | $\checkmark$ |
|--------------------------------------------------------|--------------|
|                                                        |              |
| To eliminate the need for convolutional layers         | ×            |
|                                                        |              |
| To eliminate the need for large fully connected layers | ✓            |

Select one answer. Upon selection it will submit your answer immediately.

#### ✓ Feedback for answer: 1

100% of points

Correct! The reasoning behind the network in network concept is that the representational power of regular convolutions may be low as convolution is a linear operation. By adding a small fully connected network into the convolution the number of parameters are increased locally and the representational power of the network increased.

#### × Feedback for answer: 2

0% of points

Incorrect! Convolution layers are still used in the network in network concept, they are coupled with small fully connected networks to increase the representational power.

### ✓ Feedback for answer: 3

100% of points

Correct! By adding a small fully connected network into the convolution the number of parameters are increased locally and the representational power of the network increased. This reduces the need for large fully connected layers at the output of the network, instead the learned features are more discriminative and only a global

average pooling and single fully connected layer is required for classification

QUESTION

# What is the function of the global average pooling layer (select from the options below)

|   | ○ To pool a feature map of any size down to a 1-dimensional vector                                                                                                                                                      |                |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | ○ To replace convolution layers.                                                                                                                                                                                        | ×              |
|   | To replace fully connected layers                                                                                                                                                                                       | ✓              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                               |                |
| ~ | Feedback for answer: 1                                                                                                                                                                                                  | 100% of points |
|   | Correct! Global average pooling can reduce any size feature ma<br>fixed size.                                                                                                                                           | ip to a        |
| × | Feedback for answer: 2                                                                                                                                                                                                  | 0% of points   |
|   | Incorrect. It replaces fully connected layers, not convolution lay                                                                                                                                                      | ers.           |
| ~ | Feedback for answer: 3                                                                                                                                                                                                  | 100% of points |
|   | Correct. However, the representational power (or capacity) of the (convolutional part) of the network must increase to compensa the loss of parameters when a fully connected layer is replaced global average pooling. | te for         |
|   |                                                                                                                                                                                                                         |                |

QUESTION

Why did the authors of Inception need to find a dense equivalent to a sparse network? (select the correct response from the options below)

|   | Modern GPUs are designed for dense computation.                                                                                       |                |
|---|---------------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | Dense networks have fewer parameters.                                                                                                 | X              |
|   | ○ Sparse networks are faster.                                                                                                         | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                             |                |
| ~ | Feedback for answer: 1                                                                                                                | 100% of points |
|   | Correct! Modern GPUs are designed for dense matrix multiplical operations and cannot take computational advantage of sparse matrices. |                |
| × | Feedback for answer: 2                                                                                                                | 0% of points   |
|   | Incorrect. Dense networks will have more parameters.                                                                                  |                |
| × | Feedback for answer: 3                                                                                                                | 0% of points   |
|   | Incorrect, modern GPUs are designed for dense computation an cannot take advantage of sparse matrix computation.                      | nd             |
|   |                                                                                                                                       |                |

QUESTION

# What is the purpose of the "Bottleneck" Convolution? (select from the options below)

| To reduce the spatial dimensionality of feature maps.                                                                                                                                             | ×              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| To increase the number of parameters.                                                                                                                                                             | ×              |
| To eliminate the need for pooling layers.                                                                                                                                                         | ×              |
| To reduce the amount of computation in a convolution operation.                                                                                                                                   | ✓              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                                         |                |
| × Feedback for answer: 1                                                                                                                                                                          | 0% of points   |
| Incorrect! The bottleneck convolution uses 1 x 1 convolution a therefore keeps the same spatial dimensionality from input to The number of channels is reduced, which helps reduce comp           | output.        |
| ⊠ Feedback for answer: 2                                                                                                                                                                          | 0% of points   |
| Incorrect! The overall number of parameters will be reduced of<br>the reduced no. of feature maps produced by the bottleneck l<br>(that are convolved by the next larger kernel size convolution) | ayer           |
| 区 Feedback for answer: 3                                                                                                                                                                          | 0% of points   |
| Incorrect Bottleneck convolutions do not reduce the spatial dimension, so pooling layers are not automatically eliminated                                                                         |                |
| Feedback for answer: 4                                                                                                                                                                            | 100% of points |

Correct! Bottleneck layers are introduce to reduce computation. The reduced number of feature maps produce by the 1 x 1 convolution which is then passed to the original convolution (with multiple kernels & larger kernels), helps reduce computation.

# <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107636

QUESTION

# What is the key problem encountered when training very deep networks?

| Overfitting                                                                                                                                                                       | ×             |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------|
| Vanishing gradient problem                                                                                                                                                        | <b>V</b>      |
| Lack of computational resources                                                                                                                                                   | ×             |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                         |               |
|                                                                                                                                                                                   | 0% of points  |
| Incorrect! Very deep networks will have a large number of param and are subject to overfitting, however this does not prevent trai                                                |               |
| Feedback for answer: 2                                                                                                                                                            | 00% of points |
| Correct! The vanishing gradient problem causes issues during traas the value of the gradient during backpropagation may be high attenuated preventing early layers from learning. |               |
|                                                                                                                                                                                   | 0% of points  |
| Incorrect. Although computational resources are important, the vanishing gradient problem is the primary issue.                                                                   |               |

QUESTION

# How do residual networks avoid the vanishing gradient problem and allow training of deep networks?

|          | By using batch normalisation layers                                                                                                                                                                                                   | ×              |
|----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          | By using short paths that carry large magnitude gradient values                                                                                                                                                                       | <b>V</b>       |
|          | By reducing the number of layers                                                                                                                                                                                                      | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                             |                |
| ×        | Feedback for answer: 1                                                                                                                                                                                                                | 0% of points   |
|          | Incorrect. Batch Normalisation layers can improve training time normalising the outputs of feature maps, however they don't protect the vanishing gradient problem.                                                                   |                |
|          |                                                                                                                                                                                                                                       |                |
| <b>~</b> | Feedback for answer: 2                                                                                                                                                                                                                | 100% of points |
|          | Correct! Residual networks may be visualised as an ensemble of of different lengths, the presence of short paths allows large magnitude gradient values to be carried during backpropagation avoiding the vanishing gradient problem. | paths          |
| ×        | Correct! Residual networks may be visualised as an ensemble of of different lengths, the presence of short paths allows large magnitude gradient values to be carried during backpropagation avoiding the vanishing gradient problem. | paths          |

actually allow for deeper networks by using short paths that can carry large magnitude gradient values for training.

QUESTION

# What is the principal advantage of using Batch Normalisation in a neural network?

|          | It reduces the number of parameters.                                                                                                                                                       | ×              |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          | ○ It allows a higher learning rate                                                                                                                                                         | $\checkmark$   |
|          | It eliminates the need for activation functions                                                                                                                                            | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                  |                |
| X        | Feedback for answer: 1                                                                                                                                                                     | 0% of points   |
|          | Incorrect. Batch Normalisation does not reduce the number of parameters. It helps to accelerate convergence of the network. introduces a small number of parameters itself.                | lt also        |
| <b>V</b> | Feedback for answer: 2                                                                                                                                                                     | 100% of points |
|          | Correct!                                                                                                                                                                                   |                |
| ×        | Feedback for answer: 3                                                                                                                                                                     | 0% of points   |
|          | Incorrect. Activation functions are still needed even with Batch Normalisation. Activation functions provide non-linearity to the network and are key to learning complex representations. | neural         |
|          |                                                                                                                                                                                            |                |

QUESTION

# What is the key difference between the DenseNet and ResNet architectures?

| DenseNet uses pooling layers, ResNet does not                                                                                                                                                                                                                                                                                                                                                                                                                                           | ×                               |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------|
| DenseNet concatenates feature maps, ResNet adds them                                                                                                                                                                                                                                                                                                                                                                                                                                    | ✓                               |
| DenseNet uses larger convolution filters                                                                                                                                                                                                                                                                                                                                                                                                                                                | ×                               |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                                                                                                                                                                                                                                                                               |                                 |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0% of points                    |
| Incorrect. Both networks use pooling layers. The DenseNet uses pooling between each stage and global average pooling at the enthe network. The resnet does not use a significant amount of population using a pooling layer early in the network to reduce the size feature map in the first non-residual stage. Feature dimensions reduced using stride =2 convolutions in residual blocks as an alternative to pooling. A global average pooling stage is used at end of Resnet also. | nd of<br>oling,<br>of an<br>are |
| ✓ Feedback for answer: 2                                                                                                                                                                                                                                                                                                                                                                                                                                                                | 100% of points                  |
| Correct! The concatenation approach used by DenseNet can gree increase the depth of feature maps (and computation), so the sy is broken in to blocks with no concatenation taking place between blocks (only within blocks).                                                                                                                                                                                                                                                            | /stem                           |
|                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 0% of points                    |

Incorrect. The difference lies in how they either add or concatenate feature maps, not the specific size of convolution filters (In general both networks use stacks of small  $3 \times 3$  and  $1 \times 1$  convolutions, which create large receptive fields)

# <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107641

**QUESTION** 

Select the statement that best explains the concept of transfer learning for image classification.

| Training a new network from scratch.                                                   | ×           |
|----------------------------------------------------------------------------------------|-------------|
| Using a pre-trained network's feature extractor with a new classifier.                 | <           |
| Using only fully connected layers for classification.                                  | ×           |
| Select one answer. Upon selection it will submit your answer immediately.              |             |
| ⊠ Feedback for answer: 1 00000000000000000000000000000000000                           | % of points |
| Incorrect. Transfer learning involves using a pre-trained network.                     |             |
| ☑ Feedback for answer: 2                                                               | % of points |
| Correct!                                                                               |             |
| ➤ Feedback for answer: 3                                                               | % of points |
| Incorrect. Transfer learning uses the feature extractor from a pre-<br>trained network |             |
|                                                                                        |             |

QUESTION

Transfer Learning is applied from a network pretrained on the ImageNet dataset to a target domain with a small dataset, which is recognizing different dog breeds. How well would you expect transfer learning to perform versus training a new network for the target domain from scratch?

| Transfer learning will perform worse  | × |
|---------------------------------------|---|
|                                       |   |
| Transfer learning will perform better | ✓ |
|                                       |   |
| Both will perform equally             | × |

Select one answer. Upon selection it will submit your answer immediately.

#### × Feedback for answer: 1

0% of points

Incorrect. Training a new classifier from scratch will perform less well than transfer learning as the target domain has a small dataset. It is expected transfer learning will perform better due to the similarity of the pre-training and target domains.

#### ✓ Feedback for answer: 2

100% of points

Correct! The source domain ImageNet is likely to contain images of dogs, which is the subject of the target domain. Transfer learning is likely to perform very well in this case due to the similarity of the domains.

× Feedback for answer: 3

0% of points

Incorrect. Training a new classifier from scratch will perform less well than transfer learning as the target domain has a small dataset. It is expected transfer learning will perform better due to the similarity of the pre-training and target domains.

QUESTION

# A target domain has a large amount of training data available, why would a transfer learning approach still be used?

| To reduce the number of parameters | × |
|------------------------------------|---|
|                                    |   |
| ○ To speed up training             | ✓ |
|                                    |   |
| ○ To improve accuracy              | × |

Select one answer. Upon selection it will submit your answer immediately.

### 

0% of points

Incorrect. Transfer learning does not necessarily reduce the number of parameters. The number of parameters is highly dependent on the pre-trained network chosen. A network trained from scratch may be more efficient in terms of the number of parameters, however it may take longer to train.

### ✓ Feedback for answer: 2

100% of points

Correct! We often choose to use transfer learning to speed up training on the target domain, using a pre-trained network with similar features will converge much more quickly than training from scratch.

## 

0% of points

Incorrect! Transfer learning and a network trained from scratch (with enough data) should be able to reach the same accuracy, unless the

number of parameters (and hence representational capacity) in the network trained from scratch is insufficient.

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107739

**QUESTION** 

Given the confusion below matrix, calculate the accuracy, precision and recall for each class. The True class labels are given along each row and the predicted labels are along the columns

$$\begin{pmatrix} A_T & 19 & 2 & 3 \ B_T & 3 & 17 & 7 \ C_T & 9 & 6 & 31 \ & A_P & B_P & C_P \end{pmatrix}$$

(Give your answer as a 3 x 3 matrix, with the accuracy, precision and recall for each class arranged along each row. e.g.  $\begin{pmatrix} Acc_A & Pre_A & Rec_A \\ Acc_B & Pre_B & Rec_B \\ Acc_C & Pre_C & Rec_C \end{pmatrix}$ , use two

decimal places of accuracy for each value)

Detailed solution

We can calculate the accuracy precision & recall for the  $i^{th}$  class from the equations:

$$Acc_i = \frac{TP_i + TN_i}{TP_i + TN_i + FP_i + FN_i}$$

$$Pre_i = \frac{TP_i}{TP_i + FP_i}$$

$$Rec_i = \frac{TP_i}{TP_i + FN_i}$$

where  $TP_i$  are the true positives,  $TN_i$  are the true negatives,  $FP_i$  are the false positives and  $FN_i$  are the false negatives.

We must first identify the true and false positives/negatives from the table for each class before performing the calculations.

The true positives for the class "A" occur when the predicted label exactly matches the true label:  $TP_A=19$ . The false positives occur when the predictions are incorrect and wrongly identify another class.  $FP_A=9+3=12$ . The False negatives occur when instances of the class are missed.  $FN_A=2+3=5$  and finally True negatives occur when instances of other classes are (correctly) not predicted as instances of the class "A".  $TN_A=17+7+6+31=61$ 

We can then calculate the Precision, accuracy and recall for class "A" using the TP,FP,FN & TN values.

$$Acc_A = \frac{TP_A + TN_A}{TP_A + TN_A + FP_A + FN_A} = \frac{19 + 61}{19 + 61 + 12 + 5} = 0.82$$

$$Pre_A = \frac{TP_A}{TP_A + FP_A} = \frac{19}{19 + 12} = 0.61$$

$$Rec_{A} = \frac{TP_{A}}{TP_{A} + FN_{A}} = \frac{19}{19 + 5} = 0.79$$

Similarly for class "B" we find that  $TP_B=17$ ,  $FP_B=2+6=8$ ,

$$FN_B = 3 + 7 = 10$$
 and  $TN_B = 19 + 3 + 9 + 31 = 62$ .

Calculating accuracy, Precision and Recall for class "B"

$$Acc_B = \frac{TP_B + TN_B}{TP_B + TN_B + FP_B + FN_B} = \frac{17 + 62}{17 + 62 + 8 + 10} = 0.81$$

$$Pre_B = \frac{TP_B}{TP_B + FP_B} = \frac{17}{17 + 8} = 0.68$$

$$Rec_{B} = \frac{TP_{B}}{TP_{B} + FN_{B}} = \frac{17}{17 + 10} = 0.63$$

Finally for class "C" we find  $TP_C=31$  ,  $FP_C=10$  ,  $FN_C=15$  and  $TN_c=41$  .

Calculating the accuracy, precision and recall we get:

$$Acc_C = \frac{TP_C + TN_C}{TP_C + TN_C + FP_C + FN_C} = \frac{31 + 41}{31 + 41 + 10 + 15} = 0.74$$

$$Pre_C = \frac{TP_C}{TP_C + FP_C} = \frac{31}{31 + 10} = 0.76$$

$$Rec_C = \frac{TP_C}{TP_C + FN_C} = \frac{31}{31 + 15} = 0.67$$

### Question at wrong answer

We must first identify the true and false positives/negatives from the table for each class before performing the calculations.

The true positives for the class "A" occur when the predicted label exactly matches the true label:  $TP_A=19$ . The false positives occur when the predictions are incorrect and wrongly identify another class.  $FP_A=9+3=12$ . The False negatives occur when instances of the class are missed.  $FN_A=2+3=5$  and finally True negatives occur when instances of other classes are (correctly) not predicted as instances of the class "A".  $TN_A=17+7+6+31=61$ 

We can then calculate the Precision, accuracy and recall for class "A" using the TP,FP,FN & TN values.

$$Acc_A = \frac{TP_A + TN_A}{TP_A + TN_A + FP_A + FN_A} = \frac{19 + 61}{19 + 61 + 12 + 5} = 0.82$$

$$Pre_A = \frac{TP_A}{TP_A + FP_A} = \frac{19}{19 + 12} = 0.61$$

$$Rec_{A} = \frac{TP_{A}}{TP_{A} + FN_{A}} = \frac{19}{19 + 5} = 0.79$$

We can then similarly find the TP,FP,TN,FN for the other two classes and calculate the Accuracy, Precision and Recall.

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107796

QUESTION

In the AlexNet Network the 1st convolution layer operates on the input image of size  $i \times i = 224 \times 224$  with C=3 channels. The output of the convolution layer is size  $o \times o = 55 \times 55$  with N=96 feature maps and a kernel with a spatial size of  $k \times k = 11 \times 11$  is used for the convolution. Determine the number of multiplications used in this convolution layer.

#### ✓ Detailed solution

We can first calculate the number of kernel parameters as the spatial size of the kernel  $k \times k$  multiplied by the input of input channels C multiplied by the number of output feature maps N which is the same as the number of kernels in the layer.

$$N_p = k \times k \times C \times N = 11 \times 11 \times 3 \times 96 = 34848$$

We note that there are bias parameters added to each of the N=96 feature maps, however we don't included them in our calculation as they are not multiplied by the inputs (Multiplications are more

computationally expensive than additions, that is why we are interested in determining the total number of multiplication operations).

The number of multiplications is determined by the number of parameters in the kernels times the number of possible spatial positions the kernels visit during convolution.

The number of possible spatial positions is a function of the input image size, kernel size and stride, however we don't need to perform a calculation to determine this. Instead we can use the fact that the number of possible spatial positions is equal to the spatial size of the output feature maps  $o \times o = 55 \times 55$ .

The number of multiplications is therefore

$$N_m = k \times k \times C \times N \times o \times o = 11 \times 11 \times 3 \times 96 \times 55 \times 55 = 105415200$$

# ■ Question at wrong answer

We can first calculate the number of kernel parameters as the spatial size of the kernel  $k \times k$  multiplied by the input of input channels C multiplied by the number of output feature maps N which is the same as the number of kernels in the layer.

$$N_p = k \times k \times C \times N$$

The number of multiplications will be the number of parameters multiplied by the number of possible spatial positions of the kernel. This quantity is equal to the spatial size of the output feature maps  $o \times o = 55 \times 55$ .

☑ Feedback when answer is equal to 105415200

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107797

QUESTION

In the AlexNet Network the 1st convolution layer operates on the input image of size  $i \times i = 224 \times 224$  with C=3 channels. The output of the convolution layer is size  $o \times o = 55 \times 55$  with N=96 feature maps and a kernel with a spatial size of  $k \times k = 11 \times 11$  is used for the convolution. Determine the number of multiplications used in this convolution layer.

#### ✓ Detailed solution

We can first calculate the number of kernel parameters as the spatial size of the kernel  $k \times k$  multiplied by the input of input channels C multiplied by the number of output feature maps N which is the same as the number of kernels in the layer.

$$N_p = k \times k \times C \times N = 11 \times 11 \times 3 \times 96 = 34848$$

We note that there are bias parameters added to each of the N=96 feature maps, however we don't included them in our calculation as they are not multiplied by the inputs (Multiplications are more

computationally expensive than additions, that is why we are interested in determining the total number of multiplication operations).

The number of multiplications is determined by the number of parameters in the kernels times the number of possible spatial positions the kernels visit during convolution.

The number of possible spatial positions is a function of the input image size, kernel size and stride, however we don't need to perform a calculation to determine this. Instead we can use the fact that the number of possible spatial positions is equal to the spatial size of the output feature maps  $o \times o = 55 \times 55$ .

The number of multiplications is therefore

$$N_m = k \times k \times C \times N \times o \times o = 11 \times 11 \times 3 \times 96 \times 55 \times 55 = 105415200$$

# ■ Question at wrong answer

We can first calculate the number of kernel parameters as the spatial size of the kernel  $k \times k$  multiplied by the input of input channels C multiplied by the number of output feature maps N which is the same as the number of kernels in the layer.

$$N_p = k \times k \times C \times N$$

The number of multiplications will be the number of parameters multiplied by the number of possible spatial positions of the kernel. This quantity is equal to the spatial size of the output feature maps  $o \times o = 55 \times 55$ .

☑ Feedback when answer is equal to 105415200

100% of points

No specific feedback

# Exercise 111904

**QUESTION** 

Apply batch Normalisation to the given mini batch x=[4.2,1.4,3.3,3.2,0.4]. The shift  $\gamma=0.6$  and scale  $\beta=0.3$  parameters have the values shown.

Determine the output of the batch norm stage. (Give your answer a 1 x 5 matrix containing the  $y_i$  values to 1 decimal place of accuracy)

# ✓ Detailed solution

Step 1. We first determine the mean and variance of the current batch.

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i = \frac{4.2 + 1.4 + 3.3 + 3.2 + 0.4}{5} = 2.50$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m \left( x_i - \mu_B \right)^2 = \frac{\left( 4.2 - 2.50 \right)^2 + \left( 1.4 - 2.50 \right)^2 + \left( 3.3 - 2.50 \right)^2 + \left( 3.2 - 2.50 \right)^2 + \left( 0.4 - 2.50 \right)^2}{5}$$

$$\sigma_R^2=1.93$$

Step 2. Using the updated mean and variance values we can now calculate the normalised value for the input to the batch normalisation layer using the formulae. (note the parameter  $\epsilon$  is a very small value to prevent divide by zero, so it can be neglected for non-zero  $\sigma^2$ )

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

For example we calculate  $\hat{x}_0=\frac{4.2-2.50}{\sqrt{1.93}}=1.22$ , the remaining values are calculated as  $\hat{x}_1=-0.79$ ,  $\hat{x}_2=0.58$ ,  $\hat{x}_3=0.50$  and  $\hat{x}_4=-1.51$ .

Step 3.

We now apply the shift  $\gamma$  and scale  $\beta$  parameters to the normalised outputs using the equation  $y_i = \gamma \hat{x}_i + \beta$ .

For example we calculate  $y_0=\gamma\hat{x}_0+\beta=0.6\times1.22+0.3=1.0$  , similarly we calculate  $y_1=-0.2$  ,  $y_2=0.6$  ,  $y_3=0.6$  and  $y_4=-0.6$ 

### □ Question at wrong answer

Step 1. We first determine the mean and variance of the current batch.

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i = \frac{4.2 + 1.4 + 3.3 + 3.2 + 0.4}{5} = 2.50$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m \left( x_i - \mu_B \right)^2 = \frac{(4.2 - 2.50)^2 + (1.4 - 2.50)^2 + (3.3 - 2.50)^2 + (3.2 - 2.50)^2 + (0.4 - 2.50)^2}{\pi}$$

$$\sigma_B^2 = 1.93$$

Step 2. Using the updated mean and variance values we can now calculate the normalised value for the input to the batch normalisation layer using the formulae. (note the parameter  $\epsilon$  is a very small value to prevent divide by zero, so it can be neglected for non-zero  $\sigma^2$ )

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Step 3.

We now apply the shift  $\gamma$  and scale  $\beta$  parameters to the normalised outputs using the equation  $y_i=\gamma\hat{x}_i+\beta$ .

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 1.0 & -0.2 & 0.6 & 0.6 & -0.6 \end{pmatrix}$ 

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 107913

**QUESTION** 

During training of batch normalisation, a global mean  $\mu$  and variance  $\sigma^2$  are also calculated. (This global mean and variance are later used (part 2 of question) when batch normalisation is operating in inference mode). Given a batch of 1D input values to a layer x=[2.7,2.1,2.6,1.9,3.8], and assuming the previous global mean and variance for the layer are  $\mu_-=3.61$  and  $\sigma_-^2=0.9$ , calculate the updated global mean  $\mu=\mu_-\times d+\mu_B\times (1-d)$  and global variance  $\sigma^2=\sigma_-^2\times d+\sigma_B^2\times (1-d)$ . Where d=0.95 is a decay parameter and  $\mu_B$  and  $\sigma_B^2$  are the mean and variance of the current input batch. (Give your answer as a 1 x 2 matrix below)

Detailed solution

Step 1. We first determine the mean and variance of the current batch.

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i = \frac{2.7 + 2.1 + 2.6 + 1.9 + 3.8}{5} = 2.62$$

$$\begin{split} \sigma_B^2 &= \frac{1}{m} \sum_{i=1}^m \left( x_i - \mu_B \right)^2 = \ &\frac{(2.7 - 2.62)^2 + (2.1 - 2.62)^2 + (2.6 - 2.62)^2 + (1.9 - 2.62)^2 + (3.8 - 2.62)^2}{5} \end{split}$$

$$\sigma_B^2=0.44$$

Step 2. We update the global (or long term) mean  $\mu=3.61$  and variance  $\sigma^2=0.9$  using the decay rate d=0.95, using the batch mean  $\mu_B$  and variance  $\sigma_B^2$  we have just calculated.

$$\mu = \mu_- \times d + \mu_B \times (1-d) = 3.61 \times 0.95 + 2.62 \times (1-0.95) = 3.56$$

$$\sigma^2 = \sigma_-^2 \times d + \sigma_B^2 \times (1-d) = 0.9 \times 0.95 + 0.44 \times (1-0.95) = 0.88$$

### Question at wrong answer

Step 1. We first determine the mean and variance of the current batch.

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i = \frac{2.7 + 2.1 + 2.6 + 1.9 + 3.8}{5} = 2.62$$

$$\begin{split} \sigma_B^2 &= \frac{1}{m} \sum_{i=1}^m \left( x_i - \mu_B \right)^2 = \ &\frac{\left( 2.7 - 2.62 \right)^2 + \left( 2.1 - 2.62 \right)^2 + \left( 2.6 - 2.62 \right)^2 + \left( 1.9 - 2.62 \right)^2 + \left( 3.8 - 2.62 \right)^2}{5} \end{split}$$

$$\sigma_B^2 = 0.44$$

Step 2. We update the global (or long term) mean  $\mu=3.61$  and variance  $\sigma^2=0.9$  using the decay rate d=[!d!], using the batch mean  $\mu_B$  and variance  $\sigma_B^2$  we have just calculated.

$$\mu = \mu_- \times d + \mu_B \times (1-d)$$

$$\sigma^2 = \sigma_-^2 \times d + \sigma_B^2 \times (1-d)$$

 $\boxed{\text{ Feedback when answer is algebraically Equivalent to } (3.56 0.88)}$ 

100% of points

No specific feedback

QUESTION

Apply batch normalisation (including scale and shift) to a new input x=0.6 to the same layer assuming that operation is now in inference mode. As the input is a single value (not a batch) we must now use the updated global mean & variance in the normalisation calculation. The shift  $\gamma=0.9$  and scale  $\beta=0.5$  parameters have the values shown. (Give your answer containing the new normalised output values to 1 decimal place of accuracy)

### ✓ Detailed solution

Step 1. Using the updated global mean  $\mu=3.56$  and variance values  $\sigma^2=0.88$  from part 1 of the question we can now calculate the normalised value for the input to the batch normalisation layer using the formulae. (note the parameter  $\epsilon$  is a very small value to prevent divide by zero, so it can be neglected for non-zero  $\sigma^2$ )

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

We calculate 
$$\,\hat{x}_0=\frac{2.7-3.56}{\sqrt{0.9}}=-3.16$$

Step 2.

We now apply the shift  $\gamma$  and scale  $\beta$  parameters to the normalised outputs using the equation  $y_i = \gamma \hat{x}_i + \beta$ .

Therefore  $y_0 = \gamma \hat{x}_0 + \beta = 0.9 \times -3.16 + 0.5 = -2.3$ 

□ Question at wrong answer

Step 1. Using the updated mean and variance values we can now calculate the normalised value for the input to the batch normalisation layer using the formulae. (note the parameter  $\epsilon$  is a very small value to prevent divide by zero, so it can be neglected for non-zero  $\sigma^2$ )

$$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

Step 2.

We now apply the shift  $\gamma$  and scale  $\beta$  parameters to the normalised outputs using the equation  $y_i = \gamma \hat{x}_i + \beta$ .

 $\rightharpoonup$  Feedback when answer is algebraically Equivalent to -2.3

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 112291

**QUESTION** 

Calculate the cross entropy loss (for a multi-class classifier with N=5 outputs) given the output probabilities  $p_c=[0.20,0.23,0.20,0.20,0.17]$  and the ground truth class label  $\ Y_c=2$  (where the classes are in the range 0 to 4)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| 0.14                                                   |
| Check my answer                                        |

### ✓ Detailed solution

Multi class cross entropy loss is given by

$$\mathcal{L} = -\frac{1}{N}\sum_{c=1}^{N}y_{c}\log\left(p_{c}\right)$$

where  $y_c=[0.0,1,0,0]$  are class labels (one hot encoded) and  $p_c=[0.20,0.23,0.20,0.20,0.17]$  are the probabilities given by the output layer with softmax activation.

We therefore can calculate the loss by substituting for the  $y_c$  and  $p_c$  values in the equation for  $\mathcal{L}$ .

$$\mathcal{L} = -\frac{1}{N} \left( \left( 0 \log \left( 0.20 \right) + 0 \log \left( 0.23 \right) + 1 \log \left( 0.20 \right) + 0 \log \left( 0.20 \right) + 0 \log \cdot \left( \left( 0.17 \right) \right) \right)$$

We note that all but one term will be reduced to zero. For the term with  $y_c=1$ , the closer the value of  $p_c$  to 1, the smaller the value of  $\log{(p_c)}$  and loss will be close to 0, (which is the desired state after training).

$$\mathcal{L} = -\frac{1}{5} \left( 0.00 + 0.00 - 0.70 + 0.00 + 0.00 \right) = 0.14$$

### Question at wrong answer

Multi class cross entropy loss is given by

$$\mathcal{L} = -\frac{1}{N}\sum_{c=1}^{N}y_{c}\log\left(p_{c}\right)$$

where  $y_c=[0.0,1,0,0]$  are class labels (one hot encoded) and  $p_c=[0.20,0.23,0.20,0.20,0.17]$  are the probabilities given by the output layer with softmax activation.

We therefore can calculate the loss by substituting for the  $y_c$  and  $p_c$  values in the equation for  $\mathcal{L}$ .

✓ Feedback when answer is equal to 0.14

100% of points

No specific feedback