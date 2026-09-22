<u>View subject</u> | <u>Edit subject</u>

# **Neural Networks**

Slides

## **Exercises**

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 107269

**QUESTION** 

For the simple neuron as shown in the image, with inputs  $x_i=[2.6,-0.7,-1.6]$ , weights  $w_i=[0.3,0.8,0.9]$  and bias b=1.9 calculate the output of the neuron, assuming a linear activation function  $\sigma\left(z\right)=z$ , where z is the output of the sum.

| $\square$ $\times$ | $\sqrt{\Box}$ | $\pi$         | $\left[ e^{\Box} \right]$ | $\pm$         |
|--------------------|---------------|---------------|---------------------------|---------------|
|                    | $\overline{}$ | $\overline{}$ |                           | $\overline{}$ |

0.680

## Check my answer

#### ✓ Detailed solution

The output of a neuron is the sum of the inputs multiplied by the weights plus the bias, the sum has then the activation function  $\sigma$  is applied. The equation for this process is  $y=\sigma\left(z\right)=\sigma\left(\sum_{i=1}^n x_iw_i+b\right)$ 

We first multiply the weights by the inputs to calculate the summation

$$\sum_{i=1}^n x_i w_i = 2.6 \cdot 0.3 - 0.7 \cdot 0.8 - 1.6 \cdot 0.9 = -1.220$$

We then add the bias

$$z = \sum_{i=1}^{n} x_i w_i + b = -1.220 + 1.9 = 0.680$$

As the activation function is linear

$$y = \sigma(z) = z = 0.680$$

## 

The output of a neuron is the sum of the inputs multiplied by the weights plus the bias, the sum has then the activation function applied. The equation for this process is  $y=\sigma\left(z\right)=\sigma\left(\sum_{i=1}^n x_iw_i+b\right)$ 

As the activation function is linear

$$y = \sigma(z) = z$$

 $\boxed{\boldsymbol{\varphi}}$  Feedback when answer is equal to 0.680

100% of points

QUESTION

If the activation is changed to

If the activation is changed to a sigmoid activation unit (with the weights, input and bias unchanged) what will the output value  $y_i$  be in this case?

0.66

## Check my answer

✓ Detailed solution

The sigmoid activation function is given by the equation

$$\sigma\left(z\right) = \frac{1}{1 + e^{-z}}.$$

Since we have already calculated the input to the activation  $z_i=0.680$  ,  $\frac{1}{2}$ 

we can calculate the output as  $y=\sigma\left(z\right)=\frac{1}{1+e^{-z}}=0.66$ 

Question at wrong answer

The sigmoid activation function is given by the equation  $\sigma\left(z\right)=\frac{1}{1+e^{-z}}\text{, we have already calculated the quantity }z=0.680\text{ that we can substitute into the equation.}$ 

✓ Feedback when answer is equal to 0.66

100% of points

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 107287

QUESTION

Calculate the output probabilities for the softmax activation function given inputs  $z_i=[1.3,1.5,3.9,2.9,1.1]$ . (Give your answer as a 1 x 5 matrix with 2 decimal places of accuracy)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                                                                                                                                                                                                                              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                                                                                                                                                                                                                                     |
| $ \boxed{ \alpha\beta\infty  \left[ \begin{array}{c} \sin\cos \end{array} \right] \left[ \begin{array}{c} \int \sum \prod \left[ \begin{array}{c} \binom{n}{k} \ln \end{array} \right] \left[ \begin{array}{c} -1 \\ -1 \end{array} \right] \left[ \begin{array}{c} -1 \\ -1 \end{array} \right] \left[ \begin{array}{c} -1 \\ -1 \end{array} \right] } \right] } $ |
| Check my answer                                                                                                                                                                                                                                                                                                                                                     |

#### Detailed solution

The softmax activation function will produce a normalised probability value for each input  $z_i$  according to the equation  $\sigma\left(z_i\right) = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}$ .

We calculate the exponent terms  $e^{z_i}$  for each of the  $z_i$  inputs:

$$\begin{pmatrix} e^{z_1} & 3.67 \ e^{z_2} & 4.48 \ e^{z_3} & 49.40 \ e^{z_4} & 18.17 \ e^{z_5} & 3.00 \ \end{pmatrix}$$

The sum of the exponentials is calculated as:

$$\sum_{j=1}^k e^{z_j} = e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4} + e^{z_5} = 3.67 + 4.48 + 49.40 + 18.17 + 3.00 = 78.72$$

The activation for each input is then calculated:

$$\begin{pmatrix} \frac{e^{z_1}}{\sum_{j=1}^k ez_j} = & \frac{3.67}{78.72} = & 0.05 \\ \frac{e^{z_2}}{\sum_{j=1}^k ez_j} = & \frac{4.48}{78.72} = & 0.06 \\ \frac{e^{z_3}}{\sum_{j=1}^k ez_j} = & \frac{49.40}{78.72} = & 0.63 \\ \frac{e^{z_4}}{\sum_{j=1}^k ez_j} = & \frac{18.17}{78.72} = & 0.23 \\ \frac{e^{z_5}}{\sum_{j=1}^k ez_j} = & \frac{3.00}{78.72} = & 0.04 \end{pmatrix}$$

## ■ Question at wrong answer

The softmax activation function will produce a normalised probability value for each input  $z_i$  according to the equation  $\sigma\left(z_i\right) = \frac{e^{z_i}}{\sum_{j=1}^k e^{z_j}}$ . We can calculate each  $e^{z_i}$  individually, determine the sum of the exponents  $\sum_{i=1}^k e^{z_i}$  and then calculate the  $\sigma\left(z_i\right)$  for each  $z_i$ 

☑ Feedback for compound rule based answer (index 1)

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 107288

**QUESTION** 

Apply max pooling to the matrix A below, assume that the pooling operator has a size of 2 pixels and a stride of 2. (We note that pooling does not use zero padding). Give your answer as a 2 x 2 matrix in the box below.

$$A = \begin{pmatrix} 7 & 4 & 8 & 2 \ 3 & 7 & 1 & 6 \ 7 & 8 & 3 & 3 \ 6 & 8 & 2 & 5 \end{pmatrix}$$

#### ✓ Detailed solution

We apply max operation at each position of the operator in the image, as the operator has a size of  $2 \times 2$  and a stride of 2, there will be 4 positions and the output is a  $2 \times 2$  matrix.

$$P = \begin{pmatrix} 7 & 8 \\ 8 & 5 \end{pmatrix}$$

## ■ Question at wrong answer

We apply maximum operation (amongst the pixels that lie under the operator) at each position of the operator in the image, as the operator has a size of 2 and a stride of 2, there will be 4 positions and the output is of size  $2 \times 2$ .

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 7 & 8 \\ 8 & 5 \end{pmatrix}$ 

100% of points

QUESTION

Apply average pooling to same matrix. (Give your answer as a  $2 \times 2$  matrix, with 2 decimal places of rounding)

### ✓ Detailed solution

We apply an average operation at each position of the operator in the image. As the operator has a size of  $2 \times 2$  and a stride of 2, there will be 4 positions and the output is a  $2 \times 2$  matrix.

$$P = \begin{pmatrix} 5.25 & 4.25 \\ 7.25 & 3.25 \end{pmatrix}$$

#### □ Question at wrong answer

We apply an average operation (amongst the pixels that lie under the operator) at each position of the operator in the image, as the operator has a size of  $2 \times 2$  and a stride of 2, there will be 4 positions and the output is of size  $2 \times 2$ .

 $\bigspace$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 5.25 & 4.25 \ 7.25 & 3.25 \end{pmatrix}$ 

100% of points