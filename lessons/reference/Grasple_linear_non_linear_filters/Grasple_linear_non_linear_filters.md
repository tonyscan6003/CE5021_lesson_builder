<u>View subject</u> | <u>Edit subject</u>

# Linear/Non-Linear Filters

Slides

# **Exercises**

### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 105757

QUESTION

# Demonstrate Non-linear Filters

We want to calculate the result of applying a median filter with a  $3 \times 3$  size to the image shown. Assume no zero padding.

$$I = \begin{pmatrix} 2 & 3 & 5 & 3 \ 4 & 5 & 6 & 2 \ 7 & 2 & 4 & 8 \ 4 & 1 & 3 & 1 \end{pmatrix}$$

Give your answer as a 2 x 2 matrix in the box below.

#### ☑ Detailed solution

We slide the filter window around the image (similar to 2d convolution) As the image is of size  $4 \times 4$  and the image is of size  $3 \times 3$ , there will be 4 possible positions and the output will be of size  $2 \times 2$ .

The first 3 x 3 region (top left) contains the values 2 3 5 4 5 6 7 2 4, ordering these from lowest to highest we get 2 2 3 4 4 5 5 6 7. As we have 9 values the median with be the 5th value, this is 4 in this case.

The second 3 x 3 region (top right) contains the values 3 5 3 5 6 2 2 4 8, ordering these from lowest to highest we get 2 2 3 3 4 5 6 8. As we have 9 values the median with be the 5th value, this is 4 in this case.

The third 3 x 3 region (bottom left) contains the values 45672441 3, ordering these from lowest to highest we get 123444567. As we have 9 values the median with be the 5th value, this is 4 again in this case.

The fourth 3 x 3 region (bottom right) contains the values 5 6 2 2 4 8 1 3 1 ordering these from lowest to highest we get 1 1 2 2 3 4 5 6 8. As we have 9 values the median with be the 5th value, this is 3 in this case.

#### □ Question at wrong answer

The median filter operates similar to 2d convolution, the filter window is slid around the image visiting each position. A median operation is carried out in each position and the value retained.

In the question at the first 3 x 3 filter window position (top left), we have nine values : 2 3 5 4 5 6 7 2 4

We can order the values from lowest to highest: 2 2 3 4 4 5 5 6 7.

As we have 9 values the median with be the 5th value, this is 4 in this case.

Follow the same procedure to get the median of the other 3 regions of the image.

 $\bigspace$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 4 & 4 \\ 4 & 3 \end{pmatrix}$ 

100% of points

No specific feedback

#### <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 105831

QUESTION

A more computational Efficient Convolution

Perform Separable Convolution between the Image

matrix 
$$I = \begin{pmatrix} 4 & 1 & 3 \\ 2 & 0 & 3 \\ 0 & 1 & 1 \end{pmatrix}$$
 and the kernel

matrix 
$$I=\begin{pmatrix}4&1&3\\2&0&3\\0&1&1\end{pmatrix}$$
 and the kernel 
$$K=\begin{pmatrix}-1&-2&-1\\0&0&0\\1&2&1\end{pmatrix}.$$
 We want to the output image  $O$ 

size to be the same as the input image size (use one ring of zero padding around I).

Give your answer as a 3 x 3 matrix in the box below.

# Check my answer

Detailed solution

Step 1: In order to perform separable convolution we must factorise the kernel.

As K is a 3 x 3 matrix, the factors must be a 3 x 1 matrix multiplied by a 1 x 3 matrix. (In this case we determine the factors by inspection.) We also note that we can write this multiplication as a convolution. (you can verify that this is true yourself by performing the convolution)

$$K = \begin{pmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ 1 & 2 & 1 \end{pmatrix} = \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \times \begin{pmatrix} 1 & 2 & 1 \end{pmatrix} = \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \times \begin{pmatrix} 1 & 2 & 1 \end{pmatrix}$$

The original convolution can now be written as a separable convolution with two 1D kernels.

$$I \,\circledast K = I \circledast \begin{pmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ 1 & 2 & 1 \end{pmatrix} = I \circledast \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \circledast \begin{pmatrix} 1 & 2 & 1 \end{pmatrix}$$

We therefore can see that we perform convolution sequentially with each of the two factors. (As convolution is commutative we can do this in any order)

As detailed in the question we want to have the output image O size, the same as the input image I size. To achieve this we add a ring of zero padding before convolution.

$$I = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 \ 0 & 4 & 1 & 3 & 0 \ 0 & 2 & 0 & 3 & 0 \ 0 & 0 & 1 & 1 & 0 \ 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

Step 2: We convolve the image I with one of our 1D factors for K (order doesn't matter) to create an intermediate image  $I_{int}$ . Convolution is carried out in exactly the same manner as a 2D convolution (just the kernel shape is different). Importantly as this is a convolution operation we must reverse the kernel before performing

the multiplication and sum at each position in the image.

$$I_{int} = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 \ 0 & 4 & 1 & 3 & 0 \ 0 & 2 & 0 & 3 & 0 \ 0 & 0 & 1 & 1 & 0 \ 0 & 0 & 0 & 0 & 0 \end{pmatrix} \circledast \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} = \begin{pmatrix} 0 & -2 & 0 & -3 & 0 \ 0 & 4 & 0 & 2 & 0 \ 0 & 2 & 0 & 3 & 0 \end{pmatrix}$$

Step 3: We then convolve this intermediate Image  $I_{int}$  with the second factor to produce the output O. As we can see in this case the 1D kernel is symmetric, so flipping the kernel produces the same result.

$$O = \begin{pmatrix} 0 & -2 & 0 & -3 & 0 \\ 0 & 4 & 0 & 2 & 0 \\ 0 & 2 & 0 & 3 & 0 \end{pmatrix} \circledast \begin{pmatrix} 1 & 2 & 1 \end{pmatrix}$$

$$O = \begin{pmatrix} -4 & -5 & -6 \ 8 & 6 & 4 \ 4 & 5 & 6 \end{pmatrix}$$

#### ■ Question at wrong answer

Step 1: In order to perform separable convolution we must factorise the kernel. As

K is a 3 x 3 matrix, the factors must be a 3 x 1 matrix multiplied by a 1 x 3 matrix. (In this case we determine the factors by inspection.) We also note that we can write this multiplication as a convolution. (you can verify that this is true yourself by performing the convolution)

$$K = \begin{pmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ 1 & 2 & 1 \end{pmatrix} = \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \times \begin{pmatrix} 1 & 2 & 1 \end{pmatrix} = \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \times \begin{pmatrix} 1 & 2 & 1 \end{pmatrix}$$

The original convolution can now be written as a separable convolution with two 1D kernels.

$$I \,\circledast K = I \circledast \begin{pmatrix} -1 & -2 & -1 \ 0 & 0 & 0 \ 1 & 2 & 1 \end{pmatrix} = I \circledast \begin{pmatrix} -1 \ 0 \ 1 \end{pmatrix} \circledast \begin{pmatrix} 1 & 2 & 1 \end{pmatrix}$$

We therefore can see that we perform convolution sequentially with each of the two factors. (As convolution is commutative we can do this in any order)

As detailed in the question we want to have the output image O size, the same as the input image I size. To achieve this we add a ring of zero padding before convolution.

$$I = \begin{pmatrix} 0 & 0 & 0 & 0 & 0 \ 0 & 4 & 1 & 3 & 0 \ 0 & 2 & 0 & 3 & 0 \ 0 & 0 & 1 & 1 & 0 \ 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

Step 2: We convolve the image I with one of our 1D factors for K (order doesn't matter) to create an intermediate image  $I_{int}$ .

Convolution is carried out in exactly the same manner as a 2D convolution (just the kernel shape is different). Importantly as this is a convolution operation we must reverse the kernel before performing the multiplication and sum at each position in the image.

Step 3: We then perform convolution with the second factor on the result of step 2 to determine the final convolution result.

 $\bigver$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} -4 & -5 & -6 \\ 8 & 6 & 4 \\ 4 & 5 & 6 \end{pmatrix}$ 

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106018

**QUESTION** 

What is the superposition principle as it applies to image processing? (Select the correct response from the options below)

| _      |                                                                                                                                           |   |  |  |  |
|--------|-------------------------------------------------------------------------------------------------------------------------------------------|---|--|--|--|
|        | Convolving the sum of two images with a kernel is the same as convolving the images with the kernel individually and summing the results. | ✓ |  |  |  |
|        |                                                                                                                                           |   |  |  |  |
|        | A kernel is separable if it can be formed from the product of two 1D kernels                                                              | × |  |  |  |
|        |                                                                                                                                           |   |  |  |  |
|        | The sum of an image convolved with two kernels individually is the same as convolution of the image with the sum of the kernels           | ✓ |  |  |  |
|        |                                                                                                                                           |   |  |  |  |
|        | The input is shifted spatially then the output will be shifted by the same amount.                                                        | × |  |  |  |
| Select | Select one answer. Upon selection it will submit your answer immediately.                                                                 |   |  |  |  |

✓ Feedback for answer: 1 100% of points
This is the Correct definition for superposition!
✓ Feedback for answer: 2 0% of points
This defines a separable kernel, not the superposition principle.

✓ Feedback for answer: 3

100% of points

This is a correct statement but is not the primary definition of the superposition principle.

0% of points

This defines shift invariance, not the superposition principle.

QUESTION

# What do we mean if a linear filter displays Shift Invariance?

| The filter output is the same regardless of the input.                                                                          | × |  |  |  |  |
|---------------------------------------------------------------------------------------------------------------------------------|---|--|--|--|--|
| When the input is shifted spatially, the output will be shifted by the same amount.                                             | ✓ |  |  |  |  |
| The filter can be represented as the product of two 1D filters.                                                                 | × |  |  |  |  |
| The order of convolution operations does not change the result.                                                                 | × |  |  |  |  |
| Select one answer. Upon selection it will submit your answer immediately.                                                       |   |  |  |  |  |
| Feedback for answer: 1                                                                                                          |   |  |  |  |  |
| This is incorrect as shift invariance specifically refers to spatial shifts, not the output being the same regardless of input. |   |  |  |  |  |
| Feedback for answer: 2                                                                                                          |   |  |  |  |  |
| This is the correct definition of shift invariance!                                                                             |   |  |  |  |  |

This defines a separable filter, not shift invariance.

This defines the commutative property of convolution, not shift invariance.

QUESTION

# What is the commutative property of convolution?

|                          | 0                        | The order of a convolution operation does not change the result.                                                                          | $\overline{\mathbf{A}}$ |
|--------------------------|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|-------------------------|
|                          | 0                        | Convolving the sum of two images with a kernel is the same as convolving the images with the kernel individually and summing the results. | ×                       |
|                          | 0                        | When more than two images or kernels are convolved, it does not matter how the operations are grouped.                                    | ×                       |
|                          | Select                   | one answer. Upon selection it will submit your answer immediately.                                                                        |                         |
| ✓ Feedback for answer: 1 |                          |                                                                                                                                           | 100% of points          |
|                          |                          | is the correct definition for the commutative property of olution.                                                                        |                         |
|                          |                          | 0% of points                                                                                                                              |                         |
|                          |                          | defines the superposition principle, not the commutative erty.                                                                            |                         |
| ×                        | × Feedback for answer: 3 |                                                                                                                                           | 0% of points            |
|                          |                          | defines the associative property of convolution, not the mutative property.                                                               |                         |
|                          |                          |                                                                                                                                           |                         |

QUESTION

# What is the associative property of convolution?

|          |        | The order of a convolution operation does not change the result                                                                           | X              |
|----------|--------|-------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          |        | Convolving the sum of two images with a kernel is the same as convolving the images with the kernel individually and summing the results. | X              |
|          | 0      | When more than two images or kernels are convolved, it does not matter how the operations are grouped                                     | ✓              |
|          | Select | one answer. Upon selection it will submit your answer immediately.                                                                        |                |
| ×        | Feedba | ck for answer: 1                                                                                                                          | 0% of points   |
|          |        | defines the commutative property of convolution, not the ciative property                                                                 |                |
|          |        |                                                                                                                                           | 0% of points   |
|          | This   | defines the superposition principle, not the associative pro                                                                              | perty.         |
| <b>~</b> | Feedba | ck for answer: 3                                                                                                                          | 100% of points |
|          | No s   | pecific feedback                                                                                                                          |                |
|          |        |                                                                                                                                           |                |