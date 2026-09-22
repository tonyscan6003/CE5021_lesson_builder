<u>View subject</u> | <u>Edit subject</u>

### Convolutional Neural Networks

Slides

#### **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 107366

**QUESTION** 

2D Convolution (Parameterised)

Find the result of convolving the 2x2 kernel with 3 channels  $K=\begin{pmatrix}1&-2\\0&-2\end{pmatrix},\begin{pmatrix}-2&0\\2&1\end{pmatrix},\begin{pmatrix}-1&0\\2&-2\end{pmatrix}$  with

the 3x3 image also with 3 channels

the 3x3 image also with 3 change 
$$B = \begin{pmatrix} 1 & 4 & 4 \\ 2 & 4 & 4 \\ 2 & 4 & 5 \end{pmatrix}, \begin{pmatrix} 2 & 2 & 2 \\ 4 & 5 & 2 \\ 2 & 4 & 0 \end{pmatrix}$$
,  $\begin{pmatrix} 3 & 2 & 4 \\ 0 & 4 & 3 \\ 0 & 5 & 1 \end{pmatrix}$ 

(Assuming no-padding this is also called "Valid Padding"). (Give your answer as a 2 x 2 matrix)

#### ✓ Detailed solution

In multi channel convolution we perform a convolution operation between each channel of the input and the corresponding channel of the kernel.

For example the 1st channel of the input and kernel:

$$B_1 \circledast K_1 = \begin{pmatrix} 1 & 4 & 4 \ 2 & 4 & 4 \ 2 & 4 & 5 \end{pmatrix} \circledast \begin{pmatrix} 1 & -2 \ 0 & -2 \end{pmatrix}$$

To perform convolution we multiply pixels of the image at each position by the kernel and sum the individual multiplications giving a 2 x 2 matrix. We must first remember to reverse (or flip)  $K_1' = \begin{pmatrix} -2 & 0 \\ -2 & 1 \end{pmatrix}$  the individual kernel matrices before performing multiplying and summation.

$$B_1 \circledast K_1 = \begin{pmatrix} -2 & -12 \ -4 & -11 \end{pmatrix}$$

We similarly perform convolution for the other channels and sum the matrices from the convolutions to get the final output feature map:

$$B\circledast K=\begin{pmatrix} -2 & -12 \ -4 & -11 \end{pmatrix}+\begin{pmatrix} -4 & 2 \ 6 & 9 \end{pmatrix}+\begin{pmatrix} -6 & 1 \ 3 & -3 \end{pmatrix}$$

$$B \circledast K = \begin{pmatrix} -12 & -9 \\ 5 & -5 \end{pmatrix}$$

#### Question at wrong answer

We perform the convolution graphically.

We must first remember to reverse (or flip) the individual channel matrices of the kernel before convolution.

$$K'=\begin{pmatrix} -2 & 0 \ -2 & 1 \end{pmatrix}, \begin{pmatrix} 1 & 2 \ 2 & -2 \end{pmatrix}, \begin{pmatrix} -2 & 2 \ 0 & -1 \end{pmatrix}$$

We step each channel of kernel around the corresponding channel of the input. As in regular 2D kernel we multiply the kernel by the pixels of the image at each position and sum the individual multiplications. We then finally add the results of these individual convolutions to produce a  $2 \times 2$  output feature map.

 $\boxed{\textstyle }$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} -12 & -9 \\ 5 & -5 \end{pmatrix}$ 

100% of points

#### Exercise 107415

**QUESTION** 

Calculate the receptive field of a neuron in the feature map at layer 3 in a simple 1 dimensional neural network.

The first layer (that operates on the input image) performs a convolution operation and has a kernel of size  $k_1=6$  with a stride of  $s_1=2$ . The second layer is also a convolution operation with a kernel of size  $k_2=5$  stride  $s_2=2$ . The third layer has a kernel of size  $k_3=4$ , with the stride in this case  $s_3=1$ .

#### ✓ Detailed solution

Calculating the receptive field is recursive operation. The receptive field of each layer  $r_{l-1}$  (relative to the deepest layer) is dependent on the receptive field of the next deeper layer  $r_l$ .

These quantities are related by the recurrence relation  $r_{l-1}=s_l\cdot r_l+(k_l-s_l)$  which includes the stride  $s_l$  and kernel size  $k_l$  of the current layer.

For example in the image below if we assume  $r_l=1$  and  $s_l=1$  and  $k_l=3$ . The we can easily see that due to the kernel of size 3 each neuron in the feature map of layer l connects to 3 neurons in the feature map of layer l-1. Using the equation  $r_{l-1}=s_l\cdot r_l+(k_l-s_l)=1\cdot 1+(3-1)=3$  we see that we get the same result.

For the specific question we were asked, we start with a single neuron in the deepest layer which is the 3rd layer in this question. The receptive field of the deepest layer (No. 3) relative to itself will be just

one  $r_3=1.$  We can determine the receptive field  $r_2$  of the 2nd layer (again relative to a neuron in deepest 3rd layer) as

$$r_2 = s_3 \cdot r_3 + (k_3 - s_3) = 1 \cdot 1 + (4 - 1) = 4$$

We then calculate the receptive field of the 1st layer  $r_1$  relative to the deepest layer (No. 3).

$$r_1 = s_2 \cdot r_2 + (k_2 - s_2) = 2 \cdot 4 + (5 - 2) = 11$$

We finally calculate the receptive field of the input image  $r_0$  relative to the deepest layer.

$$r_0 = s_1 \cdot r_1 + (k_1 - s_1) = 2 \cdot 11 + (6 - 2) = 26$$

#### □ Question at wrong answer

Calculating the receptive field is recursive operation. The receptive field of each layer  $r_{l-1}$  (relative to the deepest layer) is dependent on the receptive field of the deeper layer  $r_l$ .

These quantities are related by the recurrence relation  $r_{l-1}=s_l\cdot r_l+(k_l-s_l)$  which includes the stride  $s_l$  and kernel size  $k_l$  of the current layer.

result.

For example in the image below if we assume  $r_l=1$  and  $s_l=3$  and  $k_l=1$ . The we can easily see that due to the kernel of size 3 each neuron in the feature map of layer l connects to 3 neurons in the feature map of layer l-1. Using the equation  $r_{l-1}=s_l\cdot r_l+(k_l-s_l)=1\cdot 1+(3-1)=3$  we see that we get the same

Convolution

(K<sub>i</sub>, s<sub>i</sub>)

Output Feature map Layer I-1

Convolution

(K<sub>i</sub>, s<sub>i</sub>)

Output Feature map Layer I

For this specific question we start with a single neuron in the deepest layer which is the 3rd layer in this question. The receptive field of the deepest layer (No. 3) relative to itself will be just one  $r_3=1$ . We can determine the receptive field  $r_2$  of the 2nd layer (again relative to a neuron in deepest 3rd layer) as

$$r_2 = s_3 \cdot r_3 + (k_3 - s_3) = 1 \cdot 1 + (4 - 1) = 4$$

We can then continue to apply the formula, calculating  $r_1$  and  $r_0$  which is the receptive field at the input relative to the deepest layer, which is the quantity we are looking for.

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to 26

100% of points

#### Exercise 107478

QUESTION

Consider the convolution operation between the two feature maps as shown, where

 $n \times m \times C = 60 \times 60 \times 88$ ,  $k \times k = 5$  and there are K = 250 kernels in the convolution layer between the feature maps. The stride of the kernel during convolution is s = 2 and the padding is zero.

Calculate the size of the output feature map  $m' \times n' \times N$  (Note the "Guide to convolutional arithmetic in Deep Learning" (https://arxiv.org/pdf/1603.07285) may be useful in

answering this question). Give your answer as  $1 \ \times 3$  matrix.

#### ✓ Detailed solution

We will firstly determine the spatial dimension  $m' \times n'$  of the output feature maps. In order to do this we can use a suitable relationship from the "Guide to Convolution Arithmetic" In the question we are told the dimensions of the input feature maps, as well as the kernel size the stride and if padding is used. Since there is no padding and the stride is greater than 1. We can use relationship 5 (section 2.3) from the guide.

Relationship 5: For any 
$$i$$
 ,  $k$  and  $s$  , and for  $p=0$  then  $o=\lfloor \frac{i-k}{s} \rfloor +1$ 

This relationship is for one dimension, but since the spatial dimension are equal in our case, we only need to use the formula once. The input size i=m=n=60, k=5, s=2so we calculate  $o=m'=n'=\lfloor\frac{i-k}{s}\rfloor+1=\lfloor\frac{60-5}{2}\rfloor+1=28$ 

Therefore the output spatial size will be equal to  $m' \times n' = 28 \times 28$ . The final size including the number of channels will be  $m' \times n' \times K = 28 \times 28 \times 250$  as the number of output channels is equal to the number of kernels. (Note that we did not need to use the number of input channels in this calculation)

#### Question at wrong answer

We will firstly determine the spatial dimension  $m' \times n'$  of the output feature maps. In order to do this we can use a suitable relationship from the "Guide to Convolution Arithmetic" In the question we are told the dimensions of the input feature maps, as well as the kernel size the stride and if padding is used. Since there is no padding and the stride is greater than 1. We can use relationship 5 (section 2.3) from the guide.

Relationship 5: For any i, k and s, and for p=0 then  $o=\lfloor \frac{i-k}{s} \rfloor +1$ 

This relationship is defined for one dimension, but since the spatial dimension are equal in our case, we only need to use the formula once.

The final size including the number of channels will be  $m' \times n' \times K = 28 \times 28 \times 250$  as the number of output channels is equal to the number of kernels.

 $\bullet$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 28 & 28 & 250 \end{pmatrix}$ 

100% of points

QUESTION

## Calculate the number of parameters in the convolution layer (assume no bias parameter).

#### ☑ Detailed solution

The number of parameters in the convolution layer will be the sum of the total number of parameters associated with each kernel (we will assume there is no bias parameter.) Each individual kernel has a spatial dimension of  $k \times k$  and a total volume of  $k \times k \times C$  where C is the number of channels in the input feature map, as the kernel must have the same number of channels in order to process it.

The total number of parameters  $N_p$  will be the dimensions of the kernel multiplied by the total number of kernels K=250.

$$N_p = k \times k \times C \times K = 5 \cdot 5 \cdot 88 \cdot 250 = 550000$$

#### × Question at wrong answer

The number of parameters in the convolution layer will be the sum of the total number of parameters associated with each kernel (we will assume there is no bias parameter.) Each individual kernel has a spatial dimension of  $k \times k$  and a total volume of  $k \times k \times C$  where C is the number of channels in the input feature map, as the kernel must have the same number of channels in order to process it.

The total number of parameters  $N_p$  will be the dimensions of the kernel multiplied by the total number of kernels K=250.

$$N_p = k \times k \times C \times K$$

☑ Feedback when answer is equal to 550000

100% of points

QUESTION

Calculate the number of multiplication operations associated with convolution.

#### ✓ Detailed solution

In order to determine the number of multiplications  $N_m$  we need to determine the number of spatial positions  $N_s$  of the kernel (in the input image during convolution) and multiply this by the number of parameters  $N_p$ .

$$N_m = N_s \times N_p$$

This can be difficult to determine and depends on the kernel spatial size, stride and padding. However the simplest way to determine  $N_s$  is to just look at the dimensions of the output feature map if (known). Each position in the output feature map corresponding to a convolution of the kernel with the input image.

We have already calculated in part 1 of the question that  $m' \times n' = 28 \times 28$ 

These values m', n' are the size of the output feature map. Therefor the number of spatial positions is simply calculated as

$$N_s=m'\times n'=28\times 28=784$$

We have also calculated in part 2 the number of parameters  $N_p=550000\,\mathrm{associated}$  with the kernels.

The number of multiplication operations  $N_p$  is then given

$$N_m = N_s \times N_p = 784 \cdot 550000 = 431200000$$

#### ■ Question at wrong answer

We have already calculated in part 1 of the question that  $m' \times n' = 28 \times 28$ 

These values m', n' are the size of the output feature map. Therefor the number of spatial positions is simply calculated as

$$N_s=m'\times n'=28\times 28=784$$

We have also calculated in part 2 the number of parameters  $N_p=550000\,\mathrm{associated}$  with the kernels.

The number of multiplication operations  $N_p$  is then given

$$N_m = N_s \times N_p$$

☑ Feedback when answer is equal to 431200000

100% of points

#### Exercise 107544

QUESTION

Consider the convolution operation between the two feature maps as shown, where

 $n \times m \times C = 18 \times 18 \times 52$ ,  $k \times k = 7$  and there are K = 58 kernels in the convolution layer between the feature maps. The stride of the kernel during convolution is s = 1 and the padding is p = 2.

Calculate the size of the output feature map  $m' \times n' \times N$  (Note the "Guide to convolutional arithmetic in Deep Learning" (https://arxiv.org/pdf/1603.07285) may be useful in

answering this question). Give your answer as  $1 \times 3$  matrix.

#### ✓ Detailed solution

We will firstly determine the spatial dimension  $m' \times n'$  of the output feature maps. In order to do this we can use a suitable relationship from the "Guide to Convolution Arithmetic" In the question we are told the dimensions of the input feature maps, as well as the kernel size the stride and if padding is used. Since the stride is equal to 1 and there is zero padding. We can use relationship 2 (section 2.2) from the guide.

Relationship 2: For any i , k and p , and for s=1 then o=(i-k)+2p+1

This relationship is for one dimension, but since the spatial dimension are equal in our case, we only need to use the formula once. The input size i=m=n=18, k=7, p=2 and s=1so we calculate  $o=m'=n'=(i-k)+2p+1=(18-7)+2\cdot 2+1=16$ 

Therefore the output spatial size will be equal to  $m' \times n' = 16 \times 16$ . The final size including the number of channels will be  $m' \times n' \times K = 16 \times 16 \times 58$  as the number of output channels is equal to the number of kernels. (Note that we did not need to use the number of input channels in this calculation)

#### ⋉ Question at wrong answer

We will firstly determine the spatial dimension  $m' \times n'$  of the output feature maps. In order to do this we can use a suitable relationship from the "Guide to Convolution Arithmetic" In the question we are told the dimensions of the input feature maps, as well as the kernel size the stride and if padding is used. Since the stride is equal to 1 and there is zero padding. We can use relationship 2 (section 2.2) from the guide.

Relationship 2: For any i , k and p , and for s=1 then o=(i-k)+2p+1

This relationship is defined for one dimension, but since the spatial dimension are equal in our case, we only need to use the formula once.

The final size including the number of channels will be  $m' \times n' \times K = 16 \times 16 \times 58$  as the number of output channels is equal to the number of kernels.

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 16 & 16 & 58 \end{pmatrix}$ 

100% of points

QUESTION

## Calculate the number of parameters in the convolution layer (assume no bias parameter).

#### ✓ Detailed solution

The number of parameters in the convolution layer will be the sum of the total number of parameters associated with each kernel (we will assume there is no bias parameter.) Each individual kernel has a spatial dimension of  $k \times k$  and a total volume of  $k \times k \times C$  where C is the number of channels in the input feature map, as the kernel must have the same number of channels in order to process it.

The total number of parameters  $N_p$  will be the dimensions of the kernel multiplied by the total number of kernels K=58.

$$N_p = k \times k \times C \times K = 7 \cdot 7 \cdot 52 \cdot 58 = 147784$$

#### × Question at wrong answer

The number of parameters in the convolution layer will be the sum of the total number of parameters associated with each kernel (we will assume there is no bias parameter.) Each individual kernel has a spatial dimension of  $k \times k$  and a total volume of  $k \times k \times C$  where C is the number of channels in the input feature map, as the kernel must have the same number of channels in order to process it.

The total number of parameters  $N_p$  will be the dimensions of the kernel multiplied by the total number of kernels K=58.

$$N_p = k \times k \times C \times K$$

✓ Feedback when answer is equal to 147784

100% of points

QUESTION

Calculate the number of multiplication operations associated with convolution.

#### ✓ Detailed solution

In order to determine the number of multiplications  $N_m$  we need to determine the number of spatial positions  $N_s$  of the kernel (in the input image during convolution) and multiply this by the number of parameters  $N_p$ .

$$N_m = N_s \times N_p$$

This can be difficult to determine and depends on the kernel spatial size, stride and padding. However the simplest way to determine  $N_s$  is to just look at the dimensions of the output feature map if (known). Each position in the output feature map corresponding to a convolution of the kernel with the input image.

We have already calculated in part 1 of the question that  $m' \times n' = 16 \times 16$ 

These values m', n' are the size of the output feature map. Therefor the number of spatial positions is simply calculated as

$$N_s=m'\times n'=16\times 16=256$$

We have also calculated in part 2 the number of parameters  $N_p=147784\,\mathrm{associated}$  with the kernels.

The number of multiplication operations  $N_p$  is then given

$$N_m = N_s \times N_p = 256 \cdot 147784 = 37832704$$

#### ■ Question at wrong answer

We have already calculated in part 1 of the question that  $m' \times n' = 16 \times 16$ 

These values m', n' are the size of the output feature map. Therefor the number of spatial positions is simply calculated as

$$N_s=m'\times n'=16\times 16=256$$

We have also calculated in part 2 the number of parameters  $N_p=147784\,\mathrm{associated}$  with the kernels.

The number of multiplication operations  $N_p$  is then given

$$N_m = N_s \times N_p$$

✓ Feedback when answer is equal to

100% of points

#### Exercise 107547

QUESTION

A bottleneck convolution is inserted into a convolution operation as shown below. (The original operation applied the  $k \times k$  kernels to the input feature map).

Calculate the reduction in the number of multiplications given  $m \times n \times C = 36 \times 36 \times 136$ ,

$$N=54$$
 and  $P=18$ 

Assume that the stride is always 1 and that "same" padding is used with the  $k \times k = 7 \times 7$  convolution.

Check my answer

#### ✓ Detailed solution

We will initially focus on the original convolution and calculate the number of multiplications in this case:

We note that we have a kernel size of k=7 a stride of s=1. From the guide to convolutional arithmetic Relationship 3, we can see that we meet the conditions for "Same" padding i.e. the image size is even, the kernel is odd and we are using unit strides. Therefore in order to ensure "same" padding as instructed in the question

$$p=\lfloor \frac{k}{2}\rfloor=\lfloor \frac{7}{2}\rfloor=6$$
 .

As we are using "same" padding this means that the output image size is the same as the input image size and  $m' \times n' = m \times n$ .

We note there will be one parameter for each spatial position in the kernel, and a  $k \times k$  kernel for each channel in the input image, giving  $k \times k \times C$ . This is then multiplied by the number of kernels giving the number of parameters in the original convolution as  $k \times k \times C \times N$ .

The number of multiplications in the original convolution  $N_{M0}$  will be the number of parameters we have just calculated, multiplied by the number of spatial positions in the input (which is the same as the spatial size of the output feature map)

$$N_{M0}=k\times k\times C\times N\times m'\times n'=7\times 7\times 136\times 54\times 36\times 36=466373376$$

We will now focus on determining the number of multiplications with the bottleneck convolution in place.

For the  $1 \times 1$  bottle neck convolution. The number of parameters in each kernel that operates on the input will be  $1 \times 1 \times C$ . Given there

are P kernels in the bottleneck layer, the total number of parameters will be  $1 \times 1 \times C \times P$  .

The number of multiplications  $N_{M1}$  will be the number of parameters multiplied by the number of spatial positions of the kernels in the input image (which is the same as the output feature map spatial size. The output feature map size is always the same as the input image with  $1 \times 1$  convolution)

$$N_{M1} = 1 \times 1 \times C \times P \times m \times n = 1 \times 1 \times 136 \times 18 \times 36 \times 36 = 3172608$$

next we calculate the number if multiplications from the  $k \times k = 7 \times 7$  convolution. We note that the  $k \times k$  kernel operates on the feature map at the output of the bottleneck convolution with P channels, (not the input with C channels). The number of parameters in each kernel will therefore be  $k \times k \times P$ , and the total number of parameters across all kernels is  $k \times k \times P \times N$ .

The total number of multiplications  $N_{m2}$  will be the number of parameters multiplied by by the number of spatial positions, as we are using "same" padding the input feature map and output feature map sizes are the identical  $m' \times n' = m \times n$ .

$$N_{m2} = k \times k \times P \times N \times m' \times n' = 7 \times 7 \times 18 \times 54 \times 36 \times 36 = 61725888$$

Finally the reduction in the number of multiplications will be  $N_{mr}=N_{m0}-N_{m1}-N_{m2}=466373376-3172608-61725888=401474880$ 

#### 

We will initially focus on the original convolution and calculate the number of multiplications in this case:

We note that we have a kernel size of k=7 a stride of s=1. From the guide to convolutional arithmetic Relationship 3, we can see that

we meet the conditions for "Same" padding i.e. the image size is even, the kernel is odd and we are using unit strides. Therefore in order to ensure "same" padding as instructed in the question

$$p=\lfloor \frac{k}{2}\rfloor=\lfloor \frac{7}{2}\rfloor=6$$
 .

As we are using "same" padding this means that the output image size is the same as the input image size and  $m' \times n' = m \times n$ .

We note there will be one parameter for each spatial position in the kernel, and a  $k \times k$  kernel for each channel in the input image, giving  $k \times k \times C$ . This is then multiplied by the number of kernels giving the number of parameters in the original convolution as  $k \times k \times C \times N$ .

The number of multiplications in the original convolution  $N_{M0}$  will be the number of parameters we have just calculated, multiplied by the number of spatial positions in the input (which is the same as the spatial size of the output feature map)

$$N_{M0}=k\times k\times C\times N\times m'\times n'=7\times 7\times 136\times 54\times 36\times 36=466373376$$

We need to then repeat this calculation for the bottleneck convolution and the  $k \times k$  convolution that is now applied to the output feature map of the bottleneck. We should observe that the number of multiplications is substantially reduced.

✓ Feedback when answer is equal to 401474880

100% of points

#### Exercise 107549

**QUESTION** 

A fully connected layer connects  $N=1024 \, \mathrm{inputs}$  to 10 outputs (neurons). Calculate the number of parameters in the fully connected layer.

#### Detailed solution

Since we have a fully connected layer, all the N inputs must connect to each of the 10 neurons producing the output. There is a weight parameter for each connection from the input to the output neurons. A bias parameter is also included for each output neuron.

$$N_p = N \times 10 + 10 = 1024 \times 10 + 10 = 10250$$

#### □ Question at wrong answer

Since we have a fully connected layer, all the N inputs must connect to each of the 10 neurons producing the output. There is a weight parameter for each connection from the input to the output neurons. A bias parameter is also included for each output neuron.

☑ Feedback when answer is equal to 10250

100% of points

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 107563

**QUESTION** 

During training of neural networks, what problems will we encounter with too low or too high learning rates? (select the correct option from the list below)

|          | Too low: Slow convergence; Too high: Non-convergence or divergent behaviour  |              |  |  |
|----------|------------------------------------------------------------------------------|--------------|--|--|
|          |                                                                              |              |  |  |
|          | Too low: Fast convergence; Too high: Smooth convergence                      | ×            |  |  |
|          |                                                                              |              |  |  |
|          | Too low: Overfitting; Too high: Underfitting                                 | ×            |  |  |
|          | Select one answer. Upon selection it will submit your answer immediately.    |              |  |  |
| <b>▽</b> | ✓ Feedback for answer: 1 100% of points                                      |              |  |  |
|          | Correct!                                                                     |              |  |  |
| ×        | Feedback for answer: 2                                                       | 0% of points |  |  |
|          | Incorrect: Fast convergence is not a problem associated with lov             | V            |  |  |
|          | learning rates (it is likely to converge slowly). With too high a lea        | rning        |  |  |
|          | rate, convergence may not occur (may likely observe divergent                |              |  |  |
|          | behaviour)                                                                   |              |  |  |
| ×        | Feedback for answer: 3                                                       | 0% of points |  |  |
|          | T COMMUNITOR WITHOUT OF                                                      | on or points |  |  |
|          | Incorrect: Overfitting is not related to learning rate, it likely due to too |              |  |  |
|          | much model complexity. Underfitting can occur due to low learning            |              |  |  |

rates, as the network gets stuck in a local minimum.

QUESTION

# What are the key differences between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent?

| ×        | Feedba | ck for answer: 3                                                                                | 0% of points  |
|----------|--------|-------------------------------------------------------------------------------------------------|---------------|
|          |        | rrect: The definitions of batch and stochastic gradient desceroped.                             | nt are        |
| ×        | Feedba | ck for answer: 2                                                                                | 0% of points  |
|          | Corr   | ect!                                                                                            |               |
| <b>~</b> | Feedba | ck for answer: 1                                                                                | 00% of points |
|          | Select | one answer. Upon selection it will submit your answer immediately.                              |               |
|          |        | All examples at once; Stochastic: A subset of examples; Mini-batch: One example at a time       | ×             |
|          | 0      | Batch: A subset of examples; Stochastic: All examples at once; Minibatch: One example at a time | X             |
|          | 0      | Batch: One example at a time; Stochastic: All examples at once; Minibatch: A subset of examples | ×             |
|          | 0      | Batch: All examples at once; Stochastic: One example at a time; Minibatch: A subset of examples | ✓             |
|          |        |                                                                                                 |               |

Incorrect: The definitions of batch, stochastic, and mini-batch gradient descent are incorrect.

Incorrect: The definitions of stochastic and mini-batch gradient descent are swapped.

QUESTION

## Name some solutions to overfitting when training deep neural networks.

| 0                                                                               | Reduced complexity, Early stopping, Regularization, Dropout, Batch normalization, More training data   | $\checkmark$ |  |  |
|---------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|--------------|--|--|
|                                                                                 |                                                                                                        |              |  |  |
| $\bigcirc$                                                                      | Increased complexity, Early stopping, Regularization, Dropout, Batch normalization, Less training data | ×            |  |  |
|                                                                                 |                                                                                                        |              |  |  |
|                                                                                 | Reduced complexity, Late stopping, Regularization, Dropout, Batch normalization, More training data    | ×            |  |  |
|                                                                                 |                                                                                                        |              |  |  |
| 0                                                                               | Reduced complexity, Early stopping, Regularization, Dropout, Batch normalization, Less training data   | ×            |  |  |
| Select                                                                          | one answer. Upon selection it will submit your answer immediately.                                     |              |  |  |
| Feedback for answer: 1 100% of points                                           |                                                                                                        |              |  |  |
| Corr                                                                            | ect!                                                                                                   |              |  |  |
| Feedba                                                                          | ck for answer: 2                                                                                       | 0% of points |  |  |
| Incorrect: Increased complexity and less training data can lead to overfitting. |                                                                                                        |              |  |  |
|                                                                                 |                                                                                                        |              |  |  |

0% of points

Incorrect: Late stopping (allowing training to continue past convergence of training loss) can lead to overfitting (and a reduction in validation accuracy).

× Feedback for answer: 4

0% of points

Incorrect: Less training data can lead to overfitting.

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 107720

**QUESTION** 

A convolutional neural network was trained using mini-batch stochasitic gradient descent with a batch size of  $B=128.00\,\mathrm{on}$  a training dataset of  $N_T=40000\,\mathrm{for}$  for  $n_t=7943282\,\mathrm{steps}$ . How many epochs  $N_e$  was the network trained for (round your answer to the nearest epoch).

#### ✓ Detailed solution

The number of epochs is simple to calculate given the batch size B, training data size N and total number of training steps  $n_t$ .

As a single batch will be processed on each step, we divide the training data size by the batch by the batch size to determine how many steps are needed to process the entire dataset. This will be the number of steps in an epoch  $n_e$ 

$$n_e = \frac{N}{B} = \frac{40000}{12800} = 313$$

The number of epochs  $N_e$  is then simply calculated by dividing the total number of steps by the number of steps in an epoch.

$$N_e=\frac{n_t}{n_e}=25378$$

Although this calculation is simple it highlights a consideration to keep in mind when training a neural network. If during training the accuracy and loss are reported at the end of an epoch, at lot of training will already have taken place, particularly if the dataset is large. Sometimes non-convergence issues can occur after only a few batches, similarly a network may begin to overfit and the initial fall and then rise of validation data is missed.

#### Question at wrong answer

The number of epochs is simple to calculate given the batch size B, training data size N and total number of training steps  $n_t$ .

As a single batch will be processed on each step, we divide the training data size by the batch by the batch size to determine how many steps are needed to process the entire dataset. This will be the number of steps in an epoch  $n_e$ 

$$n_e = \frac{N}{B}$$

The number of epochs  $N_e$  is then simply calculated by dividing the total number of steps by the number of steps in an epoch.

✓ Feedback when answer is equal to 25378

100% of points