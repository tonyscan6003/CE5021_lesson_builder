<u>View subject</u> | <u>Edit subject</u>

# **Transpose & Dilated Convolution**

Slides

# **Exercises**

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 108849

**QUESTION** 

The  $3\times 3$  image patch  $I=\begin{pmatrix}5&1&2\\3&2&4\\3&2&1\end{pmatrix}$  is convolved with the kernel  $K=\begin{pmatrix}0&3\\2&-2\end{pmatrix}$  to produce a  $2\times 2$  output  $O=\begin{pmatrix}1&8\\7&10\end{pmatrix}$ . (As the output size is reduced we assume there was no zero padding="Valid" Padding).

Perform the convolution  $I \circledast K$  using the matrix method of convolution. This requires creating a sparse matrix S of size  $4 \times 9$  that can be multiplied by against the unrolled input  $I_{ur}$  which is of size  $9 \times 1$ , to produce an unrolled output  $O_{ur} = S \times I_{ur}$ , (which will have the same values as O). Give the sparse  $9 \times 4$  matrix you have calculated in the text box below.

# Check my answer

#### Detailed solution

In computing the convolution operation we multiply and sum the (reversed) kernel with elements of the image, at each position in the image. At any given position, this is equivalent to multiplying the other non-zero elements of the image by zeros (i.e. effectively enlarging the kernel with zeros).

For example the 1st convolution product can be written as an elementwise multiplication  $(\odot)$  and sum between extended kernel and image.

$$\sum \begin{pmatrix} -2 & 2 & 0 \\ 3 & 0 & 0 \\ 0 & 0 & 0 \end{pmatrix} \odot \begin{pmatrix} 5 & 1 & 2 \\ 3 & 2 & 4 \\ 3 & 2 & 1 \end{pmatrix} = 1$$

For this specific kernel position we can re-write the elementwise multiplication and sum as the product of 1D arrays that are matrix multiplied with) by one another. One array consists of elements of the kernel with additional zeros, and the other is the unrolled elements of the image.

$$\begin{pmatrix} (-2 & 2 & 0 & 3 & 0 & 0 & 0 & 0 & 0 \end{pmatrix} \times \begin{pmatrix} 5 \ 1 \ 2 \ 3 \ 2 \ 4 \ 3 \ 2 \ 1 \end{pmatrix} = 1$$

For every individual kernel position there will be a unique 1D array to multiply by the unrolled elements of the image. There will be 4 positions for the kernel in this convolution operation (equal to the number of convolution output values). We can combine these into the sparse matrix S as shown, which is then multiplied by the unrolled image  $I_{ur}$  to give the 4 convolution products.

$$\begin{pmatrix} -2 & 2 & 0 & 3 & 0 & 0 & 0 & 0 & 0 \\ 0 & -2 & 2 & 0 & 3 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & -2 & 2 & 0 & 3 & 0 & 0 \\ 0 & 0 & 0 & 0 & -2 & 2 & 0 & 3 & 0 \end{pmatrix} \times \begin{pmatrix} 5 \\ 1 \\ 2 \\ 3 \\ 2 \\ 4 \\ 3 \\ 2 \\ 1 \end{pmatrix} =$$

We can verify that the Sparse matrix is correct if the same values (as convolution) are returned after matrix multiplication:

$$S\times I_{ur} = \begin{pmatrix} 1 \ 8 \ 7 \ 10 \end{pmatrix}$$

#### Question at wrong answer

In computing the convolution operation we multiply and sum the (reversed) kernel with elements of the image, at each position in the image. At any given position, this is equivalent to multiplying the other

non-zero elements of the image by zeros (i.e. effectively enlarging the kernel with zeros).

For example the 1st convolution product can be written as an elementwise multiplication  $(\odot)$  and sum between extended kernel and image.

$$\sum \begin{pmatrix} -2 & 2 & 0 \ 3 & 0 & 0 \ 0 & 0 & 0 \end{pmatrix} \odot \begin{pmatrix} 5 & 1 & 2 \ 3 & 2 & 4 \ 3 & 2 & 1 \end{pmatrix} = 1$$

For this specific kernel position we can re-write the elementwise multiplication and sum as the product of 1D arrays that are matrix multiplied with) by one another. One array consists of elements of the kernel with additional zeros, and the other is the unrolled elements of the image.

$$\begin{pmatrix} \left(-2 & 2 & 0 & 3 & 0 & 0 & 0 & 0 & 0 \end{pmatrix} \times \begin{pmatrix} 5 \ 1 \ 2 \ 3 \ 2 \ 4 \ 3 \ 2 \ 1 \end{pmatrix} = 1$$

For every individual kernel position there will be a unique 1D array to multiply by the unrolled elements of the image. There will be 4 positions for the kernel in this convolution operation (equal to the number of convolution output values). We can combine these into the sparse matrix S as shown, which is then multiplied by the unrolled image  $I_{ur}$  to give the 4 convolution products.

No specific feedback

QUESTION

Using the sparse matrix determined in the first part of the question perform the transpose convolution (associated with the convolution  $I \circledast K$ ) with the image patch  $X = \begin{pmatrix} 2 & 5 \\ 2 & 4 \end{pmatrix}$ . (Give your answer as an unrolled  $9 \times 1$  matrix in the box below)

#### ✓ Detailed solution

As we have seen in the first part of this question a convolution  $I \circledast K = O$  can be described as matrix multiplication  $S \times I_{ur} = O_{ur}$  where  $I_{ur}$  is the unrolled input. Using the sparse matrix S we can define a "Transpose" convolution associated with the convolution K. Typically the convolution operation reduces the size of the input

(unless we use zero padding). The Transpose convolution defines an operation where the size of the image is increased.

In order to calculate the result of the "Transpose" convolution Y defined by the Kernel K we can simply matrix multiply the input X by  $S^T$ .

$$Y = S^T \times X$$

$$Y = \begin{pmatrix} -2 & 0 & 0 & 0 \ 2 & -2 & 0 & 0 \ 0 & 2 & 0 & 0 \ 3 & 0 & -2 & 0 \ 0 & 3 & 2 & -2 \ 0 & 0 & 0 & 2 \ 0 & 0 & 3 & 0 \ 0 & 0 & 0 & 3 \ 0 & 0 & 0 & 0 \end{pmatrix} \times \begin{pmatrix} 2 \ 5 \ 2 \ 4 \end{pmatrix} = \begin{pmatrix} -4 \ -6 \ 10 \ 2 \ 11 \ 8 \ 6 \ 12 \ 0 \end{pmatrix}$$

#### 

As we have seen in the first part of this question a convolution  $I \circledast K = O$  can be described as matrix multiplication  $S \times I_{ur} = O_{ur}$  where  $I_{ur}$  is the unrolled input. Using the sparse matrix S we can define a "Transpose" convolution. Typically the convolution operation reduces the size of the input (unless we use zero padding). The Transpose convolution defines an operation where the size of the image is increased.

In order to calculate the result of the "Transpose" convolution Y defined by the Kernel K we can simply matrix multiply the input X by  $S^T$ .

$$Y = S^T \times X$$

| Feedback when answer is algebraically Equivalent to | $\begin{pmatrix} -4 \\ -6 \\ 10 \\ 2 \\ 11 \\ 8 \\ 6 \\ 12 \\ 0 \end{pmatrix}$ | 100% of points |
|-----------------------------------------------------|--------------------------------------------------------------------------------|----------------|
| No specific feedback                                |                                                                                |                |

### <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 108904

**QUESTION** 

A convolution operation convolves the image I which is of size  $3\times 3$  with a kernel  $K=\begin{pmatrix} 1 & -1 \\ -2 & 0 \end{pmatrix}$ , using a stride 1 and no zero padding resulting in a  $2\times 2$  matrix O.

Find the output Y of the associated transpose convolution with the image  $X=\begin{pmatrix} 4 & 1 \\ 4 & 4 \end{pmatrix}$  . Give your answer as a matrix in the box below.

#### ✓ Detailed solution

The convolution  $O=I\circledast K$  has a transpose convolution associated with it. We can obtain the transpose convolution for any convolution by finding the sparse matrix S. The transpose of the sparse matrix

 $S^T$  can be multiplied against any image X (usually the same asize as O) to obtain the result Y of the transpose convolution.

As an alternative to finding the sparse matrix, it is possible to obtain an equivalent direct convolution to any transpose convolution. In "The Guide to Convolutional Arithmetic" relationships are defined for equivalent direct convolutions for various forward convolutions. For the convolution  $I \circledast K$  defined in this question, we can use Section 4.3, relationship 8:

A convolution described by stride s=1,no zero padding p=0, with a kernel size k, has an associated transpose convolution described by k'=k, s'=s and p'=k-1 and the output size is o'=i'+(k-1). Where i' is the size of the input image X for the transpose convolution.

Since the original kernel K has a size of k=1 we have to apply padding to the target image of size p'=k-1=1, therefore one ring of zero padding is required.

Since the input image to the transpose convolution has a size of i'=2, the output image will be  $\ o'=i'+(k-1)=2+(2-1)=3$ . As expected the transpose convolution increases the size of the image from input to output.

We now carry out the transpose convolution as an equivalent direct convolution:

$$\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 4 & 1 & 0 \\ 0 & 4 & 4 & 0 \\ 0 & 0 & 0 & 0 \end{pmatrix} \circledast \begin{pmatrix} 1 & -1 \\ -2 & 0 \end{pmatrix} = \begin{pmatrix} 4 & -3 & -1 \\ -4 & -2 & -4 \\ -8 & -8 & 0 \end{pmatrix}$$

 <sup>□</sup> Question at wrong answer

The convolution  $O=I \circledast K$  has a transpose convolution associated with it. We can obtain the transpose convolution for any convolution by finding the sparse matrix S. The transpose of the sparse matrix  $S^T$  can be multiplied against any image X (usually the same asize as O) to obtain the result Y of the transpose convolution.

As an alternative to finding the sparse matrix, it is possible to obtain an equivalent direct convolution to any transpose convolution. In "The Guide to Convolutional Arithmetic" relationships are defined for equivalent direct convolutions for various forward convolutions. For the convolution  $I \circledast K$  defined in this question, we can use Section 4.3, relationship 8:

A convolution described by stride s=1,no zero padding p=0, with a kernel size k, has an associated transpose convolution described by k'=k, s'=s and p'=k-1 and the output size is o'=i'+(k-1). Where i' is the size of the input image X for the transpose convolution.

Since the original kernel K has a size of k=1 we have to apply padding to the target image of size p'=k-1=1, therefore one ring of zero padding is required.

Since the input image to the transpose convolution has a size of i'=2, the output image will be o'=i'+(k-1)=2+(2-1)=3. As expected the transpose convolution increases the size of the image from input to output.

The transpose convolution is carried out as an equivalent direct convolution between the input X with padding and the kernel.

 $\checkmark$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 4 & -3 & -1 \\ -4 & -2 & -4 \\ -8 & -8 & 0 \end{pmatrix}$ 

100% of points

No specific feedback

## Exercise 108948

**QUESTION** 

A convolution operation convolves the image I which  $\begin{pmatrix} 2 & 0 & 2 \end{pmatrix}$ 

is of size 
$$5\times 5$$
 with a  $3\times 3$  kernel  $K=\begin{pmatrix} 2 & 0 & 2 \ 1 & -1 & 1 \ -2 & -1 & 1 \end{pmatrix}$  ,

using a stride 2 and no zero padding resulting in a  $2 \times 2$  output matrix O.

Find the output Y of the associated transpose convolution with the image  $X=\begin{pmatrix} 4 & 5 \\ 1 & 2 \end{pmatrix}$  using an equivalent direct convolution. Give your answer as a  $5\times 5$  matrix in the box below.

Check my answer

Detailed solution

The convolution  $O=I \circledast K$  has a transpose convolution associated with it. We can obtain the transpose convolution for any convolution by finding the sparse matrix S. The transpose of the sparse matrix  $S^T$  can be multiplied against any image X (usually the same asize as O) to obtain the result Y of the transpose convolution.

As an alternative to finding the sparse matrix, it is possible to obtain an equivalent direct convolution to any transpose convolution. In "The Guide to Convolutional Arithmetic" relationships are defined for equivalent direct convolutions for various forward convolutions. For the convolution  $I \circledast K$  defined in this question, we can use Section 4.3, relationship 12:

A convolution described by no zero padding p=0, with a kernel size k and stride s whose input size is such that i-k is a multiple of s, has an associated transpose convolution described by k'=k, s'=1, p'=k-1 and where the i' is the size of the stretched input obtained by adding s-1 zeros between each input unit and the output size is o'=s (i'-1) +k.

Since the original kernel K has a size of k=3 and s=2, p=0 and the original image size i=5. We see that the condition for the relationship 12 to apply is satisfied i.e. i-k=5-3=2 is a multiple of the stride.

Since the input image to the transpose convolution has a size of i'=2, the output image will be  $o'=s\,(i'-1)+k=2\,(2-1)+3=5$ . As expected the transpose convolution increases the size of the image from input to output.

We now carry out the the transpose convolution as an equivalent direct convolution, as instructed by the relationship we must insert s-1zeros between the each input unit and add padding

p'=k-1=3-1=2, which is two rings of zero padding, we can then apply the direct convolution with the kernel K.

$$\begin{pmatrix} 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 &$$

### □ Question at wrong answer

The convolution  $O=I \circledast K$  has a transpose convolution associated with it. We can obtain the transpose convolution for any convolution by finding the sparse matrix S. The transpose of the sparse matrix  $S^T$  can be multiplied against any image X (usually the same asize as O) to obtain the result Y of the transpose convolution.

As an alternative to finding the sparse matrix, it is possible to obtain an equivalent direct convolution to any transpose convolution. In "The Guide to Convolutional Arithmetic" relationships are defined for equivalent direct convolutions for various forward convolutions. For the convolution  $I \circledast K$  defined in this question, we can use Section 4.3, relationship 12:

A convolution described by no zero padding p=0, with a kernel size k and stride s whose input size is such that i-k is a multiple of s, has an associated transpose convolution described by k'=k, s'=1, p'=k-1 and where the i' is the size of the stretched input obtained by adding s-1 zeros between each input unit and the output size is o'=s (i'-1)+k.

Since the original kernel K has a size of k=3 and s=2, p=0 and the original image size i=5. We see that the condition for the

relationship 12 to apply is satisfied i.e. i - k = 5 - 3 = 2 is a multiple of the stride.

Since the input image to the transpose convolution has a size of i'=2, the output image will be  $o'=s\,(i'-1)+k=2\,(2-1)+3=5$ . As expected the transpose convolution increases the size of the image from input to output.

We now carry out the the transpose convolution as an equivalent direct convolution, as instructed by the relationship we must insert s-1 zeros between the each input unit and add padding p'=k-1=3-1=2, which is two rings of zero padding, we can then apply the direct convolution with the kernel K.

Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 8 & 0 & 18 & 0 & 10 \\ 4 & -4 & 9 & -5 & 5 \\ -6 & -4 & 0 & -5 & 9 \\ 1 & -1 & 3 & -2 & 2 \\ -2 & -1 & -3 & -2 & 2 \end{pmatrix}$ 

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 108999

**QUESTION** 

Perform a convolution of a  $7 \times 7$  image I with a  $3 \times 3$  kernel with stride s=1, no zero padding p=0 and a dilation rate d=2. Where  $K=\begin{pmatrix} 1 & -1 & 2 \\ 0 & 0 & 0 \\ 3 & 3 & 0 \end{pmatrix}$  and

$$I = \begin{pmatrix} 3 & 5 & 5 & 1 & 5 & 2 & 4 \ 4 & 3 & 3 & 3 & 3 & 3 & 3 \ 2 & 0 & 2 & 4 & 3 & 1 & 1 \ 5 & 0 & 4 & 4 & 2 & 2 & 2 \ 1 & 2 & 4 & 3 & 4 & 4 & 2 \ 4 & 3 & 4 & 4 & 5 & 3 & 2 \ 5 & 3 & 3 & 3 & 4 & 4 & 3 \end{pmatrix} \,.$$

Give your answer as a matrix in the box below.

✓ Detailed solution

In a dilated convolution (dilation rate d>1), there is an effective increase in kernel size with zeros placed between the elements of the

kernel, and a new effective kernel size  $\hat{k}$ 

We can calculate the effective kernel size in dilated convolutions from the formula  $\hat{k}=k+(k-1)\,(d-1)$  (See Guide To Convolutional Arithmetic, Chapter 5)

In the question the kernel is k=3 and the dilation rate d=2, so the effective kernel size is will be  $\hat{k}=3+(3-1)\,(2-1)=5$ 

The output size can be calculated explicitly for the dilated convolution using relationship 15 from the guide: For any i,k,p and s and a dilation rate d, then the output size o of the convolution will be given by

$$o = \lfloor \frac{i+2p-\hat{k}}{s} \rfloor + 1 = \lfloor \frac{i+2p-k-(k-1)\left(d-1\right)}{s} \rfloor + 1$$

From the question i=7, p=0, s=1 and we have already calculated  $\hat{k}=5$ , this gives an output size of o=3.

For the convolution detailed in the question, we now rewrite the the kernel with the additional zeros inserted. The convolution is then performed as normal.

$$I \circledast \begin{pmatrix} 1 & 0 & -1 & 0 & 2 \ 0 & 0 & 0 & 0 & 0 \ 0 & 0 & 0 & 0 & 0$$

## $\boxed{\boxed}$ Question at wrong answer

In a dilated convolution (dilation rate d>1), there is an effective increase in kernel size with zeros placed between the elements of the kernel, and a new effective kernel size  $\hat{k}$ 

We can calculate the effective kernel size in dilated convolutions from the formula  $\hat{k}=k+(k-1)\,(d-1)$  (See Guide To Convolutional

Arithmetic, Chapter 5)

In the question the kernel is k=3 and the dilation rate d=2, so the effective kernel size is will be  $\hat{k}=3+(3-1)\,(2-1)=5$ 

The output size can be calculated explicitly for the dilated convolution using relationship 15 from the guide: For any i,k,p and s and a dilation rate d, then the output size o of the convolution will be given by

$$o = \lfloor \frac{i+2p-\hat{k}}{s} \rfloor + 1 = \lfloor \frac{i+2p-k-(k-1)\left(d-1\right)}{s} \rfloor + 1$$

From the question i=7, p=0, s=1 and we have already calculated  $\hat{k}=5$ , this gives an output size of o=3.

For the convolution detailed in the question, we now rewrite the the kernel with the additional zeros inserted. The convolution is then performed as normal.

Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 32 & 14 & 33 \\ 27 & 23 & 23 \\ 26 & 22 & 17 \end{pmatrix}$  100% of points

No specific feedback