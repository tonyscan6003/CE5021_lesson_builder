## <u>View subject</u> | <u>Edit subject</u>

# 2D Convolution

Slides

### **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 105731

**QUESTION** 

2D Convolution (Parameterised)

Find the result of convolving the 2x2 kernel

$$A=\begin{pmatrix} -2 & 0 \ 2 & 2 \end{pmatrix}$$
 with the 3x3 image  $B=\begin{pmatrix} -2 & 3 & 2 \ 3 & 2 & 0 \ 2 & 2 & 1 \end{pmatrix}$  .

(Assuming no-padding this is also called "Valid Padding".). Give your answer as a 2x2 matrix in the box below.

✓ Detailed solution

We perform the convolution graphically.

We must first remember to reverse (or flip the kernel) before convolution.

$$A'=\begin{pmatrix} 2 & 2 \ 0 & -2 \end{pmatrix}$$

We step the kernel around the image and multiply the kernel by the pixels of the image at each position and sum the individual multiplications.

$$\begin{pmatrix} 2 \cdot -2 + 2 \cdot 3 + 0 \cdot 3 - 2 \cdot 2 & 2 \cdot 3 + 2 \cdot 2 + 0 \cdot 2 - 2 \cdot 0 \\ 2 \cdot 3 + 2 \cdot 2 + 0 \cdot 2 - 2 \cdot 2 & 2 \cdot 2 + 2 \cdot 0 + 0 \cdot 2 - 2 \cdot 1 \end{pmatrix}$$

This is equal to  $\begin{pmatrix} -2 & 10 \\ 6 & 2 \end{pmatrix}$ 

#### Question at wrong answer

We perform the convolution graphically.

We must first remember to reverse (or flip the kernel) before convolution.

$$A'=\begin{pmatrix} 2 & 2 \ 0 & -2 \end{pmatrix}$$

We step the kernel around the image and multiply the kernel by the pixels of the image at each position and sum the individual multiplications.

For example at the first position (top left) in B, we perform element wise multiplication between  $\begin{pmatrix} -2 & 3 \\ 3 & 2 \end{pmatrix}$  and  $\begin{pmatrix} 2 & 2 \\ 0 & -2 \end{pmatrix}$  and sum the result to get the output value (also top left in output).

$$O_{00} = -2 \cdot 2 + 3 \cdot 2 + 3 \cdot 0 + 2 \cdot -2 = -2$$

Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} -2 & 10 \\ 6 & 2 \end{pmatrix}$  100% of points No specific feedback