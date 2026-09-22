## <u>View subject</u> | <u>Edit subject</u>

# 1D Convolution

Slides

### **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 105729

QUESTION

1D Convolution (Mathematical)

Given the 1D convolution equation

$$\left(x\ast h\right)\left[n\right] = \sum_{k=-\infty}^{\infty} x\left[k\right] h\left[n-k\right]$$

Perform convolution between the sequence  $x\left[n\right]=\left[-2,-2,-1\right]$  and the kernel  $h\left[n\right]=\left[0,2,-2\right]$  defined at  $n\ \in\ \left[-1,0,1\right].$ 

Give your answer as  $1 \times 5$  matrix where the elements of the matrix is the result of convolution (x\*h)[n] for  $n \in [-2, -1,0,1,2]$  in the box below:

✓ Detailed solution

We can write the reversed and shifted kernel for each value of n as described by the term h[n-k]. The positive values of n result in a delayed kernel (shifted right) and the negative values of n advanced the reversed kernel to the left. In order to calculate the final convolved values the sequence is multiplied by the reversed/shifted kernel in each position and the values summed.

Step 1: To perform convolution we first re-write the input and kernel in terms of the dummy variable k. i.e. x[k] = [-2, -2, -1]  $k \in [-1,0,1]$  with the kernel h[k] = [0,2, -2] defined at  $k \in [-1,0,1]$ . It is assumed both the sequence and kernel are zero at any k outside the defined range.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \\ x \left[ k \right] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \end{pmatrix}$$

The values of n that we calculate over are those that produce an overlap between the input sequence and the kernel. This will be the range  $n \in [-2,2]$ , in this case. As it is written mathematically in the convolution equation, we theoretically sum from  $k \in [-\infty,\infty]$ . However as with the choice of n, we only need to evaluate at values k over where the input sequence and the reversed/shifted kernel will have non-zero values when they multiply. For this example we just need to use the range  $k \in [-3,3]$ .

Step 2:

For n=-2 we calculate the reversed/shifted kernel at each k

$$h\left(n-k\right) = h\left(-3-k\right)$$

At each value of  $k \in [-3,3]$ 

$$k = -3$$
,  $h(-2 - k) = h(-2 - (-3)) = h(1) = -2$ 

$$k = -2$$
,  $h(-2 - k) = h(-2 - (-2)) = h(0) = 2$ 

$$k = -1$$
,  $h(-2 - k) = h(-2 - (-1)) = h(-1) = 0$ 

$$k = 0$$
,  $h(-2 - k) = h(-2 - (0)) = h(-2) = 0$ 

All values of h(n-k) will be zero after this point (you can check).

We can tabulate the values of k, the sequence  $x\left[n\right]$  and the position of the reversed/shifted kernel based on the  $h\left[-2-k\right]$  values we have determined.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \ x\,[k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \ h\,[-2-k] & -2 & 2 & 0 & 0 & 0 & 0 \end{pmatrix}$$

The convolution result (x\*h) for n=-2 is calculated as the elementwise multiplication and sum of the sequence  $x\left[k\right]$  and reversed/shifted kernel  $h\left[-2-k\right]$ 

$$(x*h)\left[-2\right] = \sum_{k=-3}^{k=3} x\left[k\right] h\left[-2-k\right] = 0\cdot -2 + 0\cdot 2 - 2\cdot 0 - 2\cdot 0 - 1\cdot 0 = 0$$

Step 3:

For n=-1 we calculate the reversed/shifted kernel at each  $\it k$ 

$$h\left(n-k\right) = h\left(-1-k\right)$$

At each value of  $k \in [-3,3]$ 

$$k = -3$$
,  $h(-1 - k) = h(-1 - (-3)) = h(2) = 0$   
 $k = -2$ ,  $h(-1 - k) = h(-1 - (-2)) = h(1) = -2$   
 $k = -1$ ,  $h(-1 - k) = h(-1 - (-1)) = h(0) = 2$   
 $k = 0$ ,  $h(-1 - k) = h(-1 - (0)) = h(-1) = 0$   
 $k = 1$ ,  $h(-1 - k) = h(-1 - (1)) = h(-2) = 0$   
 $k = 2$ ,  $h(-1 - k) = h(-1 - (2)) = h(-3) = 0$ 

k = 3, h(-1 - k) = h(-1 - (3)) = h(-4) = 0

We can tabulate the values of k, the sequence  $x\left[n\right]$  and the position of the reversed/shifted kernel based on the  $h\left[-1-k\right]$  values we have determined.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \\ x \left[ k \right] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ h \left[ -2 - k \right] & -2 & 2 & 0 & 0 & 0 & 0 & 0 \\ h \left[ -1 - k \right] & 0 & -2 & 2 & 0 & 0 & 0 & 0 \end{pmatrix}$$

The convolution result (x\*h) for n=-1 is calculated as the elementwise multiplication and sum of the sequence  $x\left[k\right]$  and reversed/shifted kernel  $h\left[-1-k\right]$ 

$$(x*h)[-1] = \sum_{k=-3}^{k=3} x [n] \, h [-1-k] = 0 \cdot 0 + 0 \cdot -2 - 2 \cdot 2 - 2 \cdot 0 - 1 \cdot 0 = -4$$

Step 3,4...

We repeat this procedure for each of the other values of n and show the results graphically below:

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \\ x [k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ h [-2-k] & -2 & 2 & 0 & 0 & 0 & 0 & 0 \\ h [-1-k] & 0 & -2 & 2 & 0 & 0 & 0 & 0 \\ h [-k] & 0 & 0 & -2 & 2 & 0 & 0 & 0 \\ h [1-k] & 0 & 0 & 0 & -2 & 2 & 0 & 0 \\ h [2-k] & 0 & 0 & 0 & 0 & -2 & 2 & 0 \\ (x*h) [n] & 0 & -4 & 0 & 2 & 2 \end{pmatrix}$$

#### ■ Question at wrong answer

There are two ways to complete this question.

Firstly we will detail the simpler graphical method (where we don't pay attention to values of n or k).

We reverse the kernel h[n] = [-2,2,0]

We shift the reversed kernel to the left so that it just overlaps the input sequence.

$$\begin{pmatrix} 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ -2 & 2 & 0 & 0 & 0 & 0 & 0 \end{pmatrix}$$

We then multiply and sum the reversed and shifted kernel by the sequence, which will give the first value in the output sequence.

$$0 \cdot -2 + 0 \cdot 2 - 2 \cdot 0 - 2 \cdot 0 - 1 \cdot 0 = 0$$

We then shift the reversed kernel to the right by one place

$$\begin{pmatrix} 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ 0 & -2 & 2 & 0 & 0 & 0 & 0 \end{pmatrix}$$

We again obtain the next value in the output sequence as

$$0 \cdot 0 + 0 \cdot -2 - 2 \cdot 2 - 2 \cdot 0 - 1 \cdot 0 = -4$$

You can continue to shift the kernel to the right calculating values as you go along, the last value will be when the reversed kernel again just over laps the sequence.

$$\begin{pmatrix} 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 & -2 & 2 & 0 \end{pmatrix}$$

We will now detail the more formal method which uses the Mathematical Equation for Convolution to calculate the output sequence. In this case we will use the values of n and k. By performing convolution with the mathematical equation, we will see that the reversal and shifting of the kernel happens automatically (we will get the same result as the graphical method). We also obtain the output sequence in terms of n, so we know exactly where in time or position the output sequence values are located.

We can write the reversed and shifted kernel for each value of n as described by the term h[n-k]. The positive values of n result in a delayed kernel (shifted right) and the negative values of n advanced the reversed kernel to the left. In order to calculate the final convolved values the sequence is multiplied by the reversed/shifted kernel in each position and the values summed.

Step 1: To perform convolution we first re-write the input and kernel in terms of the dummy variable k. i.e. x[k]=[-2,-2,-1]  $k\in [-1,0,1]$  with the kernel h[k]=[0,2,-2] defined at  $k\in [-1,0,1]$ . It is assumed both the sequence and kernel are zero at any k outside the defined range.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \\ x [k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \end{pmatrix}$$

The values of n that we calculate over are those that produce an overlap between the input sequence and the kernel. This will be the range  $n \in [-2,2]$ , in this case. As it is written mathematically in the convolution equation, we theoretically sum from  $k \in [-\infty,\infty]$ . However as with the choice of n, we only need to evaluate at values k over where the input sequence and the reversed/shifted kernel will have non-zero values when they multiply. For this example we just need to use the range  $k \in [-3,3]$ .

#### Step 2:

For n=-2 we calculate the reversed/shifted kernel at each k

$$h\left(n-k\right) = h\left(-3-k\right)$$

At each value of  $k \in [-3,3]$ 

$$k = -3$$
,  $h(-2 - k) = h(-2 - (-3)) = h(1) = -2$ 

$$k = -2$$
,  $h(-2 - k) = h(-2 - (-2)) = h(0) = 2$ 

$$k = -1$$
,  $h(-2 - k) = h(-2 - (-1)) = h(-1) = 0$ 

$$k = 0$$
,  $h(-2 - k) = h(-2 - (0)) = h(-2) = 0$ 

All values of  $h\left(n-k\right)$  will be zero after this point (you can check).

We can tabulate the values of k, the sequence  $x\left[n\right]$  and the position of the reversed/shifted kernel based on the  $h\left[-2-k\right]$  values we have determined.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \ x\,[k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \ h\,[-2-k] & -2 & 2 & 0 & 0 & 0 & 0 \end{pmatrix}$$

The convolution result (x\*h) for n=-2 is calculated as the elementwise multiplication and sum of the sequence  $x\left[k\right]$  and reversed/shifted kernel  $h\left[-2-k\right]$ 

$$(x*h)\left[-2\right] = \sum_{k=-3}^{k=3} x\left[k\right] h\left[-2-k\right] = 0\cdot -2 + 0\cdot 2 - 2\cdot 0 - 2\cdot 0 - 1\cdot 0 = 0$$

Step 3:

For n=-1 we calculate the reversed/shifted kernel at each k

$$h\left(n-k\right) = h\left(-1-k\right)$$

At each value of  $k \in [-3,3]$ 

$$k = -3$$
,  $h(-1 - k) = h(-1 - (-3)) = h(2) = 0$ 

$$k = -2$$
,  $h(-1 - k) = h(-1 - (-2)) = h(1) = -2$ 

$$k = -1$$
,  $h(-1 - k) = h(-1 - (-1)) = h(0) = 2$ 

$$k = 0$$
,  $h(-1 - k) = h(-1 - (0)) = h(-1) = 0$ 

$$k = 1$$
,  $h(-1 - k) = h(-1 - (1)) = h(-2) = 0$ 

$$k = 2$$
,  $h(-1 - k) = h(-1 - (2)) = h(-3) = 0$ 

$$k = 3$$
,  $h(-1 - k) = h(-1 - (3)) = h(-4) = 0$ 

We can tabulate the values of k, the sequence  $x\left[n\right]$  and the position of the reversed/shifted kernel based on the  $h\left[-1-k\right]$  values we have determined.

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \\ x [k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \\ h [-2-k] & -2 & 2 & 0 & 0 & 0 & 0 & 0 \\ h [-1-k] & 0 & -2 & 2 & 0 & 0 & 0 & 0 \end{pmatrix}$$

The convolution result (x\*h) for n=-1 is calculated as the elementwise multiplication and sum of the sequence  $x\left[k\right]$  and reversed/shifted kernel  $h\left[-1-k\right]$ 

$$(x*h)[-1] = \sum_{k=-3}^{k=3} x[n] h[-1-k] = 0 \cdot 0 + 0 \cdot -2 - 2 \cdot 2 - 2 \cdot 0 - 1 \cdot 0 = -4$$

Step 3,4...

We repeat this procedure for each of the other values of n and show the results graphically below:

$$\begin{pmatrix} k & -3 & -2 & -1 & 0 & 1 & 2 & 3 \ x\,[k] & 0 & 0 & -2 & -2 & -1 & 0 & 0 \ h\,[-2-k] & -2 & 2 & 0 & 0 & 0 & 0 & 0 \ h\,[-1-k] & 0 & -2 & 2 & 0 & 0 & 0 & 0 \ h\,[-k] & 0 & 0 & -2 & 2 & 0 & 0 & 0 \ h\,[1-k] & 0 & 0 & 0 & -2 & 2 & 0 & 0 \ h\,[2-k] & 0 & 0 & 0 & 0 & -2 & 2 & 0 \ \end{pmatrix} \ \begin{pmatrix} (x*h)\,[n] & 0 & -4 & 0 & 2 & 2 \ \end{pmatrix}$$

 $\bullet$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 0 & -4 & 0 & 2 & 2 \end{pmatrix}$ 

100% of points

No specific feedback