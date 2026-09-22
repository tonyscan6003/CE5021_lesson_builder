<u>View subject</u> | <u>Edit subject</u>

# **Edge Detection**

Slides

# **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

### Exercise 106042

**QUESTION** 

What is the general principle used to find edges in images? (Select the correct answer from the options below)

| 0 | Edges can be detected by finding a high rate of change in pixel intensity. This may be achieved with derivative based detectors. | <b>▽</b> |
|---|----------------------------------------------------------------------------------------------------------------------------------|----------|
|   |                                                                                                                                  |          |
| 0 | Edges can be detected by finding areas of uniform colour.                                                                        | ×        |
|   |                                                                                                                                  |          |
|   | Edges can be detected by averaging pixel values.                                                                                 | ×        |
|   |                                                                                                                                  |          |
| 0 | Edges can be detected by increasing the brightness of the image.                                                                 | ×        |

Select one answer. Upon selection it will submit your answer immediately.

✓ Feedback for answer: 1
 Correct!
 ✓ Feedback for answer: 2
 Incorrect. Uniform color areas do not indicate edges, edges are where there is a change in pixel intensity.

0% of points

Incorrect. Averaging pixel values smooths the image and reduces edge visibility

0% of points

Incorrect. Increasing brightness does not help in detecting edges, it only makes the image brighter and may eventually saturate the image removing edges.

QUESTION

What will happen to a noisy image if an edge detector is applied? (Select the correct answer from the options below)

|   | The noise will be reduced, making the edges clearer.                                                                              | X              |
|---|-----------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | The noise will disturb the edge detection, creating false edges.                                                                  | ✓              |
|   | The image will become completely black.                                                                                           | ×              |
|   | The image will become completely white.                                                                                           | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                         |                |
| × | Feedback for answer: 1                                                                                                            | 0% of points   |
|   | Incorrect, Edge detectors are sensitive to noise and do not redu                                                                  | ce it          |
| ~ | Feedback for answer: 2                                                                                                            | 100% of points |
|   | Correct answer, edge detection is sensitive to noise.                                                                             |                |
| × | Feedback for answer: 3                                                                                                            | 0% of points   |
|   | Incorrect, Edge detection does not turn the image completely bl will tend to find both real edges and spurious edges due to the i |                |
| × | Feedback for answer: 4                                                                                                            | 0% of points   |
|   | Incorrect, Edge detection does not turn the image completely w will tend to find both real edges and spurious edges due to the    |                |

QUESTION

What direction of edges will an X-direction kernel recover? (Select the correct answer from the options below)

|   | ○ Horizontal edges                                                                                                                                 | ×              |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | ○ Vertical Edges                                                                                                                                   | <b>✓</b>       |
|   | ○ Diagonal Edges                                                                                                                                   | ×              |
|   | ○ Circular Edges                                                                                                                                   | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                                          |                |
| × | Feedback for answer: 1                                                                                                                             | 0% of points   |
|   | Incorrect. An X-direction kernel detects vertical edges, not horiz                                                                                 | ontal.         |
| V | Feedback for answer: 2                                                                                                                             | 100% of points |
|   | Correct! Vertical edges are detected.                                                                                                              |                |
| × | Feedback for answer: 3                                                                                                                             | 0% of points   |
|   | Incorrect. Diagonal edges are not specifically targeted by an X-direction kernel. The kernel may respond to the edge.                              |                |
| × | Feedback for answer: 4                                                                                                                             | 0% of points   |
|   | Incorrect. Circular edges are not specifically targeted by an X-direction kernel. The kernel may respond to the edge (vertical portion of circle). |                |

QUESTION

# Name 3 types of edge detectors. (Select the correct answer from the options below)

|          | O Prewitt, Sobel, Laplacian                                                                                                | <b>✓</b>       |
|----------|----------------------------------------------------------------------------------------------------------------------------|----------------|
|          | Gaussian, Median Filter, Bilateral                                                                                         | ×              |
|          | O Fourier, Wavelet, Radon                                                                                                  | ×              |
|          | Canny, Prewitt, Sobel                                                                                                      | <b>✓</b>       |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                  |                |
| ~        | Feedback for answer: 1                                                                                                     | 100% of points |
|          | Correct! The Prewitt and Sobel kernels detect edges, the Laplaci sensitive to circular shapes but also finds edges.        | ian is         |
| ×        | Feedback for answer: 2                                                                                                     | 0% of points   |
|          | Incorrect. These are smoothing and noise reduction techniques edge detectors.                                              | , not          |
| ×        | Feedback for answer: 3                                                                                                     | 0% of points   |
|          | These are transform techniques, not specifically edge detectors                                                            |                |
| <b>V</b> | Feedback for answer: 4                                                                                                     | 100% of points |
|          | Correct! The Prewitt and Sobel kernels detect edges, the canny detector uses a Sobel kernel and then adds non-maximum supr |                |

and hystersis thresholding to find single edges.

QUESTION

 $[\mathsf{X}]$ 

What is the principle difference between the Laplacian kernel and the other kernels that can detect edges? (Select the correct answer from the options below)

| O It is a first order kernel                                                                                                             | ×          |  |
|------------------------------------------------------------------------------------------------------------------------------------------|------------|--|
|                                                                                                                                          |            |  |
| ○ It is a second order kernel                                                                                                            | ✓          |  |
|                                                                                                                                          |            |  |
| ○ It is a smoothing kernel                                                                                                               | ×          |  |
|                                                                                                                                          |            |  |
| It is a colour enhancing kernel                                                                                                          | ×          |  |
| Select one answer. Upon selection it will submit your answer immediately.                                                                |            |  |
| Feedback for answer: 1                                                                                                                   | 0% of poin |  |
| Incorrect. The Laplacian is a 2nd order kernel, not 1st order kernel.  Most edge detectors e.g. Prewitt and Sobel are 1st order kernels. |            |  |

sum of the x and y 2nd order partial derivatives of the Gaussian.

Correct! The lapacian is a 2nd order kernel (it is formed from the the

✓ Feedback for answer: 2

100% of points

| Executive The Laplacian is used for circular region/odge detection

Incorrect. The Laplacian is used for circular region/edge detection, not smoothing.

Incorrect. The Laplacian is not used for color enhancement.

0% of points

QUESTION

In the Gaussian Derivative kernel, what is the effect of the  $\sigma$  parameter? (Select the correct answer from the options below)

|   | 0          | It controls the brightness of the image.                                           | ×            |
|---|------------|------------------------------------------------------------------------------------|--------------|
|   |            | It controls the contrast of the image.                                             | ×            |
|   | 0          | It controls the scale of the Gaussian, affecting the level of detail in the image. | ✓            |
|   | $\bigcirc$ | It controls the color saturation of the image.                                     | ×            |
|   | Select     | one answer. Upon selection it will submit your answer immediately.                 |              |
| × | Feedba     | ck for answer: 1                                                                   | 0% of points |
|   | Inco       | rrect. The $\sigma$ parameter does not affect brightness.                          |              |
| × | Feedba     | ck for answer: 2                                                                   | 0% of points |
|   | Inco       | orrect. The σ parameter does not affect contrast.                                  |              |

| ✓ Feedback for answer: 3                                            | 100% of points |
|---------------------------------------------------------------------|----------------|
| Correct!                                                            |                |
| 区 Eeedback for answer: 4                                            | 0% of points   |
| Incorrect. The $\sigma$ parameter does not affect color saturation. |                |

### Exercise 106060

QUESTION

Calculating Gradient and Direction of Image Region

Given the 3 x 3 Image patch

$$I = \begin{pmatrix} -1.77 & -0.82 & 0.13 \ -0.95 & 3.7 & 0.95 \ -0.13 & 0.82 & 1.77 \end{pmatrix}$$
 calculate the magnitude of

the gradient of the image patch. Given the equation  $M=\sqrt{G_x^2+G_y^2}$  where the gradients  $G_x,G_y$  can be estimated by using the Sobel kernels. (Give the numeric value you obtained for M in the box below)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| 10.04                                                  |

## Check my answer

✓ Detailed solution

Step 1: We will use the Sobel Kernels to obtain the gradients  $G_x,G_y$  of the image patch. The x-direction  $K_x=\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}$  and y-direction

$$K_y = \begin{pmatrix} 1 & 2 & 1 \ 0 & 0 & 0 \ -1 & -2 & -1 \end{pmatrix}$$
 Sobel kernels can be convolved with the image

to find the gradient. In this case as the image patch and kernel are the

same size  $G_x$ ,  $G_y$  will just be scalar values. (Don't forget to flip the kernels before performing convolution)

Obtaining the result for  $G_x$ , $G_y$ 

$$G_x = I \circledast K_x$$

$$G_x = -1 \cdot -1.77 - 2 \cdot -0.95 - 1 \cdot -0.13 + 0.13 + 2 \cdot 0.95 + 1.77$$

$$G_x = 7.60$$

$$G_y = I \circledast K_y$$

$$G_y = -1 \cdot -1.77 - 2 \cdot -0.82 - 1 \cdot 0.13 - 0.13 + 2 \cdot 0.82 + 1.77$$

$$G_y = 6.56$$

Step 2: We use the calculated  $G_x$ ,  $G_y$  values to determine the magnitude M and direction  $\theta$  from the equations we have been given.

$$M=\sqrt{G_x^2+G_y^2}=\sqrt{7.60^2+6.56^2}=10.04$$

■ Question at wrong answer

Hint: Use the Sobel Kernels to obtain the gradients  $G_x,G_y$  of the image patch. The x-direction kernel  $K_x=\begin{pmatrix}1&0&-1\\2&0&-2\\1&0&-1\end{pmatrix}$  and y-direction

kernels  $K_y=\begin{pmatrix}1&2&1\0&0&0\-1&-2&-1\end{pmatrix}$  . (Don't forget to flip these kernels

before convolution)

☑ Feedback when answer is equal to 10.04

100% of points

No specific feedback

QUESTION

Calculate the angle of the gradient of the image patch. Given the equation  $\theta = \tan^{-1}\left(\frac{G_y}{G_x}\right)$  where the gradients  $G_x,G_y$  have been estimated by using the Sobel kernels. (Give the numeric value (in degrees) you obtained for  $\theta$  in the box below)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |  |
|--------------------------------------------------------|--|
| 40.68                                                  |  |
|                                                        |  |

### Check my answer

✓ Detailed solution

Correct! The angle (in degrees) is simply calculated using the given equation and the values obtained previously.

$$\theta = \tan^{-1} \left( \frac{G_y}{G_x} \right) = \tan^{-1} \left( \frac{6.56}{7.60} \right) = 40.68$$

Question at wrong answer

Hint: 
$$\theta=\tan^{-1}\left(\frac{G_y}{G_x}\right)=\tan^{-1}\left(\frac{6.56}{7.60}\right)$$

✓ Feedback when answer is equal to 40.68

100% of points

No specific feedback