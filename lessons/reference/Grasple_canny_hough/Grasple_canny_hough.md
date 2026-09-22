<u>View subject</u> | <u>Edit subject</u>

## Canny Hough Algorithms

Slides

### **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 106139

QUESTION

What is the primary purpose of applying smoothing with a Gaussian Kernel in the Canny edge detection algorithm? (Select the correct option from the choices below)

Select one answer. Upon selection it will submit your answer immediately.

### ▼ Feedback for answer: 1

Incorrect! Smoothing will tend to blur edges, so we need to be careful not to use too large a gaussian kernel, what else will the kernel remove?

✓ Feedback for answer: 2 100% of points

Correct! We want to remove any noise to prevent spurious edges from being detected.

0% of points

0% of points

Incorrect! If we sharpen the image we achieve the opposite result to smoothing which will be to enhance spurious small edges and noise.

QUESTION

Which of the following best describes the role of Sobel filters in the Canny algorithm?

| ○ To smooth the Image                                                                                                                        | ×              |
|----------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| To detect edges by calculating gradient magnitude and direction                                                                              | n 🗸            |
| ○ To apply thresholding                                                                                                                      | ×              |
| To suppress non-maximal pixels                                                                                                               | ×              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                    |                |
|                                                                                                                                              | 0% of points   |
| Incorrect. Sobel filters are used for horizontal or vertical e detection, not smoothing                                                      | dge            |
| ✓ Feedback for answer: 2                                                                                                                     | 100% of points |
| Correct. Sobel filters compute the gradient. The magnitud direction of the gradient can be calculated from this which for identifying edges. |                |
| ➤ Feedback for answer: 3                                                                                                                     | 0% of points   |
| Incorrect. Thresholding is applied later in the Canny algori                                                                                 | ithm.          |
|                                                                                                                                              | 0% of points   |
|                                                                                                                                              |                |

Incorrect. Non-maximum suppression is a separate step that follows gradient calculation

QUESTION

# What is the main objective of non-maximum suppression in the Canny edge detection process?

|   | ○ To blur the image                                                                                            | ×              |
|---|----------------------------------------------------------------------------------------------------------------|----------------|
|   |                                                                                                                |                |
|   | To enhance the gradient magnitude                                                                              | ×              |
|   |                                                                                                                |                |
|   | To thin out the edges by keeping only the local maxima                                                         | ✓              |
|   |                                                                                                                |                |
|   | To apply double thresholding                                                                                   | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                      |                |
| < | Feedback for answer: 1                                                                                         | 0% of points   |
|   | Incorrect. Non-maximum suppression does not blur the image; refines edge detection                             | it             |
| < | Feedback for answer: 2                                                                                         | 0% of points   |
|   | Incorrect. This step does not enhance gradients but selects the magnitude edge pixels.                         | largest        |
| 2 | Feedback for answer: 3                                                                                         | 100% of points |
|   | Correct. Non-maximum suppression removes non-maximum gr<br>magnitude values, keeping only the strongest edges. | adient         |
| < | Feedback for answer: 4                                                                                         | 0% of points   |
|   |                                                                                                                |                |

Incorrect. Double thresholding is applied in the hysteresis thresholding step.

QUESTION

# What is the purpose of hysteresis thresholding in the Canny algorithm?

|          | ○ To smooth the image                                                                                                      | ×              |
|----------|----------------------------------------------------------------------------------------------------------------------------|----------------|
|          | To connect edge segments by using high and low thresholds                                                                  | <b>V</b>       |
|          | To calculate gradient direction                                                                                            | ×              |
|          | To apply non-maximum suppression                                                                                           | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                  |                |
| ×        | Feedback for answer: 1                                                                                                     | 0% of points   |
|          | Incorrect. Smoothing is done at the beginning of the process wit gaussian kernel.                                          | th a           |
| <b>√</b> | Feedback for answer: 2                                                                                                     | 100% of points |
|          | Correct. Hysteresis thresholding uses two thresholds to identify and weak edges, connecting them to form continuous edges. | strong         |
| ×        | Feedback for answer: 3                                                                                                     | 0% of points   |
|          | Incorrect. Gradient direction is calculated earlier using Sobel filt                                                       | ers.           |
| ×        | Feedback for answer: 4                                                                                                     | 0% of points   |
|          | Incorrect. Non-maximum suppression is a separate step that prhysteresis thresholding.                                      | ecedes         |

#### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 106178

QUESTION

A canny edge detector has been programmed with the hysterisis thresholds  $T_{\rm max}=142$  and  $T_{\rm min}=92$ . Determine if the pixel P=109 will be accepted or rejected as part of an edge given it is connected to pixel A=180. Give your answer "T" (True or accepted) "F" (False or rejected) in the box below.

#### Detailed solution

The correct answer is T

We are given the  $T_{\rm max}=142$  and  $T_{\rm min}=92$ . The pixel P=109 is connected to pixel A=180.

As the pixel A is greater than the  $T_{\rm max}$  it is part of a valid edge. If P is less than  $T_{\rm max}$  but greater than  $T_{\rm min}$  then it is accepted as part of the same edge as A. However if  $P < T_{\rm min}$  it is rejected.

#### ■ Question at wrong answer

In the Canny algorithm we use thresholding to determine if a given pixel is part of an edge. With a single threshold all connected pixels with values greater than the threshold form an edge. All pixels with values less than the threshold are not part of the same edge (even if connected). As seen in the image below this results in broken lines (due to pixels that have a magnitude that drops below the threshold)

Hystersis thresholding is used in the canny edge detection algorithm to produce more continuous edges. Using two thresholds helps produce more continuous (or less broken) edges than by using a single threshold.

The thresholding operates by simple rules based on connectedness. We are given the upper threshold  $T_{\rm max}$  and lower threshold  $T_{\rm min}$ .

Assuming a pixel P is connected to pixel A. The pixel A is greater than the  $T_{\rm max}$  and is part of a valid edge. If P is less than  $T_{\rm max}$  but greater than  $T_{\rm min}$  then it is accepted as part of the same edge as A. However was  $P < T_{\rm min}$  it is rejected.

We also note that if the pixel P is not connected to A and it is less than  $T_{\max}$  it is rejected.

If the pixel P is not connected to A and it is greater than  $T_{\max}$  it is accepted as a new edge (independent of the edge containing A).

 $\boldsymbol{\varphi}$  Feedback when answer is algebraically Equivalent to T

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 106187

QUESTION

A set of points corresponding to an edge is applied to the hough transform. Given (in the tables) the set of parameter values  $(r,\theta)$  obtained from each of the points, determine the most likely line that connects the points. (Note if this question shows negative values for r, please button above for new random parameter selection)

$$\begin{pmatrix} \theta & r \\ 15 & 0.81 \\ 30 & 0.90 \\ 45 & 0.92 \\ 60 & 0.88 \\ 75 & 0.78 \end{pmatrix} \text{ and } \begin{pmatrix} \theta & r \\ 15 & 0.94 \\ 30 & 0.90 \\ 45 & 0.79 \\ 60 & 0.63 \\ 75 & 0.43 \end{pmatrix}$$

Give your answer  $\theta$  value for the best fitting line in degrees in the box below.

✓ Detailed solution

We note that for each candidate line at a given angle (red, green and blue) drawn through A and B that the calculated distance r is different. However for the blue line case the distance r is the same. Where the distances match we can be confident the same line passes through both points.

#### ■ Question at wrong answer

We use the Hough transform to find the best fit lines that correspond to edges. The lines can be represented by a slope m and an intercept value c, however this representation has the problem that the slope goes to infinity if the line is vertical so it is not used. Instead a polar representation of the line is used, where each line is represented by an angle  $\theta$  and distance from the origin r.

As seen in the figure below we show two points A and B that correspond to an edge. There are many candidate lines that we can draw at different angles through each point so we need to determine which line is the correct one that connects the points.

We can obtain the perpendicular distance r from the origin for each line charaterised by its polar coordinate angle  $\theta$ . This is obtained from the equation  $r = x \cdot \cos(\theta) + y \cdot \sin(\theta)$  (We note that the angle  $\theta$  is the angle of the polar representation of the line, not the drawn angle of the line which is  $\alpha = 90 - \theta$ )

We note that for each candidate line (red, green and blue) with the same polar coordinate angle drawn through A and B that the calculated distance r is different. However for the blue line case the distance r is the same. In the Hough algorithm by identifying where r is the same for many edge points a line can be identified.

| Feedback when answer is algebraically Equivalent to 30.00 | 100% of points |
|-----------------------------------------------------------|----------------|
| No specific feedback                                      |                |

<u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 118450

**QUESTION** 

The hough transform has been applied to a series of points from an image. The resulting sine waves in parameter  $(\rho,\theta)$  space are produced for the candidate lines through the points.

Determine the slope m of the primary line in the image.

| $\begin{array}{c c} \hline \\ \hline \\ \hline \\ \hline \\ \hline \\ \hline \\ \hline \\ \hline \\ \hline \\ \hline $ |
|------------------------------------------------------------------------------------------------------------------------|
| -3.48                                                                                                                  |
| Check my answer                                                                                                        |

✓ Detailed solution

In order to answer this question we need to have an understanding of the Hough Algorithm (Polar Form):

A point in cartesian image space  $(x_i,y_i)$  may be part of a line (which must be determined by the algorithm). A series of candidate lines can be drawn through the point. Figure (a) shows a possible candidate line through  $(x_i,y_i)$ . The blue line in figure (a) has a representation in polar coordinates  $(\rho,\theta)$ . Where  $\rho$  is defined as the distance from the point to the origin and  $\theta$  defines the angle with the origin, (x,y) is the cartesian coordinate point on the line where the polar line from the origin intersects the candidate line perpendicularly and c is where the candidate line intercepts the y-axis.

Any candidate line through  $(x_i,y_i)$  can be described by the standard equation of a line y=mx+c. We can easily write the intercept coordinates x,y in terms of  $(\rho,\theta)$   $x=\rho\cos(\theta)$  and  $y=\rho\sin(\theta)$ . We then substitute these expressions into the equation of the line,  $\rho\sin(\theta)=m\cdot\rho\cos(\theta)+c$ , solving for  $\rho$  we get  $\rho=\frac{c}{\sin(\theta)-m\cos(\theta)}$ 

Since the candidate line is perpendicular to the polar line from the origin, it will have a slope of  $m=-\frac{1}{m_p}$ , where  $m_p=\frac{\sin{(\theta)}}{\cos{(\theta)}}$  is the slope of the polar line. Therefore  $m=-\frac{1}{m_p}=-\frac{\cos{(\theta)}}{\sin{(\theta)}}$ 

Substituting for m the expression for  $\rho$  becomes

$$\rho = \frac{c}{\sin(\theta) + \frac{\cos(\theta)}{\sin(\theta)}\cos(\theta)} = \frac{c}{\frac{\sin^2(\theta)}{\sin(\theta)} + \frac{\cos^2(\theta)}{\sin(\theta)}} = \frac{c\sin(\theta)}{1}$$

(We note that we could also have obtained this equation more simply by inspecting the figure (a) and noting the angle the candidate line makes with the y-axis is also  $\theta$  and the sine of this angle will be  $\sin(\theta)=\frac{\rho}{c}$ )

We now wish to eliminate the intercept value c. We re-use the equation of the line y=mx+c, but this time substitute the known values  $x_i,y_i$  of the point in cartesian image space that are also on the candidate line (we re-use the slope determined previously). Therefore  $y_i=-\frac{\cos(\theta)}{\sin{(\theta)}}x_i+c$  and rearranging for  $c=y_i+x_i\frac{\cos{(\theta)}}{\sin{(\theta)}}$ .

We then substitute the expression for c into  $\rho=c\sin(\theta)$ . This gives us the expression  $\rho=y_i\sin(\theta)+x_i\cos(\theta)$  that allows us to compute the length of the polar vector to the candidate line through any point in the image  $(x_i,y_i)$  at any angle  $\theta$ .

Using the equation, we plot the value of  $\rho$  for a series of  $\theta$  values (from 0 to  $\pi$  radians) at each  $(x_i,y_i)$  point in the image. Each  $\theta$  value corresponds to a candidate line at a different angle passing though each point. As shown in the graph (repeated from the question) below, for each point we obtain a sinusoidal shape in parameter space. We can see that the curves intersect at a particular values of  $(\rho,\theta)$  corresponding to the equations of the candidate lines connecting the points. Obviously each pair of points can form a line so there will several intersections corresponding to these supurious lines. We are looking for the most likely edge in the image which should connect several points, we see that there is an intersection between 3 lines at  $(\rho,\theta)=(0.13,0.28)$ .

The intercept points  $(\rho,\theta)$  are the polar representation of the line. We can then convert this into a cartesian equation of the line (corresponding to edge points)

with a slope and intercept (m,c).

The slope can be calculated using the equation from earlier

$$m = -\frac{1}{m_p} = -\frac{\cos(\theta)}{\sin(\theta)} = -\frac{\cos(0.28)}{\sin(0.28)} = -3.48$$
.

#### Question at wrong answer

In order to answer this question we need to have an understanding of the Hough Algorithm (Polar Form):

A point in cartesian image space  $(x_i,y_i)$  may be part of a line (which must be determined by the algorithm). A series of candidate lines can be drawn through the point. Figure (a) shows a possible candidate line through  $(x_i,y_i)$ . The blue line in figure (a) has a representation in polar coordinates  $(\rho,\theta)$ . Where  $\rho$  defined the distance from the point to the origin and  $\theta$  defines the angle with the origin, (x,y) is the cartesian coordinate point on the line where the polar line from the origin intersects the candidate line perpendicularly and c is where the candidate line intercepts the y-axis.

Any candidate line through  $(x_i,y_i)$  can be described by the standard equation of a line y=mx+c. We can easily write the intercept coordinates x,y in terms of  $(\rho,\theta)$   $x=\rho\cos(\theta)$  and  $y=\rho\sin(\theta)$ . We then substitute these expressions into the equation of the line,  $\rho\sin(\theta)=m\cdot\rho\cos(\theta)+c$ , solving for  $\rho$  we get  $\rho=\frac{c}{\sin(\theta)-m\cos(\theta)}$ 

Since the candidate line is perpendicular to the polar line from the origin, it will have a slope of  $m=-\frac{1}{m_p}$ , where  $m_p=\frac{\sin{(\theta)}}{\cos{(\theta)}}$  is the slope of the polar line. Therefore  $m=-\frac{1}{m_p}=-\frac{\cos{(\theta)}}{\sin{(\theta)}}$ 

Substituting for m the expression for  $\rho$  becomes

$$\rho = \frac{c}{\sin(\theta) + \frac{\cos(\theta)}{\sin(\theta)}\cos(\theta)} = \frac{c}{\frac{\sin^2(\theta)}{\sin(\theta)} + \frac{\cos^2(\theta)}{\sin(\theta)}} = \frac{c\sin(\theta)}{1}$$

(We note that we could also have obtained this equation more simply by inspecting the figure (a) and noting the angle the candidate line makes with the y-axis is also  $\theta$  and the sine of this angle will be  $\sin(\theta) = \frac{\rho}{c}$ )

We now wish to eliminate the intercept value c. We re-use the equation of the line y=mx+c, but this time substitute the known values  $x_i,y_i$  of the point in cartesian image space that are also on the candidate line (we re-use the slope determined previously). Therefore  $y_i=-\frac{\cos(\theta)}{\sin{(\theta)}}x_i+c$  and rearranging for  $c=y_i+x_i\frac{\cos{(\theta)}}{\sin{(\theta)}}$ .

We then substitute the expression for c into  $\rho=c\sin(\theta)$ . This gives us the expression  $\rho=y_i\sin(\theta)+x_i\cos(\theta)$  that allows us to compute the length of the polar vector to the candidate line through any point in the image  $(x_i,y_i)$  at any angle  $\theta$ .

Using the equation, we plot the value of  $\rho$  for a series of  $\theta$  values (from 0 to  $\pi$  radians) at each  $(x_i,y_i)$  point in the image. Each  $\theta$  value corresponds to a candidate line at a different angle passing though each point. As shown in the graph (repeated from the question) below, for each point we obtain a sinusoidal shape in parameter space. We can see that the curves intersect at a particular values of  $(\rho,\theta)$  corresponding to the equations of the candidate lines connecting the points. Obviously each pair of points can form a line so there will several intersections corresponding to these supurious lines. We are looking for the most likely edge in the image which should connect several points, we see that there is an intersection between 3 lines at  $(\rho,\theta)=(0.13,0.28)$ .

The intercept points  $(\rho,\theta)$  are the polar representation of the line. We can then convert this into a cartesian equation of the line (corresponding to edge points) with a slope and intercept (m,c).

The slope can be calculated using the equation from earlier

$$m = -\frac{1}{m_n} = -\frac{\cos(\theta)}{\sin(\theta)}$$

| Feedback when answer is equal to -3.48                         | 100% of points |
|----------------------------------------------------------------|----------------|
| ✓ Feedback when answer is equal to -3.48  No specific feedback | 100% of points |

QUESTION

Determine the intercept  $\it c$  of the primary line in the image.

0.47

Check my answer

✓ Detailed solution

We can also calculate the intercept of the line from the relationship  $\rho=c\,\sin(\theta)$ , therefore  $c=\frac{\rho}{\sin(\theta)}=\frac{0.13}{\sin(0.28)}=0.47$ .

We can also calculate the intercept of the line from the relationship  $\rho = c \, \sin(\theta)$ 

✓ Feedback when answer is equal to 0.47

100% of points

No specific feedback