<u>View subject</u> | <u>Edit subject</u>

## Object Detection

Slides

## **Exercises**

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 105760

**QUESTION** 

Calculate Intersection over union.

Calculate the Intersection over Union of the two bounding boxes defined (by the coordinate sets) below. The Predicted box  $B_p$  has coordinates at the top left  $(x_{b1},y_{b1}=44,39)$  and bottom right  $(x_{b2},y_{b2}=72,60)$ . The ground truth box  $B_{gt}$  has coordinates at the top left  $(x_{gt1},y_{gt2}=38,45)$  and bottom right  $(x_{gt2},y_{gt2}=54,88)$ .

Give your answer in the box below to 2 decimal places of accuracy.

Note that depending on the given coordinates, the overlap of the boxes may look like any one of the configurations below. (It is recommended to sketch a diagram for the set of coordinates given in the question)

✓ Detailed solution

The intersection over union is calculated as the intersection area divided by the union area of the two boxes.  $IoU = \frac{Area\left(B_p \cap B_{gt}\right)}{Area\left(B_p \cup B_{gt}\right)}$ 

Step1: Firstly we obtain the intersection area  $Area\left(B_p\cap B_{gt}\right)$  of the two boxes. As we can see in the image from the question the bounding boxes may overlap with different orientations, and the exact coordinates that define the overlap area are different in each case. (We can determine these by inspection or) we can use simple rules to determine the coordinates of the intersection area.

The  $x_{I1},y_{I1}$  coordinates of the top left of the intersection area are given as the maximum of the bounding box or ground truth coordinates:

$$x_{I1} = Max(x_{b1}, x_{at1}) = Max(44,38) = 44$$

$$y_{I1} = Max\left(y_{b1}, y_{gt1}\right) = Max\left(39, 45\right) = 45$$

Similarly the  $x_{I2},y_{I2}$  coordinates of the bottom right of the intersection area are given as the minimum of the bounding box or ground truth coordinates:

$$x_{I2} = Min\left(x_{b2}, x_{qt2}\right) = Min\left(72, 54\right) = 54$$

$$y_{I2} = Min\left(y_{b1}, y_{gt1}\right) = Min\left(60, 88\right) = 60$$

The area of the intersection rectangle is given by

$$Area\left(B_{p}\cap B_{gt}\right)=\left(x_{I2}-x_{I1}\right)\cdot\left(y_{I2}-y_{I1}\right)$$

$$Area\left(B_{v}\cap B_{ot}\right)=(54-44)\cdot(60-45)=150$$

Example of finding  $x_{I1},y_{I1}$  and  $x_{I2},y_{I2}$  (your case may be different)

Step 2: Secondly we obtain the union area between the two boxes. As you can see in the image below this is an irregular area, but can be easily calculated as the sum of the area of the two boxes minus the intersection area.

Lets first calculate the areas of the boxes.

$$Area(B_p) = (72 - 44) \cdot (60 - 39) = 588$$

$$Area(B_{gt}) = (54 - 38) \cdot (88 - 45) = 688$$

$$Area\left(B_{p}\cup B_{gt}\right)=Area(B_{p})+Area(B_{gt})-Area\left(B_{p}\cap B_{gt}\right)$$

$$Area\left(B_{p}\cup B_{qt}\right)=1126$$

Step 3: Now we can calculate the Intersection over Union using the areas we have just calculated.

$$IoU = \frac{Area\left(B_p \cap B_{gt}\right)}{Area\left(B_p \cup B_{gt}\right)} = \frac{150}{1126} = 0.13$$

### 

The intersection over union is calculated as the intersection area divided by the union area of the two boxes.  $IoU = \frac{Area\left(B_p \cap B_{gt}\right)}{Area\left(B_p \cup B_{gt}\right)}$ 

The Intersection area is calculated as the overlapping area between the two boxes.

The union area is the total area occupied by the two boxes. It is easily calculated as the sum of the areas of the two boxes minus the intersection area.

 $\boxdot$  Feedback when answer is equal to 0.13

100% of points

No specific feedback

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106314

**QUESTION** 

Obtain the integral image of the image patch

$$I = \begin{pmatrix} 1 & 2 & 3 & 1 & 2 \ 0 & 1 & 4 & 4 & 5 \ 1 & 4 & 3 & 2 & 5 \ 4 & 4 & 4 & 4 & 4 \ 4 & 4 & 5 & 4 & 2 \end{pmatrix}$$
 , and find the sum of the pixels

in the centre 3 x 3 region using values from the integral image. (For the sum calculation write your answer in the form A-B-C+D where A,B,C,D are obtained from the integral image)

#### Detailed solution

We obtain the integral image from the original Image I. We first sum (starting at the top) along the columns creating an intermediate image  $I_1$ . (We note that the first value at the start of each column is the same value as the original image.)

$$I_1 = \begin{pmatrix} 1 & 2 & 3 & 1 & 2 \ 1 & 3 & 7 & 5 & 7 \ 2 & 7 & 10 & 7 & 12 \ 6 & 11 & 14 & 11 & 16 \ 10 & 15 & 19 & 15 & 18 \end{pmatrix}$$

We then sum along the rows of the intermediate image to produce the final integral image  $I_I$ .

$$I_I = \begin{pmatrix} 1 & 3 & 6 & 7 & 9 \ 1 & 4 & 11 & 16 & 23 \ 2 & 9 & 19 & 26 & 38 \ 6 & 17 & 31 & 42 & 58 \ 10 & 25 & 44 & 59 & 77 \end{pmatrix}$$

It is then possible to obtain the sum by summing the 4 values A,B,C,D with locations as indicated on the figure below:

| D |  | В |  |
|---|--|---|--|
|   |  |   |  |
|   |  |   |  |
| С |  | А |  |
|   |  |   |  |

The sum is obtained as A-B-C+D=42-7-6+1=30. (This answer can be checked by manually summing the 9 values in the centre 3 x3 region from the original image, we see that also equals 30 )

Note: Even with this small example we can see it is more computationally efficient to sum 4 values rather than 9. The computational cost of producing the integral image will be offset by the need to perform the sums of lots of different shaped regions in the Viola Jones algorithm.

### □ Question at wrong answer

As shown in the example below, we form the integral image by first summing the columns of the original image creating an intermediate image.

We then sum the rows of the intermediate images to produce the final integral image.

We can then find the sum of the central 3 x 3 region with the computation A-B-C+D Where values A,B,C,D are value from the integral image as indicated in the figure below.

| D |  | В |  |
|---|--|---|--|
|   |  |   |  |
|   |  |   |  |
| С |  | А |  |
|   |  |   |  |

Note that your answer computed from the integral image should be the same as the sum of the 9 values in the  $3 \times 3$  region of the original image.

 $\boxed{\checkmark}$  Feedback when answer is algebraically Equivalent to  $\ 42-7-6+1$ 

100% of points

No specific feedback

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106315

QUESTION

Determine the Hausdorff directed distance  $h\left(A,B\right)=\max_{a\in A}\min_{b\in B}\parallel a-b\parallel$  between the point sets  $A=\left(\left(4.5,4.3\right),\left(3.4,3.9\right),\left(3.2,2.2\right),\left(\left(4.5,1.7\right),\left(3.9,2.2\right)\right)\right)$  and

$$B = ((4.6,5.0), (3.3,4.5), (3.2,3.1), (5.0,2.5), (5.5,4.5))$$
.

Both point sets describe pentagons, it is recommended to roughly sketch these to aid in answering the question.

#### ✓ Detailed solution

The equation can be interpreted as a simple algorithm. For every For every  $a \in A$  find the smallest distance to any point of  $b \in B$ , keep the maximum distance among all points. We highlight the shortest distance between a and the points of b in each case. The distance  $\|a-b\|$  is simply the Euclidean distance (or L2 norm).

 $\parallel a-b \parallel = \sqrt{(a_x-b_x)^2+(a_y-b_y)^2}$  where  $a_x,a_y$  and  $b_x,b_y$  are coordinates from shapes A and B

We will explicitly calculate the distance from each point of A to all the points of B, retaining the shortest distance.

$$a_0 = (4.5, 4.3)$$

$$\begin{pmatrix} Points\ in\ B & \parallel a_0 - b_i \parallel \ b_0 = (4.6, 5.0) & 0.71 \ b_1 = (3.3, 4.5) & 1.22 \ b_2 = (3.2, 3.1) & 1.77 \ b_3 = (5.0, 2.5) & 1.87 \ b_4 = (5.5, 4.5) & 1.02 \end{pmatrix}$$

We then find and record the shortest distance for this case

$$\min_{b \in B} \parallel a_0 - b \parallel = 0.71$$

We can repeat the same process and obtain the distances for the other elements of  $a \in A$ 

$$\begin{pmatrix} Points \ in \ B & \parallel a_1 - b_i \parallel & \parallel a_2 - b_i \parallel & \parallel a_3 - b_i \parallel & \parallel a_4 - b_i \parallel \ b_0 = (4.6, 5.0) & 1.63 & 3.13 & 3.30 & 2.89 \ b_1 = (3.3, 4.5) & 0.61 & 2.30 & 3.05 & 2.38 \ b_2 = (3.2, 3.1) & 0.82 & 0.90 & 1.91 & 1.14 \ b_3 = (5.0, 2.5) & 2.13 & 1.82 & 0.94 & 1.14 \ b_4 = (5.5, 4.5) & 2.18 & 3.25 & 2.97 & 2.80 \ \end{pmatrix}$$

The shortest distances in each case are:

$$\min_{b \in B} \| a_1 - b \| = 0.61$$

$$\min_{b \in B} \| a_2 - b \| = 0.9$$

$$\min_{b \in B} \parallel a_3 - b \parallel = 0.94$$

$$\min_{b \in B} \parallel a_4 - b \parallel = 1.14$$

The Hausdorff directed distance from A to B is then given by the longest of these shortest distances.

$$h\left(A,B\right) = \max_{a \in A} \min_{b \in B} \parallel a - b \parallel = 1.14$$

Note: We don't need to calculate every value as outlined above. As we are looking for the "longest shortest distance", by sketching the two pentagons we can easily see which points of A are furthest from B and just calculate the shortest distances for those points and determine the longest one which will be equal to the hausdorff distance.

### ■ Question at wrong answer

The equation can be interpreted as a simple algorithm. For every For every  $a \in A$  find the smallest distance to any point of  $b \in B$ , keep the maximum distance among all points. We highlight the shortest distance between a and the points of b in each case. The distance  $\|a-b\|$  is simply the Euclidean distance (or L2 norm) .  $\|a-b\| = \sqrt{(a_x-b_x)^2+(a_y-b_y)^2}$  where  $a_x,a_y$  and  $b_x,b_y$  are coordinates from shapes A and B.

We will explicitly calculate the distance from each point of A to all the points of B, retaining the shortest distance.

$$a_0 = (4.5, 4.3)$$

$$\begin{pmatrix} Points \ in \ B & \parallel a_0 - b_i \parallel \ b_0 = (4.6, 5.0) & 0.71 \ b_1 = (3.3, 4.5) & 1.22 \ b_2 = (3.2, 3.1) & 1.77 \ b_3 = (5.0, 2.5) & 1.87 \ b_4 = (5.5, 4.5) & 1.02 \end{pmatrix}$$

We then find and record the shortest distance.

$$\min_{b \in B} \parallel a_0 - b \parallel = 0.71$$

We will then need to repeat this for the other elements of A and then look for the  $\max_{a\in A}$  which will be the longest of the shortest distances we have found.

Feedback when answer is algebraically Equivalent to 1.14

No specific feedback

QUESTION

# Using the same values determine the directed Hausdorff distance $h\left(B,A\right) = \max_{b \in B} \min_{a \in A} \parallel a - b \parallel$

#### ✓ Detailed solution

We can use exactly the same approach as the previous question but this time we will get the distance from each point in B to all the points in A and record the shortest value.

$$\begin{pmatrix} b_0 = (4.6,5.0) & b_1 = (3.3,4.5) & b_2 = (3.2,3.1) & b_3 = (5) \\ \| a_i - b_0 \| & \| a_i - b_1 \| & \| a_i - b_2 \| & \| a_i - b_3 \| \\ a_0 = (4.5,4.3) & 0.71 & 1.22 & 1.77 & 1.8 \\ a_1 = (3.4,3.9) & 1.63 & 0.61 & 0.82 & 2.1 \\ a_2 = (3.2,2.2) & 3.13 & 2.30 & 0.90 & 1.8 \\ a_3 = (4.5,1.7) & 3.30 & 3.05 & 1.91 & 1.8 \\ a_4 = (3.9,2.2) & 2.89 & 2.38 & 1.14 & 1.1 \\ \min_{a \in A} \| a - b \| & 0.71 & 0.61 & 0.82 & 0.8 \end{pmatrix}$$

The Hausdorff directed distance from B to A is then given by the longest of these shortest distances.

$$h\left(B,A\right) = \max_{b \in B} \min_{a \in A} \parallel a - b \parallel = 1.02$$

We note that the  $h\left(B,A\right)$  may be same or different to  $h\left(A,B\right)$ . The overall hausdorff distance  $H\left(A,B\right)$  is the longer of the two.

$$H(A,B) = \max(h(A,B),h(B,A)) = 1.14$$

## $\boxed{\boxed}$ Question at wrong answer

We can use exactly the same approach as the previous question but this time we will get the distance from each point in B to all the points in A and record the shortest value.

For example we calculate the distance from the point  $b_0 = (4.6,5.0)$  to all the points of A and record the shortest distance.

$$\begin{aligned} b_0 &= (4.6, 5.0) \ &\parallel a_i - b_0 \parallel \ a_0 &= (4.5, 4.3) & 0.71 \ a_1 &= (3.4, 3.9) & 1.63 \ a_2 &= (3.2, 2.2) & 3.13 \ a_3 &= (4.5, 1.7) & 3.30 \ a_4 &= (3.9, 2.2) & 2.89 \ \min_{a \in A} \parallel a - b \parallel & 0.71 \ \end{pmatrix}$$

We perform a similar calculation for all the other points in B. The Hausdorff directed distance from B to A is then given by the longest of these shortest distances.

$$h\left(B,A\right) = \max_{b \in B} \min_{a \in A} \parallel a - b \parallel$$

Feedback when answer is algebraically Equivalent to 1.02

100% of points

No specific feedback

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106345

QUESTION

If a Histograms of Gradient based descriptor is comprised of: cells of  $5\times 5$  pixels, blocks of  $7\times 7$  cells, histograms with 10 orientations in each cell and  $10\times 30$  overlapping blocks at the top level; What is the length of the Descriptor vector?

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| 147000                                                 |
| Check my answer                                        |

#### ✓ Detailed solution

The Description vector in the Histogram of Orientated gradients algorithm is comprised of one histogram per cell, which has a specific number of orientations 10. Each block concatenates the values (which equals the number of orientations 10) from each of the cell histograms to produce a short vector. As there are  $7 \times 7$  cells this short vector in each block will be of length  $7 \cdot 7 \cdot 10 = 490$ 

As we always produce one histogram from each cell, the number of pixels in each cell doesn't matter, just the number of cells.

These short vectors from each block are concatenated to produce the full descriptor. There are  $10 \times 30 = 300$  blocks, so we must further

multiply by length of the short vector by the number of blocks to get the final length  $Length_{Des}=490\cdot 300=147000$ 

## □ Question at wrong answer

The Description vector in the Histogram of Orientated gradients algorithm is comprised of one histogram per cell, which has a specific number of orientations. Each block concatenates the values from each of the cell histograms to produce a short vector. These vectors from each block are concatenated to produce the full descriptor.

✓ Feedback when answer is equal to 147000

100% of points

No specific feedback

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106552

**QUESTION** 

In the Histogram of Orientated gradients algorithm a histogram is constructed in each 4 x 4 cell using the magnitude & direction information of each pixel. Calculate the magnitude contribution to the two nearest bins for a pixel with a magnitude  $\mu=83$  and direction  $\theta=95.6$  (degrees). Assume bilinear distribution of the magnitude between adjacent bins. Assume the histogram will use 9 orientations as shown. (give your answer as 1 x 2 matrix containing the calculated magnitudes

## Check my answer

### ✓ Detailed solution

As the direction value of the pixel is given as 95.6 the magnitude will be divided between the two nearest bins at  $\phi_1=90.00$  and  $\phi_2=110.00$  .

We can easily calculate the contribution to the two bins using binlinear interpolation.

We can write an equation for the contribution to bin 1 and bin2. (Note that the contribution to each bin is inversely proportional to the distance i.e. this is why we get the distance from  $\theta$  to the bin 2 centre  $\phi_2$  when calculating the magnitude contribution to bin 1)

Magnitude Contribution to bin 1:

$$\mu_1 = \mu \frac{|\phi_2 - \theta|}{|\phi_2 - \phi_1|} = 83 \cdot \frac{|110.00 - 95.6|}{|110.00 - 90.00|} = 60$$

Magnitude Contribution to bin2:

$$\mu_2 = \mu \frac{\mid \theta - \phi_1 \mid}{\mid \phi_2 - \phi_1 \mid} = 83 \cdot \frac{\mid 95.6 - 90.00 \mid}{\mid 110.00 - 90.00 \mid} = 23$$

We can easily calculate the contribution to the two bins using binlinear interpolation.

For example we can write an equation for the contribution to bin 1. (Note that the contribution to each bin is inversely proportional to the distance i.e. this is why we get the distance from  $\theta$  to the bin 2 centre  $\phi_2$  when calculating the magnitude contribution to bin 1)

$$\mu_1 = \mu \frac{\mid \phi_2 - \theta \mid}{\mid \phi_2 - \phi_1 \mid}$$

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 60 & 23 \end{pmatrix}$ 

100% of points

No specific feedback

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106864

QUESTION

## Why is a sliding window often necessary for object detection?

| ○ To find objects in the image                                                                                                                      | ✓              |
|-----------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| ○ To reduce noise in the image                                                                                                                      | ×              |
| To increase the resolution of the image                                                                                                             | ×              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                           |                |
| Feedback for answer: 1                                                                                                                              | 100% of points |
| This is correct, the detector is applied at regular grid points in th image (similar to convolution ) and this is called a sliding window approach. |                |
| × Feedback for answer: 2                                                                                                                            | 0% of points   |
| Reducing noise is not related to the sliding window technique. Needuction is typically handled by other preprocessing methods sa gaussian blur.     |                |
| × Feedback for answer: 3                                                                                                                            | 0% of points   |
| Increasing the resolution of the image is not the purpose of a sli window. The sliding window is used in object detection to exhau                  | J              |

scan the image in a grid pattern, each time applying the detection algorithm to a pixel region that fits detector window.

QUESTION

Why do classical object detectors such as the Histogram of Orientated Gradients have a fixed detection size?

| The training data only included pedestrians of one size. | × |
|----------------------------------------------------------|---|
|                                                          |   |
| The detector size is limited by computation.             | × |

Select one answer. Upon selection it will submit your answer immediately.

## ✓ Feedback for answer: 1 100% of points

This is correct, the descriptor is calculated from histograms obtained from overlapping blocks. The overlapping blocks span a fixed number of pixels. For an input object such as a pedestrian to be detected it must occupy the same number of pixels to produce a recognisable (by a classifier) descriptor vector. If the pedestrian occupies a larger or smaller pixel range the descriptor may be not be representative of the pedestrians (as expected by the classifier).

It is true the training data consisted of pedestrians of a fixed size, however this is a consequence of the detector having a fixed detection size (and requiring data to be the right size), not the other way around.

This statement is incorrect, the Histogram of Orientated Gradients is computationally lightweight and the detector size can be expanded for various image sizes. However an image pyramid is typically used to reduce image sizes and allow detection of large objects, rather than building a large detector (or detectors for each required scale).

QUESTION

## What additional problem with a fixed detector window size does using an image pyramid solve?

|          | Allows detection of different scale objects                                                                                                                                                                                                                            | ✓              |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          | Allows detection of objects in different positions                                                                                                                                                                                                                     | ×              |
|          | Allows detection of rotated objects in an image                                                                                                                                                                                                                        | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                                                              |                |
| <b>✓</b> | Feedback for answer: 1                                                                                                                                                                                                                                                 | 100% of points |
|          | This is correct, the image pyramid allows detection of larger objected in size than the detection window. By gradually scaling of the image, eventually large objects will occupy a region of pixels compatible with the detector window size.                         | down           |
| ×        | Feedback for answer: 2                                                                                                                                                                                                                                                 | 0% of points   |
|          | This is incorrect. The sliding window instead address the proble locating objects in different regions of an image.                                                                                                                                                    | m of           |
| ×        | Feedback for answer: 3                                                                                                                                                                                                                                                 | 0% of points   |
|          | The image pyramid helps with detecting objects of different scalar an image. Examples of objects in images that are rotated may not detected by classical detector, particularly detectors such as HC rely on detecting specific orientations of gradients in objects. | ot be          |

QUESTION

## What is the main drawback with using a sliding window and image pyramid approach?

| It decreases the accuracy of object detection | × |
|-----------------------------------------------|---|
|                                               |   |
| It increases computation time.                | ✓ |
|                                               |   |
| It reduces the image resolution.              | × |

Select one answer. Upon selection it will submit your answer immediately.

### 

0% of points

We typically would expect detection accuracy to be increased by using a sliding window and image pyramid. However there is a trade off between accuracy and computation. Adding more levels in pyramid and reducing steps taken by the sliding window will increase computation.

#### ✓ Feedback for answer: 2

100% of points

This is correct, using a pyramid with lots of levels and short steps in the sliding window will improve accuracy at the cost of increased computation.

### 

0% of points

The image resolution is reduced successively (stages of gaussian blur and downsampling) in constructing the image pyramid, however this not a drawback apart from the computational cost of doing this. (Note we will retain a copy of the original image, so there is no information loss)

QUESTION

## What is the purpose of non-maximum suppression (NMS) in object detection?

| ○ To remove duplicate detections                                                                 |       |
|--------------------------------------------------------------------------------------------------|-------|
|                                                                                                  |       |
| ○ o enhance the image quality ×                                                                  | J     |
| Select one answer. Upon selection it will submit your answer immediately.                        |       |
| ⊠ Feedback for answer: 1 0% of p                                                                 | oints |
| NMS does not increase the number of detected objects; it helps in refining the detections.       |       |
| ✓ Feedback for answer: 2                                                                         | oints |
| Correct!                                                                                         |       |
| ➤ Feedback for answer: 3 0% of p                                                                 | oints |
| Enhancing image quality is not the purpose of NMS, it reduces the number of duplicate detections |       |

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106871

**QUESTION** 

This question is not available in English. Please click the flag icon in the top right of the screen to switch to Dutch.

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106978

**QUESTION** 

In the bag of words we want to determine the Tf-idf (term Frequency-inverse document frequency) weighted word frequency  $t_i$  to a word ithat appears  $n_{id}=44$  times in a document, given that  $n_d=743$  is the total number of words in document,  $n_i=59066$  is the number of times the word appears in a database document, and N=900000 is the number of documents in the database. (you can give your answer in the box below and round it to 3 decimal places)

### Detailed solution

We can use the formula  $t_i=\frac{n_{id}}{n_d}\log\left(\frac{N}{n_i}\right)$  where the terms are as given in the question,  $n_d=743$ ,  $n_{id}=44$ ,  $n_i=59066$ , N=900000, the logarithm is base 10.

Therefore 
$$t_i=\frac{n_{id}}{n_{di}}\log\left(\frac{N}{n_i}\right)=\frac{44}{743}\log\left(\frac{900000}{59066}\right)=0.070$$

### Question at wrong answer

We can use the formula  $t_i=\frac{n_{id}}{n_d}\log\left(\frac{N}{n_i}\right)$  where the terms are as given in the question,  $n_d=743$ ,  $n_{id}=44$ ,  $n_i=59066$ , N=900000 and the log is base 10.

✓ Feedback when answer is equal to 0.070

100% of points

No specific feedback

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106979

**QUESTION** 

The bag of words algorithm matches similar documents by comparing word frequency.

Incorrect. The bag of words algorithm does match similar documents

by comparing word frequency.

QUESTION

The bag of features representation can be used for image classification.

|   | ○ True                                                                                                                     |                |
|---|----------------------------------------------------------------------------------------------------------------------------|----------------|
|   |                                                                                                                            |                |
|   | ○ False                                                                                                                    | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                  |                |
| ~ | Feedback for answer: 1                                                                                                     | 100% of points |
|   | Correct!                                                                                                                   |                |
| × | Feedback for answer: 2                                                                                                     | 0% of points   |
|   | Incorrect. The bag of features can be used for image classificati although it requires modifications for object detection. | on,            |
|   |                                                                                                                            |                |

QUESTION

## The Bag of Features uses an Orderless Representation.

| ○ True  | ✓ |
|---------|---|
|         |   |
| ○ False | X |

Select one answer. Upon selection it will submit your answer immediately.

#### ✓ Feedback for answer: 1

100% of points

Correct! By an orderless representation, we mean that the position in the image that feature was obtained from is not included in building the final descriptor. The loss of this position information, could be considered a disadvantage, however the Bag of Features still performs well at image classification.

### 

0% of points

Incorrect. The bag of features uses an orderless representation, which is effective for image classification. By and orderless representation, we mean that the position in the image that feature was obtained from is not included in building the final descriptor. The loss of this position information, could be considered a disadvantage, however the Bag of Features still performs well at image classification.

QUESTION

Which popular feature (or keypoint) descriptor do we use in the Bag of Features to represent image features (or small pixel regions)

| Feedback for answer: 1                                                    | 0% of po |
|---------------------------------------------------------------------------|----------|
| Select one answer. Upon selection it will submit your answer immediately. |          |
| ○ ORB                                                                     | ×        |
|                                                                           |          |
| ○ SURF                                                                    | ×        |
|                                                                           |          |
| ○ SIFT                                                                    | ✓        |
|                                                                           |          |
| ○ HOG                                                                     | ×        |
|                                                                           |          |

## $|\mathsf{X}|$

ints

Incorrect. HOG (Histogram of Oriented Gradients) descriptor is used for object detection, not typically for bag of features

Feedback for answer: 2

100% of points

Correct! SIFT (Scale-Invariant Feature Transform) is commonly used for its robustness in representing image features within the bag of features.

0% of points

Incorrect. SURF (Speeded-Up Robust Features) is an alternative to SIFT but is less commonly used in the bag of features.

× Feedback for answer: 4

0% of points

Incorrect. ORB (Oriented FAST and Rotated BRIEF) is another alternative feature detector and descriptor. The BRIEF descriptor is binary based and not as robust as SIFT for the purpose of feature description.

QUESTION

## Why is the SIFT descriptor preferred in the Bag of Features Approach.

|        | ) It is faster than other descriptors to compute                                             | ×              |
|--------|----------------------------------------------------------------------------------------------|----------------|
|        | ○ It produces robust descriptors suitable for clustering                                     | ✓              |
|        | ○ It is simpler to implement                                                                 | ×              |
| Sele   | ect one answer. Upon selection it will submit your answer immediately.                       |                |
| ⊠ Feed | dback for answer: 1                                                                          | 0% of points   |
|        | correct. SIFT is not necessarily the fastest approach to compuscriptor from.                 | te a           |
| ✓ Feed | dback for answer: 2                                                                          | 100% of points |
|        | errect. SIFT is preferred because it produces robust descriptone e effective for clustering. | rs that        |
| × Feed | dback for answer: 3                                                                          | 0% of points   |
|        | T is not the simplest descriptor to implement, requiring seve ocessing steps.                | ral            |
|        |                                                                                              |                |

QUESTION

Select from the options below the correct explanation for how the Bag of Features Representation is generated for an image.

|   | 0 | Extract features, cluster features into a visual vocabulary, assign features to visual words, generate a term vector. | <b>✓</b> |
|---|---|-----------------------------------------------------------------------------------------------------------------------|----------|
| _ |   |                                                                                                                       |          |
|   | 0 | Extract features, sort features by size, assign features to visual words, generate a term vector.                     | ×        |
| _ |   |                                                                                                                       |          |
|   | 0 | Extract features, cluster features into a visual vocabulary, generate a term vector from cluster centres.             | ×        |

Select one answer. Upon selection it will submit your answer immediately.

☑ Feedback for answer: 1
   100% of points

Correct! The first step in the bag of features approach is to build a visual vocabulary. To achieve this we first extract and represent the features from all images in a dataset. The features may be obtained with a feature detector or by random or uniform sampling. The features are then clustered by similarity into a visual vocabulary. Once the vocabulary is created we can then assign terms to the test image, we first extract the image features (in the same manner as was done for the visual vocabulary) and then assign features to the nearest visual words. We finally generate a term vector by recording the counts of each visual words to form a histogram

Incorrect! All the feature descriptors extracted by SIFT will be of exactly the same size. The features are clustered to find common descriptors across all the images of the dataset.

Incorrect! We do not generate the term vector directly from the cluster centres. Instead for each test image we want to represent, feature descriptors are extracted and the cluster these descriptors belong to identified (i.e. visual words). The term vector is a histogram of the counts of the visual words the for test image.