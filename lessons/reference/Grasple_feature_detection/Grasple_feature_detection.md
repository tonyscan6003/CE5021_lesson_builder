<u>View subject</u> | <u>Edit subject</u>

## **Feature Detection**

Slides

### **Exercises**

### <u>View exercise</u> | <u>Edit exercise</u>

### Exercise 106282

QUESTION

### List some qualities of a good feature detector

|          | List some quanties of a good feature detector |                                                                                   |                |
|----------|-----------------------------------------------|-----------------------------------------------------------------------------------|----------------|
|          | $ \bigcirc$                                   | Finds features that are local, invariant, numerous, and robust to noise and blur. | ✓              |
|          | $\boxed{\bigcirc}$                            | Finds features that are global, invariant, few in number, and sensitive to noise. | o ×            |
|          |                                               | Finds features that are local, variant, numerous, and robust to noise and blur.   | ×              |
|          | $\bigcirc$                                    | Finds features that are global, variant, few in number, and sensitive to noise.   | ×              |
|          | Select                                        | one answer. Upon selection it will submit your answer immediately.                |                |
| <b>✓</b> | Feedba                                        | ck for answer: 1                                                                  | 100% of points |
|          | Corr                                          | ect!                                                                              |                |
| ×        | Feedba                                        | ck for answer: 2                                                                  | 0% of points   |
|          | Inco                                          | rrect: A good feature detector should find local features, not                    | -<br>-         |

global ones, and should be robust to noise

0% of points

Incorrect: Features should be invariant, not variant. Invariant features are those which don't change under transformation for example a rotation of the image.

× Feedback for answer: 4

0% of points

Incorrect: Features should be numerous and robust to noise, not few and sensitive.

QUESTION

# In feature matching of images, how is a match between two descriptor vectors detected?

|          | By comparing the largest Euclidean distance                                                   | ×              |
|----------|-----------------------------------------------------------------------------------------------|----------------|
|          | By comparing the smallest Euclidean distance.                                                 | ✓              |
|          | By comparing the largest Manhattan distance.                                                  | X              |
|          | By comparing the smallest Manhattan distance.                                                 | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                     |                |
| ×        | Feedback for answer: 1                                                                        | 0% of points   |
|          | Incorrect: Feature matching is done by finding the smallest, not largest, Euclidean distance. | the            |
| <b>✓</b> | Feedback for answer: 2                                                                        | 100% of points |
|          | Correct! The smallest euclidean distance between feature vecto typically used for matching.   | rs is          |
| ×        | Feedback for answer: 3                                                                        | 0% of points   |
|          | Incorrect: Euclidean distance is typically used for descriptor mat<br>not Manhattan distance. | ching,         |
| ×        | Feedback for answer: 4                                                                        | 0% of points   |
|          |                                                                                               |                |

Incorrect: Euclidean distance is typically used for descriptor matching, not Manhattan distance.

QUESTION

## If the Laplacian of Gaussian "blob" feature detector is to be made scale invariant, how will this be achieved?

|          | By keeping the value of sigma constant.                                                                                                                                                                                  | X              |
|----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          | By changing the value of sigma for each scale                                                                                                                                                                            | ✓              |
|          | By using a fixed kernel size.                                                                                                                                                                                            | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                |                |
| ×        | Feedback for answer: 1                                                                                                                                                                                                   | 0% of points   |
|          | Incorrect: Keeping sigma constant does not achieve scale invar (detector will always be the same scale).                                                                                                                 | iance          |
| <b>V</b> | Feedback for answer: 2                                                                                                                                                                                                   | 100% of points |
|          | Correct: We must vary sigma to change the scale of detection.                                                                                                                                                            |                |
| ×        | Feedback for answer: 3                                                                                                                                                                                                   | 0% of points   |
|          | Incorrect. A fixed kernel size does not account for different scale must change the sigma value to change the scale of the detector the kernel size must increase along with sigma in order to avoid truncated detector. | r and          |
|          |                                                                                                                                                                                                                          |                |

QUESTION

### What is the primary purpose of the SIFT algorithm?

|     | To compress images for storage                                                                                                                                                                                                                                                    | ×                            |
|-----|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------|
|     | ○ To convert images to grayscale                                                                                                                                                                                                                                                  | ×                            |
|     | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                                                                         |                              |
| × I | Feedback for answer: 1                                                                                                                                                                                                                                                            | 0% of points                 |
|     | Incorrect: The SIFT algorithm is designed to detect and describe features in images, making it useful for object recognition.                                                                                                                                                     | local                        |
|     |                                                                                                                                                                                                                                                                                   |                              |
| ✓ 1 | Feedback for answer: 2                                                                                                                                                                                                                                                            | 100% of points               |
|     | Feedback for answer: 2  Correct!                                                                                                                                                                                                                                                  | 100% of points               |
|     |                                                                                                                                                                                                                                                                                   | 100% of points  0% of points |
| × I | Correct!                                                                                                                                                                                                                                                                          | 0% of points                 |
| × I | Feedback for answer: 3 Incorrect: The SIFT algorithm is designed to detect and describe features in images. In more complex algorithms local features (for by SIFT) from an image may be combined to produce an overall representation of the image. But this is not a compressed | 0% of points                 |

Incorrect: Greyscale images are produced by combining red green and blue channels of a colour image or directly from a single channel image sensor. The SIFT algorithm is not involved.

QUESTION

# What is the reason behind determining the characteristic orientation in the SIFT keypoint descriptor?

|          | To match keypoints for different scales of an object                                      | ×              |
|----------|-------------------------------------------------------------------------------------------|----------------|
|          | To match keypoints for different orientations of an object.                               | ✓              |
|          | To match keypoints for different colours of an object.                                    | ×              |
|          | To match keypoints for different textures of an object.                                   | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                 |                |
| ×        | Feedback for answer: 1                                                                    | 0% of points   |
|          | Incorrect: Characteristic orientation is for matching different orientations, not scales. |                |
| <b>~</b> | Feedback for answer: 2                                                                    | 100% of points |
|          | Correct!                                                                                  |                |

Incorrect: Colour is not considered in the SIFT keypoint descriptor.

Incorrect: Texture is not considered in the SIFT keypoint descriptor.

QUESTION

# Which function is used to produce the Scale-Spaces used in the Extrema Detection stage of the SIFT algorithm?

|          | ○ Laplacian                                                                                                                                                                                                                                                | ×              |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|          | Gaussian                                                                                                                                                                                                                                                   | $\checkmark$   |
|          | Fourier                                                                                                                                                                                                                                                    | ×              |
|          | ○ Sobel                                                                                                                                                                                                                                                    | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                                                                                                                                                                                  |                |
| ×        | Feedback for answer: 1                                                                                                                                                                                                                                     | 0% of points   |
|          | Incorrect: The Scale-Space Extrema Detection stage uses a Gaus function to produce the scale space. The Laplacian function can used to produce a scale space but it not used in the SIFT algorithm Gaussian scale space is more computationally efficient. | be             |
| <b>▽</b> | Feedback for answer: 2                                                                                                                                                                                                                                     | 100% of points |
|          | Correct! A Gaussian scale space is used in the SIFT Algorithm.                                                                                                                                                                                             |                |
| ×        | Feedback for answer: 3                                                                                                                                                                                                                                     | 0% of points   |
|          | Incorrect: The fourier transform is not used to create a scale spa                                                                                                                                                                                         | ace.           |
| ×        | Feedback for answer: 4                                                                                                                                                                                                                                     | 0% of points   |

Incorrect: The Sobel function (or operator) is used as an edge detector and not used to create a scale space (as it only has one fixed scale).

QUESTION

# How are keypoints assigned orientations in the SIFT algorithm?

| By using the x-direction Sobel operator                                                                                                                                   | ×              |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| By computing the gradient magnitude and orientation                                                                                                                       | ✓              |
| By applying a Fourier transform                                                                                                                                           | ×              |
| Select one answer. Upon selection it will submit your answer immediately.                                                                                                 |                |
| ☑ Feedback for answer: 1                                                                                                                                                  | 0% of points   |
| The sobel operator will find the gradient in one direction one however it is necessary to find the y-direction gradient and them into a gradient magnitude and direction. | _              |
| ✓ Feedback for answer: 2                                                                                                                                                  | 100% of points |
| Correct!                                                                                                                                                                  |                |
|                                                                                                                                                                           | 0% of points   |
| The fourier transform is not used to obtain gradient inform                                                                                                               | nation         |
|                                                                                                                                                                           |                |

QUESTION

## What is the typical size of the feature vector used in the SIFT Algorithm

| <u> </u> | ✓ |
|----------|---|
|          |   |
|          | × |
|          |   |
| ○ 512    | × |

Select one answer. Upon selection it will submit your answer immediately.

#### ✓ Feedback for answer: 1

100% of points

Correct: The feature vector in the SIFT algorithm typically contains 128 elements, derived from 16 histograms with 8 orientation bins each.

0% of points

Incorrect: The SIFT algorithm typically divides the image patch into 16 histograms with 8 bins giving 128 elements.

0% of points

Incorrect: The SIFT algorithm typically divides the image patch into 16 histograms with 8 bins giving 128 elements.

QUESTION

# What technique is used to detect stable keypoint locations in the scale-space?

|   | O Difference of Gaussians                                                                                                                                | ✓              |
|---|----------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
|   | ○ Canny edge detection                                                                                                                                   | ×              |
|   | O Hough transform                                                                                                                                        | ×              |
|   | Laplacian of Gaussian                                                                                                                                    | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                                                                |                |
| ~ | Feedback for answer: 1                                                                                                                                   | 100% of points |
|   | Correct! The SIFT algorithm uses the Difference of Gaussian to f extrema in scale space. The DoG is an approximation of the Lap of gaussian scale space. |                |
| × | Feedback for answer: 2                                                                                                                                   | 0% of points   |
|   | Incorrect: Canny edge detection will only find edges not extrear                                                                                         | na.            |
| × | Feedback for answer: 3                                                                                                                                   | 0% of points   |
|   | Incorrect: The Hough algorithm will only find lines from edges, ridentify extrema in scale space.                                                        | not            |
| × | Feedback for answer: 4                                                                                                                                   | 0% of points   |
|   |                                                                                                                                                          |                |

The Laplacian of Gaussian of gaussian could also be used to directly produce a scale spaces that highlights extrema. However the difference of Gaussians is more computationally efficient to compute.

### Exercise 106300

QUESTION

Perform BiLinear Interpolation in SIFT Algorithm

In constructing the descriptor vector in the SIFT algorithm, a  $4 \times 4$  grid of histograms is obtained from a region of pixels centered on the identified keypoint. (The exact size of the grid depends on the scale of detection).

Assuming a pixel  $(x_i,y_j)=0.7,1.5$  with magnitude (8.2), determine the contribution of the magnitude to the histogram centre at  $(x_1,y_1)=(0,0)$ .

### Check my answer

#### ✓ Detailed solution

In order to divide the pixel magnitude between the histogram centres, a form of bilinear interpolation can be used. The contribution of the magnitude from the pixel  $(x_i,y_j)$  to any of the histogram centres is inversely proportional to the distance from the original pixel to the histogram centre.

Rather than calculating distances, the simplest way calculate the bilinear interpolation is with a geometric approach. The scaling of the magnitude contribution from the pixel  $(x_i,y_j)=(0.7,1.5)$  to any of the histogram centres may simply be determined as the area diagonally opposite the histogram centre. For example the intensity contribution to the red pixel  $(x_1,y_1)$  is proportional to the diagonally opposite red shaded area.

As the question only asks for the contribution to the red pixel  $(x_1,y_1)$  we will just calculate the red shaded area. (Note that for images the coordinate system has the origin in the top left corner):

$$A_{x_1,y_1(red)} = \left(y_2 - y_i\right)\left(x_2 - x_i\right) = \left(2 - 1.5\right)\left(2 - 0.7\right) = 0.65$$

We then multiply the calculated area by the magnitude of the pixel  $(x_i,y_j)$  to be divided between the histograms. (We also normalise the answer by dividing by the sum of the areas

$$M_{x_1,y_1} = \frac{M_{x_i,y_j} A_{x_1,y_1}}{4}$$

$$M_{x_1,y_1} = 1.33$$

### □ Question at wrong answer

In order to divide the pixel magnitude between the histogram centres, a form of bilinear interpolation can be used. The contribution of the magnitude from the pixel  $(x_i,y_j)$  to any of the histogram centres is inversely proportional to the distance from the original pixel to the histogram centre.

Rather than calculating distances, the simplest way calculate the bilinear interpolation is with a geometric approach. The scaling of the magnitude contribution from the pixel  $(x_i,y_j)=(0.7,1.5)$  to any of the histogram centres may simply be determined as the area diagonally opposite the histogram centre. For example the intensity contribution to the red pixel  $(x_1,y_1)$  is proportional to the diagonally opposite red shaded area.

| Feedback when answer is equal to 1.33 | 100% of points |
|---------------------------------------|----------------|
| No specific feedback                  |                |