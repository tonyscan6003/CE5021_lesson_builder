<u>View subject</u> | <u>Edit subject</u>

# Object Detection (w Deep Learning)

Slides

# Exercises

### <u>View exercise</u> | <u>Edit exercise</u>

### Exercise 105760

QUESTION

Calculate Intersection over union.

Calculate the Intersection over Union of the two bounding boxes defined (by the coordinate sets) below. The Predicted box  $B_p$  has coordinates at the top left  $(x_{b1},y_{b1}=44,39)$  and bottom right  $(x_{b2},y_{b2}=72,60)$ . The ground truth box  $B_{gt}$  has coordinates at the top left  $(x_{gt1},y_{gt2}=38,45)$  and bottom right  $(x_{gt2},y_{gt2}=54,88)$ .

Give your answer in the box below to 2 decimal places of accuracy.

Note that depending on the given coordinates, the overlap of the boxes may look like any one of the configurations below. (It is recommended to sketch a diagram for the set of coordinates given in the question)

### ✓ Detailed solution

The intersection over union is calculated as the intersection area divided by the union area of the two boxes.  $IoU = \frac{Area\left(B_p \cap B_{gt}\right)}{Area\left(B_p \cup B_{gt}\right)}$ 

Step1: Firstly we obtain the intersection area  $Area\left(B_p\cap B_{gt}\right)$  of the two boxes. As we can see in the image from the question the bounding boxes may overlap with different orientations, and the exact coordinates that define the overlap area are different in each case. (We can determine these by inspection or) we can use simple rules to determine the coordinates of the intersection area.

The  $x_{I1},y_{I1}$  coordinates of the top left of the intersection area are given as the maximum of the bounding box or ground truth coordinates:

$$x_{I1} = Max(x_{b1}, x_{at1}) = Max(44,38) = 44$$

$$y_{I1} = Max\left(y_{b1}, y_{gt1}\right) = Max\left(39, 45\right) = 45$$

Similarly the  $x_{I2},y_{I2}$  coordinates of the bottom right of the intersection area are given as the minimum of the bounding box or ground truth coordinates:

$$x_{I2} = Min\left(x_{b2}, x_{qt2}\right) = Min\left(72, 54\right) = 54$$

$$y_{I2}=Min\left(y_{b1},y_{gt1}\right)=Min\left(60,88\right)=60$$

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

 $\boxed{\boldsymbol{\varphi}}$  Feedback when answer is equal to 0.13

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

### Exercise 109371

QUESTION

The output of an object detector is given as a series of predictions with a pair of Intersection over Union Values and Confidence Values for each prediction (each pair corresponds to the same position in the matrix of values).

$$IoU = \begin{pmatrix} 0.67 & 0.68 & 0.84 & 0.13 \\ 0.34 & 0.52 & 0.79 & 0.33 \\ 0.68 & 0.75 & 0.42 & 0.29 \\ 0.57 & 0.8 & 0.86 & 0.79 \\ 0.68 & 0.79 & 0.27 & 0.6 \end{pmatrix}$$

$$Conf = \begin{pmatrix} 0.22 & 0.46 & 0.49 & 0.43 \\ 0.81 & 0.21 & 0.4 & 0.22 \\ 0.74 & 0.4 & 0.34 & 0.22 \\ 0.43 & 0.71 & 0.55 & 0.42 \\ 0.65 & 0.76 & 0.74 & 0.23 \end{pmatrix}$$

Determine the **Precision** of the object detector for an IoU threshold  $T_{IoU}=0.50$  and Confidence Threshold  $T_C=0.20$ , given the number of ground truth objects in the input images to the detector is  $N_{GT}=27$ .

#### ✓ Detailed solution

In order to calculate the precision and recall (next part of question) we must first determine the number of True Positives, False Positive and False Negatives. (Note for object detection we don't calculate the accuracy or count the True Negatives as these correspond to the background and would dominate the statistics.)

We can use the thresholds provided to determine the True/False positives.

We first apply the confidence threshold  $T_C=0.20$  to the set of detections. Any detection with a confidence of less than or equal to 0.20 is rejected. The indicator matrix  $I_C$  below corresponds to the applying  $T_C$  threshold to the matrix of confidence values (1 if greater than  $T_C$ )

$$I_C = \begin{pmatrix} 1 & 1 & 1 & 1 \ 1 & 1 & 1 & 1 \ 1 & 1 &$$

Similarly we can find the detections that are greater than the IoU threshold and create another indicator matrix  $I_{IOU}$ 

$$I_{IOU} = \begin{pmatrix} 1 & 1 & 1 & 0 \ 0 & 1 & 1 & 0 \ 1 & 1 & 0 & 0 \ 1 & 1 & 1 & 1 \ 1 & 1 & 0 & 1 \end{pmatrix}$$

Only detections that pass pour and threshold (given by the combined indicator matrix  $\begin{pmatrix} 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 0 & 1 \end{pmatrix}$  ) are

considered to be True Positives.

$$N_{TP} = 14$$

Detections that pass the confidence threshold but not the IoU

threshold (given by the indicator matrix  $\begin{pmatrix} 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}$  ) are

considered to be False Positives.

$$N_{FP}=6$$

We can then calculate the precision from these values.

$$P = \frac{N_{TP}}{N_{TP} + N_{FP}} = \frac{14}{14 + 6} = 0.70$$

#### □ Question at wrong answer

In order to calculate the precision and recall (next part of question) we must first determine the number of True Positives, False Positive and False Negatives. (Note for object detection we don't calculate the accuracy or count the True Negatives as these correspond to the background and would dominate the statistics.)

We can use the thresholds provided to determine the True/False positives.

We first apply the confidence threshold  $T_C=0.20$  to the set of detections. Any detection with a confidence of less than or equal to 0.20 is rejected. The indicator matrix  $I_C$  below corresponds to the applying  $T_C$  threshold to the matrix of confidence values (1 if greater than  $T_C$ )

$$I_C = \begin{pmatrix} 1 & 1 & 1 & 1 \ 1 & 1 & 1 & 1 \ 1 & 1 &$$

Similarly we can find the detections that are greater than the IoU threshold and create another indicator matrix  $I_{IOU}$ 

$$I_{IOU} = \begin{pmatrix} 1 & 1 & 1 & 0 \ 0 & 1 & 1 & 0 \ 1 & 1 & 0 & 0 \ 1 & 1 & 1 & 1 \ 1 & 1 & 0 & 1 \end{pmatrix}$$

Only detections that pass both the confidence threshold and IoU

threshold (given by the combined indicator matrix  $\begin{pmatrix} 1 & 1 & 1 & 0 \\ 0 & 1 & 1 & 0 \\ 1 & 1 & 0 & 0 \\ 1 & 1 & 1 & 1 \\ 1 & 1 & 0 & 1 \end{pmatrix}$  ) are considered to be True Positives

$$N_{TP}=14$$

Detections that pass the confidence threshold but not the IoU

threshold (given by the indicator matrix  $\begin{pmatrix} 3 & 3 & 3 & 1 \\ 1 & 0 & 0 & 1 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 &$ 

considered to be False Positives.

$$N_{FP}=6$$

We can then calculate the precision from these values.

$$P = \frac{N_{TP}}{N_{TP} + N_{FP}}$$

QUESTION

Calculate the **Recall** for Object detection given the IoU and Confidence values. (Assume same Tc=0.20 and

$$T_{IoU} = 0.50$$
)

0.52

# Check my answer

#### ✓ Detailed solution

As the confidence and IoU thresholds stay the same from the 1st part of the question, the number of True Positives are the same. In order to calculate the Recall we determine the number of false negatives as the number of Ground truth Objects minus the number of True positives.

$$N_{FN} = 27 - 14 = 13$$

$$R = \frac{N_{TP}}{N_{TP} + N_{FN}} = \frac{14}{14 + 13} = 0.52$$

### 

We determine the number of false negatives as the number of Ground truth Objects minus the number of True positives.

$$N_{FN} = 27 - 14 = 13$$

The Recall can then be calculated as

$$R = \frac{N_{TP}}{N_{TP} + N_{FN}}$$

 ✓ Feedback when answer is equal to 0.52
 100% of points

 No specific feedback

QUESTION

Assuming we change the confidence Threshold to  $T_C=0.75$  determine the precision in this case.

✓ Detailed solution

We can use the same methodology as in the first part of the question (with the new confidence value) to find the revised number of True/False positives and use this to determine precision and recall.

We first apply the confidence threshold  $T_C=0.75$  to the set of detections. Any detection with a confidence of less than or equal to 0.75 is rejected. The indicator matrix  $I_C$  below corresponds to the applying  $T_C$  threshold to the matrix of confidence values (1 if greater than  $T_C$ )

$$I_C = \begin{pmatrix} 0 & 0 & 0 & 0 \ 1 & 0 & 0 & 0 \ 0 & 0 & 0 & 0 \ 0 & 0 & 0$$

Similarly we can find the detections that are greater than the IoU threshold and create another indicator matrix  $I_{IOU}$ 

$$I_{IOU} = \begin{pmatrix} 1 & 1 & 1 & 0 \ 0 & 1 & 1 & 0 \ 1 & 1 & 0 & 0 \ 1 & 1 & 1 & 1 \ 1 & 1 & 0 & 1 \end{pmatrix}$$

Only detections that pass both the confidence threshold and IoU

threshold (given by the combined indicator matrix  $\begin{pmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 &$ 

considered to be True Positives.

$$N_{TP} = 1$$

Detections that pass the confidence threshold but not the IoU

threshold (given by the indicator matrix  $\begin{pmatrix} 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0$ 

considered to be False Positives.

$$N_{FP}=1$$

We can then calculate the precision from these values.

$$P = \frac{N_{TP}}{N_{TP} + N_{FP}} = \frac{1}{1+1} = 0.50$$

## ■ Question at wrong answer

We can use the same methodology as in the first part of the question (with the new confidence value) to find the revised number of True/False positives and use this to determine precision and recall.

☑ Feedback when answer is equal to 0.50

100% of points

QUESTION

# Calculate the recall for this Tc=0.75

0.04

# Check my answer

#### ☑ Detailed solution

In order to calculate the Recall we determine the number of false negatives as the number of Ground truth Objects minus the number of True positives.

$$N_{FN} = 27 - 1 = 26$$

$$R = \frac{N_{TP}}{N_{TP} + N_{FN}} = \frac{1}{1 + 26} = 0.52$$

### □ Question at wrong answer

In order to calculate the Recall we determine the number of false negatives as the number of Ground truth Objects minus the number of True positives.

$$N_{FN} = 27 - 1 = 26$$

$$R = \frac{N_{TP}}{N_{TP} + N_{FN}}$$

✓ Feedback when answer is equal to 0.04

100% of points