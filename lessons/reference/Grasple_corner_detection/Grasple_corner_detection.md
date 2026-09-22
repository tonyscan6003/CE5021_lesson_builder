## <u>View subject</u> | <u>Edit subject</u>

# Corner Detection

Slides

## **Exercises**

### <u>View exercise</u> | <u>Edit exercise</u>

#### Exercise 105758

**QUESTION** 

## Harris Corner Algorithm

In the Harris Corner Detection Algorithm, the second moment matrix is formed from approximate derivatives of the image convolved with a Gaussian window. Given the second moment matrix  $M = \begin{pmatrix} A & C \\ C & B \end{pmatrix} = \begin{pmatrix} 2.6 & 1.8 \\ 1.8 & 7.8 \end{pmatrix} \text{ calculate the corner}$ 

response function R and determine if the current point is an edge, corner or flat region (allow  $\alpha=0.04$ )

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |  |
|--------------------------------------------------------|--|
| $\frac{7946}{625}$                                     |  |

Check my answer

✓ Detailed solution

We can calculate the corner response function using the equation  $R = Det(M) - \alpha \left( Trace(M) \right)^2$ 

The determinant and trace of the second moment matrix are easily calculated from the matrix components:

$$Det(M) = AB - C^2 = 2.6 \cdot 7.8 - 1.8^2$$

and

$$Trace(M) = A + B = 2.6 + 7.8$$

We calculate R as

$$R = Det(M) - \alpha \left(Trace(M)\right)^2 = \frac{426}{25} - \alpha \cdot \frac{52}{5}^2 = \frac{7946}{625}$$

If the value is significantly greater than zero then a corner is likely. Values close to zero represent flat regions while values less than zero correspond to edges.

#### × Question at wrong answer

We can calculate the corner response function using the equation  $R = Det(M) - \alpha \left( Trace(M) \right)^2$ 

The determinant and trace of the second moment matrix are easily calculated from the matrix components:

$$Det(M) = AB - C^2$$

and

$$Trace(M) = A + B$$

✓ Feedback when answer is equal to \frac{7946}{625}

No specific feedback