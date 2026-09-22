<u>View subject</u> | <u>Edit subject</u>

# PCA & Eigenvectors/Values

Slides

# **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 106986

QUESTION EIGENVECTORS & EIGENVALUES CALCULATION

Determine the Eigenvalues of the 2 x 2 matrix  $A=\begin{pmatrix}2&-1\\-1&2\end{pmatrix}$  (give your answer as 1 x 2 matrix where each cell is an eigenvalue)

#### ✓ Detailed solution

An eigenvector v is a vector that when multiplied by a matrix A, is equal to a scalar  $\lambda$  times the eigenvector.

$$Av = \lambda v$$

The above equation will only have a solution if the characteristic equation is satisfied.

$$|A - \lambda I| = 0$$

Where I is the identity matrix and the vertical brackets refer to obtaining the determinant of the matrix. (We note that the determinant of any  $2 \times 2$  matrix is given by

$$\left( det \ M = |M| = |\begin{pmatrix} a & b \ c & d \end{pmatrix}| = ad - bc \right)$$

In the question where  $A=\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$  we can write the characteristic equation as:

$$|A - \lambda I| = |\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} - \lambda \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}| = 0$$

$$|\begin{pmatrix} 2 - \lambda & -1 \\ -1 & 2 - \lambda \end{pmatrix}| = 0$$

Obtaining the determinant ad - bc and still equating to zero

$$(2-\lambda)\cdot(2-\lambda)+1\cdot-1=0$$

$$\lambda^2 - (2+2) \cdot \lambda + 2 \cdot 2 + 1 \cdot -1 = 0$$

$$\lambda^2 - 4 \cdot \lambda + 3 = 0$$

In order to find the eigenvalues we must solve this quadratic equation for  $\lambda$  . The solution to any quadratic  $x^2+bx+c=0$  has the form

$$x = \frac{-b \pm \sqrt{b^2 - 4c}}{2}$$

Therefore the solutions to the characteristic equation are given by:

$$\lambda_{1,2} = \frac{-\left(-4\right)\pm\sqrt{4^2-43}}{2}$$

and we can obtain the values  $\lambda_1=3.00$  and  $\lambda_2=1.00$ 

#### 

An eigenvector v is a vector that when multiplied by a matrix A, is equal to a scalar  $\lambda$  times the eigenvector.

$$Av = \lambda v$$

The above equation will only have a solution if the characteristic equation is satisfied.

$$|A - \lambda I| = 0$$

Where I is the identity matrix and the vertical brackets refer to obtaining the determinant of the matrix. (We note that the determinant of any  $2 \times 2$  matrix is given by

$$\left( det \ M = |M| = |\begin{pmatrix} a & b \ c & d \end{pmatrix}| = ad - bc \right)$$

In the question where  $A=\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix}$  we can write the characteristic equation as:

$$|A-\lambda I|=|\begin{pmatrix}2&-1\-1&2\end{pmatrix}-\lambda\begin{pmatrix}1&0\0&1\end{pmatrix}|=0$$

$$|\begin{pmatrix} 2-\lambda & -1 \\ -1 & 2-\lambda \end{pmatrix}| = 0$$

Obtaining the determinant ad-bc and still equating to zero

$$(2-\lambda)\cdot(2-\lambda)+1\cdot-1=0$$

$$\lambda^2 - (2+2) \cdot \lambda + 2 \cdot 2 + 1 \cdot -1 = 0$$

$$\lambda^2 - 4 \cdot \lambda + 3 = 0$$

In order to find the eigenvalues we must solve this quadratic equation for  $\lambda$ . The solution to any quadratic  $x^2+bx+c=0$  has the form

$$x = \frac{-b \pm \sqrt{b^2 - 4c}}{2}$$

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $(3.00 \quad 1.00)$ 

100% of points

No specific feedback

**QUESTION** 

Having determined the Eigenvalues of the matrix, what are the associated Eigenvectors. (Give your answer as a 2 x2 matrix where each column is an Eigenvector, you can enter the values for the Eigenvalues in the following question)

#### Detailed solution

In order to obtain the associated Eigenvectors, we individually substitute the eigenvectors into the characteristic equation  $Av=\lambda v$  and solve for the eigenvector.

For the Eigenvalue  $\lambda_1=3.00$  the characteristic equation becomes

$$Av_1=\lambda_1v_1$$

$$\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} v_1 = 3.00 \cdot v_1$$

As the Eigenvector  $v_1$  is a vector quantity (a 2 x 1 matrix in this case), we will explicitly write it as such showing the two components.

$$\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} v_{11} \\ v_{12} \end{pmatrix} = \begin{pmatrix} 3.00 \cdot v_{11} \\ 3.00 \cdot v_{12} \end{pmatrix}$$

We can then matrix multiply the  $2\times 2$  and  $2\times 1$  matrices on left side of the equation (giving a  $2\times 1$  matrix).

$$\begin{pmatrix} 2 \cdot v_{11} - 1 \cdot v_{12} \ -1 \cdot v_{11} + 2 \cdot v_{12} \end{pmatrix} = \begin{pmatrix} 3.00 \cdot v_{11} \ 3.00 \cdot v_{12} \end{pmatrix}$$

This can be interpreted as two simultaneous equations

$$2 \cdot v_{11} - 1 \cdot v_{12} = 3.00 \cdot v_{11}$$
 and

$$-1 \cdot v_{11} + 2 \cdot v_{12} = 3.00 \cdot v_{12}$$

If we simplify the equations further, then we can write  $v_{11}$  in terms of  $v_{12}$  (we note that the same relation results from both equations)

$$v_{12} = \frac{3.00-2}{-1} v_{11} = -1.00 v_{11} \, {\rm or}$$

$$v_{12} = \frac{+1}{2-3.00}v_{11} = -1.00v_{11}$$

We can choose as an eigenvector any pair of values that satisfies the relation between  $v_{11}$  and  $v_{12}$ . The magnitude of eigenvectors are not important only the direction. However we typically chose values that give a magnitude equal to 1 i.e.  $\sqrt{v_{11}^2+v_{12}^2}=1$ 

We can calculate normalised eigenvectors by substitution into the magnitude equation.

$$1 = \sqrt{v_{11}^2 + v_{12}^2} = v_{11}^2 + (-1.00v_{11})^2$$

$$1 = (1 - 1.00^2) v_{11}^2$$

$$v_{11} = \sqrt{\frac{1}{(1 - 1.00^2)}} = 0.71$$

and therefor we can calculate  $v_{12}$  directly from the value of  $v_{11}$  using the relationships from earlier

$$v_{12} = -1.00 \cdot v_{11} = -0.71$$

We can also find the Eigenvectors for the vector  $v_2$ , in this case substituting into the characteristic equation gives:

$$\begin{pmatrix} 2 & -1 \ -1 & 2 \end{pmatrix} v_2 = 1.00 \cdot v_2$$

Again writing the eigenvector in terms of it's components  $v_2=\begin{pmatrix} v_{21} \\ v_{22} \end{pmatrix}$  the substituted characteristic equation becomes:

$$\begin{pmatrix} 2 & -1 \ -1 & 2 \end{pmatrix} \begin{pmatrix} v_{21} \ v_{22} \end{pmatrix} = \begin{pmatrix} 1.00v_{21} \ 1.00v_{22} \end{pmatrix}$$

We then perform the matrix multiplication on the left side of the equation

$$\begin{pmatrix} 2v_{21} - 1v_{22} \ -1v_{21} + 2v_{22} \end{pmatrix} = \begin{pmatrix} 1.00v_{21} \ 1.00v_{22} \end{pmatrix}$$

Again we are left with two simultaneous equations

$$2v_{21} - 1v_{22} = 1.00v_{21}$$

$$-1v_{21} + 2v_{22} = 1.00v_{21}$$

We can re-arrange the simultaneous equations which will yield the same relation between  $v_{21}$  and  $v_{22}$  in both cases.

$$v_{22}=\frac{1.00-2}{-1}v_{21}=1.00v_{21}$$
 or

$$v_{22} = \frac{+1}{2-1.00}v_{21} = 1.00v_{21}$$

We can chose any pair of values that satisfy the relationship, but we will select a pair that normalise to 1.

We can again choose as an eigenvector any pair of values that satisfies the relation between  $v_{21}$  and  $v_{22}$ . The magnitude of eigenvectors are not important only the direction. However we typically chose values that give a magnitude equal to 1 i.e.  $\sqrt{v_{21}^2+v_{22}^2}=1$ 

We can calculate normalised eigenvectors by substitution into the magnitude equation.

$$1 = \sqrt{v_{21}^2 + v_{22}^2} = v_{21}^2 + \left(1.00v_{21}\right)^2$$

$$1 = \left(1 + 1.00^2\right) v_{21}^2$$

$$v_{21} = \sqrt{\frac{1}{(1+1.00^2)}} = 0.71$$

and therefor we can calculate  $v_{12}$  directly from the value of  $v_{11}$  using the relationships from earlier

$$v_{22} = 1.00 \cdot v_{21} = 0.71$$

#### 

In order to obtain the associated Eigenvectors, we individually substitute the eigenvectors into the characteristic equation  $Av=\lambda v$  and solve for the eigenvector.

For the Eigenvalue  $\lambda_1=3.00$  the characteristic equation becomes

$$Av_1 = \lambda_1 v_1$$

$$\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} v_1 = 3.00 \cdot v_1$$

As the Eigenvector  $v_1$  is a vector quantity (a 2 x 1 matrix in this case), we will explicitly write it as such showing the two components.

$$\begin{pmatrix} 2 & -1 \\ -1 & 2 \end{pmatrix} \begin{pmatrix} v_{11} \\ v_{12} \end{pmatrix} = \begin{pmatrix} 3.00 \cdot v_{11} \\ 3.00 \cdot v_{12} \end{pmatrix}$$

We can then matrix multiply the  $2\times 2$  and  $2\times 1$  matrices on left side of the equation (giving a  $2\times 1$  matrix).

$$\begin{pmatrix} 2 \cdot v_{11} - 1 \cdot v_{12} \\ -1 \cdot v_{11} + 2 \cdot v_{12} \end{pmatrix} = \begin{pmatrix} 3.00 \cdot v_{11} \\ 3.00 \cdot v_{12} \end{pmatrix}$$

This can be interpreted as two simultaneous equations

$$2 \cdot v_{11} - 1 \cdot v_{12} = 3.00 \cdot v_{11}$$
 and

$$-1 \cdot v_{11} + 2 \cdot v_{12} = 3.00 \cdot v_{12}$$

If we simplify the equations further, then we can write  $v_{11}$  in terms of  $v_{12}$  (we note that the same relation results from both equations)

$$v_{12}=\frac{3.00-2}{-1}v_{11}=-1.00v_{11}$$
 or

$$v_{12} = \frac{+1}{2-3.00}v_{11} = -1.00v_{11}$$

We can choose as an eigenvector any pair of values that satisfies the relation between  $v_{11}$  and  $v_{12}$ . The magnitude of eigenvectors are not important only the direction. However we typically chose values that give a magnitude equal to 1 i.e.  $\sqrt{v_{11}^2+v_{12}^2}=1$ 

We can calculate normalised eigenvectors by substitution into the magnitude equation.

$$1 = \sqrt{v_{11}^2 + v_{12}^2} = v_{11}^2 + (-1.00v_{11})^2$$

$$1 = \left(1 - 1.00^2\right) v_{11}^2$$

$$v_{11} = \sqrt{\frac{1}{(1-1.00^2)}} = 0.71$$

and therefor we can calculate  $\it v_{12}$  directly from the value of  $\it v_{11}$  using the relationships from earlier

$$v_{12} = -1.00 \cdot v_{11} = -0.71$$

You can find the eigenvector corresponding to  $\lambda_2$  by following the same procedure

 $\boxed{\boxed}$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 0.71 & 0.71 \\ -0.71 & 0.71 \end{pmatrix}$ 

100% of points

No specific feedback

#### <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 107146

QUESTION MCQ PCA EIGENVECTORS/VALUES

In principle component analysis we multiply the original data by all the Eigenvectors. Assuming 2d data with covariance between the dimensions, after multiplication will the resulting data still be covariant or not covariant?

Select one answer. Upon selection it will submit your answer immediately.

✓ Feedback for answer: 1

QUESTION

# What does an eigenvalue represent in the context of an eigenvector?

|   | The angle between the eigenvector and the original corrdinate system                                        | n 🗙            |
|---|-------------------------------------------------------------------------------------------------------------|----------------|
|   | The variance of the data along the direction of each eigenvector                                            | $\checkmark$   |
|   | The determinant of the covariance matrix                                                                    | ×              |
|   | Select one answer. Upon selection it will submit your answer immediately.                                   |                |
| × | Feedback for answer: 1                                                                                      | 0% of points   |
|   | Incorrect: Eigenvalues do not represent angles                                                              |                |
| V | Feedback for answer: 2                                                                                      | 100% of points |
|   | Correct! The Eigenvalue represents the variance of the data alo direction of the associated eigenvector.    | ng the         |
| × | Feedback for answer: 3                                                                                      | 0% of points   |
|   | The determinant of the matrix is related to the product of its eigenvalues but is not an eigenvalue itself. |                |
|   |                                                                                                             |                |

QUESTION

In principle component analysis if the original data is multiplied by a subset of the eigenvectors, what will happen to the dimensionality of the data?

|          | O It is reduced                                                                                     | $\checkmark$   |
|----------|-----------------------------------------------------------------------------------------------------|----------------|
|          | it is increased                                                                                     | ×              |
|          | Select one answer. Upon selection it will submit your answer immediately.                           |                |
| <b>~</b> | Feedback for answer: 1                                                                              | 100% of points |
|          | Correct! We multiply by a subset of oigenvectors to obtain a                                        |                |
|          | Correct! We multiply by a subset of eigenvectors to obtain a compressed representation of the data. |                |
| ×        |                                                                                                     | 0% of points   |