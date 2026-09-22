<u>View subject</u> | <u>Edit subject</u>

# Generative Models

Slides

## **Exercises**

## <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 115363

**QUESTION** 

Given the gaussian distributed samples

$$X_n =$$

$$\{-6.51, -1.26, -0.78, 2.23, 4.74, 6.05, -1.11, 6.34, -1.12, 8, \mathcal{N}(\mu, \sigma)\}$$

determine the parameters  $\theta = (\mu, \sigma)$  of the distribution using maximum likelihood estimation. In this first part of the question determine the value of the  $\mu$ and give your answer in the box below.

# Check my answer

#### ✓ Detailed solution

The goal is to find the maximum of the likelihood i.e. determine the set of parameters that best fit the samples to the current gaussian model with values  $\theta=(\mu,\sigma)$ . There is good amount of theory in this explanation, but the calculations are is simple and intuitive.

For each sample the conditional probability of the sample belonging to the distribution is given by  $P\left(x_n\left|\theta\right|\right)=\frac{1}{\sigma\sqrt{2\pi}}e^{\left\{-\frac{1}{2}\left(\frac{x_n-\mu}{\sigma}\right)^2\right\}}$ 

To get the probability (or likelihood) that the set of samples fit the model we obtain the product of the individual probabilities.

$$P\left(X\left|\theta\right|\right) = \mathcal{L}\left(X\left|\theta\right|\right) = \prod_{i=1}^{n} P\left(x_{n}\left|\theta\right|\right) = \prod_{i=1}^{n} \frac{1}{\sigma\sqrt{2\pi}}e^{\left\{-\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}\right\}}$$

This equation has the product term which is difficult to work with so we obtain the log likelihood. To go from the previous equation to the one below we use the product rule of logs  $\log{(mn)} = \log{(m)} + \log{(n)}$  which allows us to replace the product with a sum.

$$log \; \mathcal{L}\left(X\left|\theta\right|\right) = \sum_{i=1}^{n} \log \; \left(\frac{1}{\sigma\sqrt{2\pi}}e^{\left\{-\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}\right\}\right)}$$

We then apply the product rule again to split the exponent term from the constant term  $\frac{1}{\sigma\sqrt{2\pi}}$  . We note the exponent and log terms cancel after this split.

$$\log\mathcal{L}\left(X\left|\theta\right|\right)=\sum_{i=1}^{n}\log\;\frac{1}{\sigma\sqrt{2\pi}}+\sum_{i=1}^{n}-\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}$$
 (1)

Recognising that  $\log \frac{1}{\sigma \sqrt{2\pi}} = \log \frac{1}{\sqrt{\sigma^2 2\pi}} = \log \left(2\pi\sigma^2\right)^{-\frac{1}{2}} = -\frac{1}{2}\log \left(2\pi\sigma^2\right)$  we can apply the power law of logs  $\log m^k = k\log m$  to simplify the 1st term of equation 1. We can also remove the sum (in the first term), since we obtain the sum of a constant, this can be replaced by multiplication by n.

$$log~\mathcal{L}\left(X\left|\theta\right|\right) = -\frac{n}{2}\log\left(2\pi\sigma^{2}\right) + \sum_{i=1}^{n} -\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}$$

Typically we want to maximise the log likelihood, but we can instead minimise it if we multiply the equation above by -1

$$argmin\ log\ \mathcal{L}\left(X\left|\theta\right|\right)=\frac{n}{2}\log\left(2\pi\sigma^{2}\right)+\sum_{i=1}^{n}\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}$$
 (2)

We will first consider minimising with respect to the  $\mu$ term. We do this by finding where the partial derivative is zero  $\frac{\delta}{\delta\mu}=0$ .

The derivative of the first term  $\frac{n}{2}\log\left(2\pi\sigma^2\right)$  will be zero as it is a constant. We can differentiate the 2nd term (equ. 2) using the chain rule  $\frac{\delta y}{\delta \mu} = \frac{\delta y}{\delta u} \cdot \frac{\delta u}{\delta \mu}$  for differentiation where  $u = x_n - \mu$  and  $y = (x_n - \mu)^2 = u^2$  therefore.

$$\frac{\delta}{\delta\mu}\sum_{i=1}^n\frac{1}{2}\left(\frac{x_n-\mu}{\sigma}\right)^2=\frac{1}{\sigma^2}\sum_{i=1}^n-(x_n-\mu)=0$$

Therefore we get that  $\sum_{i=1}^n x_n = n\mu$  or  $\mu = \frac{\sum_{i=1}^n x_n}{n}$  which is just the mean of the samples which is intuitively what we expect.

We calculate

$$\mu = \frac{-6.51 - 1.26 - 0.78 + 2.23 + 4.74 + 6.05 - 1.11 + 6.34 - 1.12 + 8.62}{10}$$

$$\mu = 1.72$$

## 

The goal of this exercise is to learn how the maximum likelihood estimation approach works by analysing a simple case where we have 1 dimensional data and a simple gaussian model with 2 parameters.

Ultimately the approach produces a formulation which is intuitively what you would expect. The mean of the distribution may be estimated directly from the samples.

✓ Feedback when answer is equal to 1.72

100% of points

QUESTION

Determine the standard deviation using the maximum likelihood approach assuming the same set of samples.

# Check my answer

☑ Detailed solution

For the variance or  $v=\sigma^2$  term we, also again obtain the partial derivative and set it equal to zero at the minimum. This time both terms in the likelihood

$$argmin\ log\ \mathcal{L}\left(X\left|\theta\right|\right)=\frac{n}{2}\log\left(2\pi\sigma^{2}\right)+\sum_{i=1}^{n}\frac{1}{2}\left(\frac{x_{n}-\mu}{\sigma}\right)^{2}$$
 will have non zero values. This derivation is simpler if we replace  $\sigma^{2}=v$  and split

the 1st log term with the product rule.

$$\begin{split} \frac{\delta}{\delta v}\left(\frac{n}{2}\log\left(2\pi v\right) + \sum_{i=1}^{n}\frac{1}{2}\frac{\left(x_{n}-\mu\right)^{2}}{v}\right) = \ \frac{\delta}{\delta v}\left(\frac{n}{2}\log\left(2\pi\right) + \frac{n}{2}\log\left(v\right) + \sum_{i=1}^{n}\frac{1}{2}\frac{\left(x_{n}-\mu\right)^{2}}{v}\right) = 0 \end{split}$$

The derivative of the constant term  $\frac{\delta}{\delta v}\left(\frac{n}{2}\log\left(2\pi\right)\right)=0$  is zero and the derivative of a log is well known  $\frac{\delta\log\left(x\right)}{\delta x}=\frac{1}{x}$  therefore  $\frac{\delta}{\delta v}\left(\frac{n}{2}\log\left(v\right)=\frac{n}{2}\frac{1}{v}\right)$  and the third term is evaluated as

Grasple

$$\frac{\delta}{\delta v}\left(\sum_{i=1}^{n}\frac{1}{2}\frac{\left(x_{n}-\mu\right)^{2}}{v}\right)=\frac{1}{2}\sum_{i=1}^{n}\left(x_{n}-\mu\right)^{2}\frac{\delta}{\delta v}\left(\frac{1}{v}\right)=-\frac{1}{2}\sum_{i=1}^{n}\frac{\left(x_{n}-\mu\right)^{2}}{v^{2}}$$

Therefore we get (remember to equate to zero as we want minimum)

$$\frac{\delta}{\delta v} \left( \frac{n}{2} \log \left( 2 \pi v \right) + \sum_{i=1}^n \frac{1}{2} \frac{\left( x_n - \mu \right)^2}{v} \right) = \frac{n}{2} \frac{1}{v} - \frac{1}{2} \sum_{i=1}^n \frac{\left( x_n - \mu \right)^2}{v^2} = 0$$

Rearranging the equation:  $\frac{n}{2}\frac{1}{v}=\frac{1}{2}\sum_{i=1}^n\frac{\left(x_n-\mu\right)^2}{v^2}$  and this can be simplified to  $v=\frac{1}{n}\sum_{i=1}^n\left(x_n-\mu\right)^2$ .

In this equation we substitute the mean value we determined earlier and sum over the samples to determine the variance or standard deviation.

$$v = \frac{1}{n} \sum_{i=1}^{n} (x_n - \mu)^2 = \frac{1}{10} \left( (-6.51 - 1.72)^2 + (-1.26 - 1.72)^2 + (-0.78 - 1.72)^2 + (2.23 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72)^2 + (-0.78 - 1.72$$

$$v = \frac{1}{n} \sum_{i=1}^{n} (x_n - \mu)^2 = 19.60$$

The standard deviation will be the square root of the variance

$$\sigma = \sqrt{v} = 4.43$$

#### Question at wrong answer

As with the mean we find in this case of a simple gaussian that the maximum likelihood estimation approach produces a simple formulation for finding the standard deviation or variance.

☑ Feedback when answer is equal to 4.43

100% of points

## Exercise 130560

**QUESTION** 

Given a data set  $x=\{9.2,11.47\}$  which is drawn from a Gaussian Distribution  $\mathcal{N}(\theta)$  where the set of parameters for the distribution  $\theta=(\mu,\sigma)=(8.9,-1.9)$ . Given the probability density function for the gaussian  $P\left(x\left|\theta\right|\right)=\frac{1}{\sqrt{2\pi\sigma^2}}e^{\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2\right)}$  calculate the total Log Likelihood  $\sum_{n=1}^{N}\log P\left(x_n|\theta\right)$  for the dataset.

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| -4.09                                                  |

## Check my answer

#### ✓ Detailed solution

In the question we are given a dataset of two points  $x=\{9.2,11.47\}$  , which are part of the distribution with parameters  $\theta=(\mu,\sigma)=(8.9,-1.9)$  .

We calculate the probability for each point in the dataset using  $P\left(x\left|\theta\right.\right)=\frac{1}{\sqrt{2\pi\sigma^{2}}}e^{\left(-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right)} \text{ and then sum the log of the results to obtain the Likelihood } \sum_{n=1}^{N}\log P\left(x_{n}|\theta\right).$ 

For the first point  $x_1 = -0.17$ :

$$P\left(9.2\left|\theta\right)=\frac{1}{\sqrt{2\pi\left(-1.9^{2}\right)}}e^{\left(-\frac{1}{2}\left(\frac{9.2-8.9}{-1.9}\right)^{2}\right)}=0.21$$

For the second point  $x_2 = -1.35$ :

$$P\left(11.47\left|\theta\right)=\frac{1}{\sqrt{2\pi\left(-1.9^{2}\right)}}e^{\left(-\frac{1}{2}\left(\frac{11.47-8.9}{-1.9}\right)^{2}\right)}=0.08$$

We then sum the natural logarithm of the values

$$\log \mathcal{L}\left(x\right) = \sum_{n=1}^{N} \log P\left(x_{n} | \theta\right) = \log \left(P\left(9.2 \left| \theta\right)\right) + \log \left(P\left(11.47 \left| \theta\right)\right)$$

$$\log \mathcal{L}(x) = -1.56 - 2.53 = -4.09$$

No specific feedback

✓ Feedback when answer is equal to -4.09

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 115439

**QUESTION** 

The Kullback Leibler Divergence between two distributions  $KL\left[q\left(x\right)\parallel p\left(x\right)\right]$  has a closed form solution for the case when the input is two gaussians  $q\left(x\right)\sim\mathcal{N}\left(\mu_{q},\sigma_{q}\right)$  and  $p\left(x\right)\sim\mathcal{N}\left(\mu_{p},\sigma_{p}\right)$ . Determine the divergence between  $q\left(x\right)$  and  $p\left(x\right)$  given  $\mu_{q}=2.7$ ,  $\sigma_{q}=0.9$ ,  $\mu_{p}=2.3$ ,  $\sigma_{p}=0.4$  and the closed form of the Kullback Leibler Divergence:

$$KLD = \log \; \frac{\sigma_p}{\sigma_q} + \frac{\left(\mu_q - \mu_p\right)^2 + \sigma_q^2}{2\sigma_p^2} - \frac{1}{2}$$

1.72

Check my answer

#### ✓ Detailed solution

We can easily obtain the Kullback-Leibler Divergence using the closed form given and substituting the values of  $\mu_q, \sigma_q, \mu_p, \sigma_p$ 

$$KLD = \log \ \frac{\sigma_p}{\sigma_q} + \frac{\left(\mu_q - \mu_p\right)^2 + \sigma_q^2}{2\sigma_p^2} - \frac{1}{2} = \log \frac{0.4}{0.9} + \frac{\left(2.7 - 2.3\right)^2 + 0.9^2}{2 \cdot 0.4^2} - \frac{1}{2}$$

$$KLD = 1.72$$

#### Question at wrong answer

We can easily obtain the Kullback-Leibler Divergence using the closed form given and substituting the values of  $\mu_q, \sigma_q, \mu_p, \sigma_p$ 

$$KLD = \log \ \frac{\sigma_p}{\sigma_q} + \frac{\left(\mu_q - \mu_p\right)^2 + \sigma_q^2}{2\sigma_p^2} - \frac{1}{2} = \log \frac{0.4}{0.9} + \frac{\left(2.7 - 2.3\right)^2 + 0.9^2}{20.4^2} - \frac{1}{2}$$

☑ Feedback when answer is equal to 1.72

100% of points

## Exercise 130576

**QUESTION** 

A Variational Autoencoder (VAE) uses a latent space of dimension  $D_z=2$ . The prior P(z) is a standard normal distribution  $\mathcal{N}(0,I)$ . For a specific input image, the Encoder outputs the following parameter values for the approximate posterior q(z|x): Mean vector  $\mu=[2,7]$  and diagonal covariance matrix  $\Sigma=\begin{pmatrix} 9 & 0 \\ 0 & 2 \end{pmatrix}$ 

Calculate the KL Divergence using the form  $\mathcal{L_{KL}} = 0.5 \left(Tr\left[\Sigma\right] + \mu^T \mu - D_z - \log\left(det\left[\Sigma\right]\right) \text{, give your}$  answer in the box below.

## Check my answer

#### ✓ Detailed solution

We can calculate the required terms in the  $\mathcal{L}_{\mathcal{KL}}$  equation.

The Trace of the covariance matrix is the sum of the diagonal elements

$$Tr\left[\Sigma\right] = 9 + 2 = 11$$

The Dot product (inner product) of the means is given by

$$\mu^T\mu = \begin{pmatrix} 2 & 7 \end{pmatrix} \times \begin{pmatrix} 2 \ 7 \end{pmatrix} = 53$$

The  $D_z$  is the dimensionality of the latent space which is 2

The determinant of the covariance matrix is the product of the diagonal elements

 $det\left[\Sigma\right]=9\times2=18$  , obtaining the natural log of this gives  $Log\left(det\left[\Sigma\right]\right)=2.89$ 

Finally summing the terms gives:

$$\mathcal{L_{KL}} = 0.5\left(Tr\left[\Sigma\right] + \mu^T\mu - D_z - \log\left(det\left[\Sigma\right]\right) = 0.5\left(11 + 53 - 2 - 2.89\right) = 2.89$$

## □ Question at wrong answer

Hints:

For a diagonal matrix the Trace is the sum of the diagonal elements  $Tr\left[\Sigma\right]=9+2$  and the Determinant  $det\left[\Sigma\right]=9\times2$  is the product of the diagonal elements.

The Dot product (or innner product) is given by which  $\mu^T \mu = \begin{pmatrix} \mu_0 & \mu_1 \end{pmatrix} \times \begin{pmatrix} \mu_0 \\ \mu_1 \end{pmatrix}$  returns a scalar value.

✓ Feedback when answer is equal to 29.56

100% of points

## Exercise 115440

QUESTION

Assuming a GAN with a Discriminator (parameters  $\phi$ ) and Generator (with parameters  $\theta$ .)

Determine the value of the Discriminator loss for the current batch assuming there are n=5 images samples  $\hat{x}_j$  from the generator  $\hat{x}_j=g\left[z_j,\theta\right]$ , producing values at the output of the discriminator  $y_j=d\left[\hat{x}_j,\phi\right]$  of  $y_j=\{0.63,0.03,0.12,0.78,0.73\}$  (where  $z_j$  are the random latent space values at the generator input). Secondly we assume there are m=5 samples  $x_i$  input from the training distribution, producing values of at the output of the training discriminator  $y_i=d\left[x_i,\phi\right]$  where  $y_i=\{0.41,0.13,0.50,0.97,0.73\}$ . Give a value for the discriminator loss in the text box below.

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| 7.93                                                   |
| Check my answer                                        |

Detailed solution

From the notes we can see that the loss for a discriminator is given by

$$\mathcal{L}\left[\phi\right] = \sum_{j=0}^{n-1} -\log\left(1-d\left[g\left[z_{j},\theta\right],\phi\right]\right) - \sum_{i=0}^{m-1}\log\left(d\left[x_{i},\phi\right]\right)$$

This loss (as discussed in the notes) is based on a binary cross entropy loss and the term  $d\left[x_i,\phi\right]=y_i$  is the output of the discriminator with the inputs  $x_i$  from true distribution. The term  $d\left[g\left[z_j,\theta\right],\phi\right]=y_j$  is the output of the discriminator with samples from the generator  $\hat{x}_j=g\left[z_j,\theta\right]$ .

We can simply calculate the loss by substituting the values for  $y_i$  and  $y_j$  in the loss equation.

For the first term in the loss:

$$\begin{aligned} \sum_{j=0}^{n-1} -\log\left(1-d\left[g\left[z_{j},\theta\right],\phi\right]\right) &= \sum_{j=0}^{n-1} -\log\left(1-y_{j}\right) \ &= \end{aligned}$$

$$-\log (1 - 0.63) - \log (1 - 0.03) - \log (1 - 0.12) - \log (1 - 0.78) - \log (1 - 0.78)$$

$$= +0.99 + 0.03 + 0.13 + 1.51 + 1.31 = 3.97$$

For the second term in the loss:

$$\begin{split} &-\sum_{i=0}^{m-1}\log\left(d\left[x_{i},\phi\right]\right)\\ &=-\log\left(0.41\right)-\log\left(0.13\right)-\log\left(0.50\right)-\log\left(0.97\right)-\log\left(0.73\right)\\ &=+0.89+2.04+0.69+0.03+0.31=3.96 \end{split}$$

The final loss for the discriminate is the sum of the two values we have obtained above  $\mathcal{L}~[\phi]=3.97+3.96=7.93$ 

From the notes we can see that the loss for a discriminator is given by

$$\mathcal{L}\left[\phi\right] = \sum_{j=0}^{n-1} -\log\left(1-d\left[g\left[z_{j},\theta\right],\phi\right]\right) - \sum_{i=0}^{m-1}\log\left(d\left[x_{i},\phi\right]\right)$$

This loss (as discussed in the notes) is based on a binary cross entropy loss and the term  $d\left[x_i,\phi\right]=y_i$  is the output of the discriminator with the inputs  $x_i$  from true distribution. The term  $d\left[g\left[z_j,\theta\right],\phi\right]=y_j$  is the output of the discriminator with samples from the generator  $\hat{x}_j=g\left[z_j,\theta\right]$ .

We can simply calculate the loss by substituting the values for  $y_i$  and  $y_j$  in the loss equation.

✓ Feedback when answer is equal to 7.93

100% of points

## Exercise 115570

QUESTION

Given that the Loss function for the Variational Autoencoder

$$\mathcal{L}\left(\theta,\phi,x\right)\approx log\ P\left(x|\hat{z},\theta\right)-D_{KL}\left[q\left(z|x,\phi\right)||p\left(z\right)\right]$$
 . The term representing the reconstruction loss is assumed to be given by the Bernolli distribution

$$log \ P\left(x | \hat{z}, \theta\right) = \sum_{i=1}^{D} x_i \log y_i + (1-x_i) \log \left(1-y_i\right)$$
 where

the output dimension size D=4. The KL divergence term is given by

$$D_{KL}\left[q\left(z|x,\!\phi\right)\left\|p\left(z\right)\right] \ = -\frac{1}{2}\sum_{j=1}^{J}\left(1+\log\left(\sigma_{j}^{2}\right)-u_{j}^{2}-\sigma_{j}^{2}\right)$$

for the case where the latent space is a simple prior  $p\left(z\right)\sim\mathcal{N}\left(0,1\right)$  with latent space size J=2.

Determine the value of the loss for the case where the VAE generates latent space values  $\sigma_j=\{0.84,3.79\}$ ,  $u_j=\{1.71,5.0\}$  for a single input. The output predictions generated by the decoder are given by  $y_i=\{0.15,0.05,0.56,0.85\}$  and the values of the input

binary data are  $x_i = \{0,1,0,1\}$  . (Give your answer in the text box below)

## Check my answer

#### ✓ Detailed solution

We can simply determine the loss by substituting the relevant terms into the loss equations.

For the reconstruction loss term:

$$log \ P \left( x | \hat{z}, \theta \right) = \sum_{i=1}^{D} x_i \log y_i + (1 - x_i) \log \left( 1 - y_i \right) = 0 \log 0.15 + \\ (1 - 0) \log \left( 1 - 0.15 \right) \\ + 1 \log 0.05 + (1 - 1) \log \left( 1 - 0.05 \right) + 0 \log 0.56 + (1 - 0) \log \left( 1 - 0.56 \right) \\ + 1 \log 0.85 + (1 - 1) \log \left( 1 - 0.85 \right) \\ = 0.00 - 0.16 - 3.00 + 0.00 + 0.00 - 0.82 - 0.16 + 0.00 = -4.14$$

For the Kullback Leibler Divergence term: (note that J=2) so we will sum over

$$\begin{aligned} D_{KL}\left[q\left(z|x,\phi\right) \| p\left(z\right)\right] &= -\frac{1}{2} \sum_{j=1}^{J} \left(1 + \log\left(\sigma_{j}^{2}\right) - u_{j}^{2} - \sigma_{j}^{2}\right) \ &= -\frac{1}{2} \left(1 + \log\left(0.84^{2}\right) - 1.71^{2} - 0.84^{2}\right) - \frac{1}{2} \left(1 + \log\left(3.79^{2}\right) - 5.0^{2} - 3.79^{2}\right) \ &= 1.49 + 17.85 = 19.34 \end{aligned}$$

The value for the loss  $\mathcal{L}\left(\theta,\phi,x\right)\approx\log P\left(x|\hat{z},\theta\right)-D_{KL}\left[q\left(z|x,\phi\right)\|p\left(z\right)\right]$  is given by the sum of the two terms.

$$= -4.14 - 19.34 = -23.48$$

#### □ Question at wrong answer

We can simply determine the loss by substituting the relevant terms into the loss equations and summing the result.

For the reconstruction loss term:

$$log \ P\left(x | \hat{z}, \theta\right) = \sum_{i=1}^{D} x_i \log y_i + (1-x_i) \log \left(1-y_i\right)$$
 and Kull Leibler

$$\mathsf{Divergence}\,\,D_{KL}\left[q\left(z|x,\!\phi\right)\|p\left(z\right)\right]\,=-\frac{1}{2}\sum_{j=1}^{J}\left(1+\log\left(\sigma_{j}^{2}\right)-u_{j}^{2}-\sigma_{j}^{2}\right)$$

$$\mathcal{L}\left(\theta,\phi,x\right)\approx log\ P\left(x|\hat{z},\theta\right)-D_{KL}\left[q\left(z|x,\phi\right)\left\|p\left(z\right)\right]$$

✓ Feedback when answer is equal to -23.48

100% of points