<u>View subject</u> | <u>Edit subject</u>

# **Fourier Series**

Slides

# **Exercises**

#### <u>View exercise</u> | <u>Edit exercise</u>

## Exercise 123102

**QUESTION** 

Determine the Fourier series expansion of the periodic square wave with even symmetry

$$x\left( t\right) =\begin{cases} 1 & \left| t\right| <\pi/2 \ -1 & \pi/2 <\left| t\right| <\pi \end{cases}.$$

Given the Trigonometric form of the Fourier Series

$$x\left(t\right) = a_0 + \sum_{k=1}^{N} \left(a_k \cos\left(\frac{2 \ \pi k t}{T}\right) + b_k \sin\left(\frac{2 \pi k t}{T}\right)\right)$$

where the coefficients are given by  $a_{0}=\frac{1}{T}\int_{T}x\left( t\right) dt$  ,

$$a_k = \frac{2}{T} \int_T x\left(t\right) \, \cos\left(\frac{2\,\pi kt}{T}\right)$$
 and  $b_k = \frac{2}{T} \int_T x\left(t\right) \sin\left(\frac{2\pi kt}{T}\right)$ . Give your answer as a mathematical expression  $x\left(t\right) = \text{ in the box below.}$ 

$$\begin{array}{c|c} \hline \\ \hline \\ \hline \\ \hline \end{array} \end{array} \begin{array}{c} \times \end{array} \begin{array}{c} \blacksquare \Box \\ \hline \end{array} \begin{array}{c} \sqrt{\Box} \end{array} \begin{array}{c} \sqrt{\Box} \end{array} \begin{array}{c} \pi \\ \hline \end{array} \begin{array}{c} e^{\Box} \\ \hline \end{array} \begin{array}{c} \pm \\ \hline \end{array}$$

$$x\left(t\right) = \sum_{k=1}^{N} \left(\frac{4}{k\pi}\cos(kt)\right)$$

$$\boxed{\alpha\beta\infty} \quad \boxed{\sin\cos\left[\int\sum\prod\right] \left[ \cdot \binom{n}{k}\ln\right] \quad \boxed{\Box} \ x_{\Box}} \quad \left\{ \right\}, (,) \quad \boxed{=\neq\leq} \quad \boxed{\neg\vee} \quad \boxed{\left( \begin{matrix} \Box & \Box \\ \Box & \Box \end{matrix} \right) } \bullet \boxed{}$$

# Check my answer

#### ✓ Detailed solution

We will initially calculate the values for the  $a_k$  coefficients.

Firstly we calculate the  $a_0$  term which is the integral over one period  $T=2\pi$  of the waveform, we can break waveform this into 3 components (as the waveform is piecewise linear).

$$a_{0}=\frac{1}{T}\int_{T}x\left( t\right) dt=\frac{1}{2\pi}\int_{-\pi}^{-\pi/2}-1dt+\frac{1}{2\pi}\int_{-\pi/2}^{\pi/2}1dt\ +\frac{1}{2\pi}\int_{\pi/2}^{\pi}-1dt$$

$$a_0 = \frac{1}{\pi} - t \Big|_{-\pi}^{-\pi/2} + t \Big|_{-\pi/2}^{\pi/2} - t \Big|_{\pi/2}^{\pi} = \frac{1}{2\pi} (-\frac{\pi}{2} + \pi - \frac{\pi}{2}) = 0$$

As expected the  $a_0$  term evaluates to zero as the waveform is symmetrical about the x-axis (and the integral is the area under the line).

Secondly we calculate the  $a_k$ , again we will evaluate the integral over the 3 components.

$$a_k = \frac{2}{T} \int_T x\left(t\right) \, \cos\left(\frac{2 \, \pi k t}{T}\right)$$

$$a_k=\frac{1}{\pi}\int_{-\pi}^{-\pi/2}-\cos\left(kt\right)dt+\frac{1}{\pi}\int_{-\pi/2}^{\pi/2}\cos\left(kt\right)dt\ +\frac{1}{\pi}\int_{\pi/2}^{\pi}-\cos\left(kt\right)dt$$
 , the integral of  $\cos\left(kt\right)=\frac{\sin\left(kt\right)}{k}$ 

 $$$ \cline{-1}{\pi(kt)}_{k}\otimes_{|}_{-\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{-\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{-\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{-\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^{-\pi(kt)}_{k}\otimes_{|}_{\pi(kt)}^$ 

We know that  $\sin\left(\frac{k\pi}{2}\right)=1$  ,  $\sin\left(\frac{-k\pi}{2}\right)=-1$  and  $\sin\left(\pm k\pi\right)=0$  , therefore

$$a_k = \frac{1}{\pi} \left( \frac{1}{k} + \frac{1}{k} + \frac{1}{k} + \frac{1}{k} \right) = \frac{4}{k\pi}$$

We will calculate the  $b_k$  coefficients  $b_k=\frac{2}{T}\int_T x\left(t\right)\sin\left(\frac{2\pi kt}{T}\right)$ , once again breaking the waveform into components (as the waveform is piecewise liner),

$$b_k=\frac{1}{\pi}\int_{-\pi}^{-\pi/2}-\sin\left(kt\right)dt+\frac{1}{\pi}\int_{-\pi/2}^{\pi/2}\sin\left(kt\right)dt\ +\frac{1}{\pi}\int_{\pi/2}^{\pi}-\sin\left(kt\right)dt$$
 , the integral of  $\sin\left(kt\right)=\frac{-\cos\left(kt\right)}{k}$ 

\$

We know that  $\cos\left(\frac{\pm k\pi}{2}\right) \,= 0$  ,  $\cos\left(\pm\pi\right) \,= -1$  , therefore

 $\label{lem:lem:lem:lem:lem:lem:lem:lem:lem:lem:$ 

We observe that the  $b_k$  coefficients are all zero. Therefore for the square wave with even symmetry, we only have the cosine  $a_k$  coefficients.

The resulting Fourier Series approximation of the Square wave with even symmetry will be

$$x\left(t\right) = a_0 + \sum_{k=1}^{N} \left(a_k \cos\left(\frac{2 \ \pi k t}{T}\right) + b_k \sin\left(\frac{2 \pi k t}{T}\right)\right)$$

$$x\left(t\right) = \sum_{k=1}^{N} \left(\frac{4}{k\pi}\cos\left(\frac{2\pi kt}{T}\right)\right) = \sum_{k=1}^{N} \left(\frac{4}{k\pi}\cos\left(kt\right)\right)$$

#### Question at wrong answer

Calculate the values for the  $a_k$  and  $b_k$  coefficients using a piecewise linear approximation of the integrals. For example for the  $a_k$  term we obtain the integral. (Note that  $T=2\pi$ )

$$a_k = \; \frac{1}{\pi} \int_{-\pi}^{-\pi/2} -\cos{(kt)} \, dt + \frac{1}{\pi} \int_{-\pi/2}^{\pi/2} \cos{(kt)} \, dt \; + \frac{1}{\pi} \int_{\pi/2}^{\pi} -\cos{(kt)} \, dt$$

We note that the  $a_0$  term should evaluate to zero as the waveform is symmetrical about the x-axis. Also the  $b_k$  terms should evaluate to zero for this specific case of the Square wave with even symmetry (Note that it is the opposite for the Square wave with odd symmetry, where the  $a_k$  coefficients are zero).

After finding the values of the  $a_k,b_k$  coefficients the approximation is obtained by substituting into the expression for

$$x\left(t\right) = a_0 + \sum_{k=1}^{N} \left(a_k \cos\left(\frac{2 \, \pi k t}{T}\right) + b_k \sin\left(\frac{2 \pi k t}{T}\right)\right)$$

 $\bigspace$  Feedback when answer is algebraically Equivalent to  $x\left(t\right)=\sum_{k=1}^{N}\left(\frac{4}{k\pi}\cos\left(kt\right)\right)$ 

100% of points

No specific feedback

View exercise | Edit exercise

16/06/2026, 10:32 Grasple

### Exercise 123934

QUESTION

Determine the Fourier series expansion of the periodic square wave with even symmetry  $x(t) = \frac{t}{2}$ .

Given the Trigonometric form of the Fourier Series

$$x\left(t\right) = a_0 + \sum_{k=1}^{N} \left(a_k \cos\left(\frac{2 \, \pi k t}{T}\right) + b_k \sin\left(\frac{2 \pi k t}{T}\right)\right)$$

where the coefficients are given by  $a_{0}=\frac{1}{T}\int_{\mathbb{T}}x\left( t\right) dt$  ,

$$a_k = \frac{2}{T} \int_T x\left(t\right) \, \cos\left(\frac{2 \, \pi k t}{T}\right) \, \text{and}$$

$$b_k = \frac{2}{T} \int_T^{\bar{t}} x\left(t\right) \sin\left(\frac{2\pi kt}{T}\right)$$
 . Give your answer as a

mathematical expression x(t) = in the box below.

$$x\left(t\right) \ = \ \sum_{k=1}^{N} \left(-\,\frac{2}{\pi k} \left(-1\right)^k \sin(kt)\right)$$

## Check my answer

#### ✓ Detailed solution

As this function demonstrates odd symmetry x(t) = -x(-t) we can avoid calculating the  $a_0$  and  $a_k$  terms as these will be zero (as in the case of the periodic step function with odd symmetry).

We will calculate the  $b_k$  coefficients  $b_k = \frac{2}{T} \int_T x\left(t\right) \sin\left(\frac{2\pi kt}{T}\right)$ , as the sawtooth waveform is a straight line we can integrate from  $-\pi$  to  $\pi$ 

 $b_k=\frac{1}{\pi}\int_{-\pi}^\pi\frac{t}{\pi}\sin\left(kt\right)dt$  , evaluating this integral using the product rule of integration.

We can solve this integral using the product rule of integration is  $\int u \frac{dv}{dt} dt = uv + \int v \frac{du}{dt} dt.$ 

we let 
$$u=\frac{t}{\pi}$$
 and  $\frac{du}{dt}=\frac{1}{\pi}$  and let  $\frac{dv}{dt}=\sin{(kt)}$  and therefore  $v=\int\frac{dv}{dt}dt=\int\sin{(kt)}=-\frac{\cos{(kt)}}{k}$ .

Therefore the result is

$$uv+\int v\frac{du}{dt}dt=-\frac{t}{\pi}\frac{\cos{(kt)}}{k}+\int \frac{\cos{(kt)}}{k}\frac{1}{\pi}dt=-\frac{t}{k\pi}\cos{(kt)}+\frac{1}{k^2\pi}\sin{(kt)}$$

We substitute the result of the integral back into the expression for  $b_k$  and evaluate this over the range  $-\pi$  to  $\pi$ 

 $$$ (\displaystyle b_{k}=\frac{1}{\pi c_{1}}\left(-\frac{t} {k\pi_{c_{1}}\left(\frac{1}{\pi c_{t}} {\kappa^2\pi_{c_{1}}\left(\frac{1}{\pi c_{t}}\right)\right)} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}} \\ (\displaystyle b_{k}=\frac{1}{\pi c_{1}$ 

Expanding the above expression

$$b_k = -\frac{1}{\pi k}\cos\left(k\pi\right) + \frac{1}{\pi k}\cos\left(-k\pi\right) + \frac{1}{k^2\pi^2}\sin\left(k\pi\right) - \frac{1}{k^2\pi^2}\sin\left(-k\pi\right)$$

We can simplify this expression using the trigonometric identities  $\sin{(-a)} = -\sin{(a)}$  and  $\cos{(a)} = \cos{(-a)}$ 

$$b_k = -\frac{2}{\pi k}\cos\left(k\pi\right) + \frac{2}{k^2\pi}\sin\left(k\pi\right)$$

We can simplify this further noting that  $\sin{(k\pi)} = 0$  for all k and  $\cos{(k\pi)} = 1$  when k is even and  $\cos{(k\pi)} = -1$  when k is odd.

Therefore  $b_k$  coefficients are written as

$$b_k = -\frac{2}{\pi k} \left(-1\right)^k$$

The resulting Fourier Series approximation of the Sawtooth wave with even symmetry will be

$$x\left(t\right)=a_{0}+\sum_{k=1}^{N}\left(a_{k}\cos\left(\frac{2\,\pi kt}{T}\right)+b_{k}\sin\left(\frac{2\pi kt}{T}\right)\right)$$
 which simplifies to

$$x\left(t\right) = \sum_{k=1}^{N} \left(-\frac{2}{\pi k} \left(-1\right)^{k} \sin\left(kt\right)\right)$$

We can verify our derivation by obtaining the individual fourier series (fixed k) over a range of t values and summing.

#### Question at wrong answer

As this function demonstrates odd symmetry x(t) = -x(-t) we can avoid calculating the  $a_0$  and  $a_k$  terms as these will be zero (as in the case of the periodic step function with odd symmetry).

We will calculate the  $b_k$  coefficients  $b_k = \frac{2}{T} \int_T x\left(t\right) \sin\left(\frac{2\pi kt}{T}\right)$ , as the sawtooth waveform is a straight line we can integrate from  $-\pi$  to  $\pi$ 

Hint: We can solve this integral using the product rule of integration is  $\int u \frac{dv}{dt} dt = uv + \int v \frac{du}{dt} dt \,.$ 

 $\bigspace{ }$  Feedback when answer is algebraically Equivalent to  $x\left(t\right) = \sum_{k=1}^{N} \left(-\frac{2}{\pi k} \left(-1\right)^k \sin\left(kt\right)\right)$ 

100% of points

No specific feedback