<u>View subject</u> | <u>Edit subject</u>

# Vision Transformers

Slides

# **Exercises**

### <u>View exercise</u> | <u>Edit exercise</u>

# Exercise 113676

QUESTION

A self-attention block in a transformer uses input tokens of dimension D=4. The block has learned key, query and values weight matrices

$$W_K = \begin{pmatrix} 3 & 2 & 0 & 4 \ 3 & 4 & 3 & 0 \ 0 & 3 & 3 & 1 \ 1 & 3 & 0 & 1 \end{pmatrix}$$
 ,  $W_Q = \begin{pmatrix} 2 & 1 & 2 & 1 \ 1 & 3 & 4 & 4 \ 0 & 5 & 0 & 3 \ 0 & 4 & 4 & 2 \end{pmatrix}$  and  $W_V = \begin{pmatrix} 3 & 2 & 5 & 1 \ 4 & 2 & 5 & 3 \ 1 & 4 & 0 & 1 \ 3 & 1 & 4 & 1 \end{pmatrix}$  . Firstly Determine the Key Query

and Value tokens, given the Input token  $X = \begin{pmatrix} 1 \\ 1 \end{pmatrix}$ .

(Give your answer as a  $D \times 3 = 4 \times 3$  matrix where each column is the the Key, Query and Value token obtained from the input token)

#### ✓ Detailed solution

In order to obtain the Key, Query and Value tokens we simply matrix multiply the corresponding weight matrix  $W_K, W_Q$  or  $W_V$  by the input token X. The image below shows matrix multiplication of the input tokens X by the key weights.

$$D \qquad \qquad X \qquad D \qquad X \qquad M \qquad \qquad M \qquad M \qquad M \qquad M \qquad M \qquad M \qquad M \qquad$$

We note that the  $W_K$  matrix is of dimension  $D \times D$  and corresponds to a fully connected neural network (as shown below). Each input (elements of token) is connected to each neuron, therefore there are D=4 weights corresponding to each of inputs. There are D=4 neurons as in the transformer self-attention block, as we want the output token size to remain the same as the input.

For this questions we calculate:

$$K=W_K\times X=\begin{pmatrix} 3 & 2 & 0 & 4 \ 3 & 4 & 3 & 0 \ 0 & 3 & 3 & 1 \ 1 & 3 & 0 & 1 \end{pmatrix}\times\begin{pmatrix} 2 \ 1 \ 2 \ \end{pmatrix}=\begin{pmatrix} 16 \ 13 \ 8 \ 7 \end{pmatrix}$$

$$Q=W_Q\times X=\begin{pmatrix} 2&1&2&1\ 1&3&4&4\ 0&5&0&3\ 0&4&4&2 \end{pmatrix}\times\begin{pmatrix} 2\ 1\ 1\ 2 \end{pmatrix}=\begin{pmatrix} 9\ 17\ 11\ 12 \end{pmatrix}$$

$$V=W_V\times X=\begin{pmatrix} 3 & 2 & 5 & 1 \ 4 & 2 & 5 & 3 \ 1 & 4 & 0 & 1 \ 3 & 1 & 4 & 1 \end{pmatrix}\times\begin{pmatrix} 2 \ 1 \ 1 \ 2 \end{pmatrix}=\begin{pmatrix} 15 \ 21 \ 8 \ 13 \end{pmatrix}$$

## Question at wrong answer

In order to obtain the Key, Query and Value tokens we simply matrix multiply the corresponding weight matrix  $W_K, W_Q$  or  $W_V$  by the input token X. The image below shows matrix multiplication of the input tokens X by the key weights.

$$D \qquad \qquad X \qquad D \qquad X \qquad M \qquad \qquad M \qquad M \qquad M \qquad M \qquad M \qquad M \qquad M \qquad$$

We note that the  $W_K$  matrix is of dimension  $D \times D$  and corresponds to a fully connected neural network (as shown below). Each input (elements of token) is connected to each neuron, therefore there are D=4 weights corresponding to each of inputs. There are D=4 neurons as in the transformer self-attention block, as we want the output token size to remain the same as the input.

 $\begin{tabular}{|c|c|c|c|c|} \hline \checkmark & Feedback when answer is algebraically Equivalent to & \begin{pmatrix} 16 & 9 & 15 \\ 13 & 17 & 21 \\ 8 & 11 & 8 \\ 7 & 12 & 13 \\ \end{tabular}$ 

100% of points

QUESTION

Given the Key 
$$K = \begin{pmatrix} 0.4 & 0.8 & 0.1 \\ 0.4 & 0.1 & 0.7 \\ 0.7 & 0.7 & 0.1 \\ 0.8 & 0.3 & 0.2 \end{pmatrix}$$
 and Query

$$Q = \begin{pmatrix} 0.8 & 0.4 & 0.4 \\ 0.5 & 0.2 & 0.8 \\ 0.3 & 0.5 & 0.8 \\ 0.6 & 0.5 & 0.2 \end{pmatrix} \text{ tokens in a transformer with an}$$

input token X, calculate the attention matrix A. (Give your answer as a  $N \times N$  matrix in the box below, with two decimal places for each value)

For the values given in the question we calculate:

$$\begin{split} K^TQ &= \begin{pmatrix} 0.40 & 0.40 & 0.70 & 0.80 \\ 0.80 & 0.10 & 0.70 & 0.30 \\ 0.10 & 0.70 & 0.10 & 0.20 \end{pmatrix} \times \begin{pmatrix} 0.80 & 0.40 & 0.40 \\ 0.50 & 0.20 & 0.80 \\ 0.30 & 0.50 & 0.80 \\ 0.60 & 0.50 & 0.20 \end{pmatrix} \\ &= \begin{pmatrix} 1.210 & 0.990 & 1.200 \\ 1.080 & 0.840 & 1.020 \\ 0.580 & 0.330 & 0.720 \end{pmatrix} \end{split}$$

The final attention matrix A is formed by applying the softmax function to the columns (in this case) of the matrix  $K^TQ$ . The softmax is applied to normalise the attention from  $k_m$   $m \in [1,2..N]$  to each  $q_n$ , therefore  $a\left[k_m,q_n\right]=\frac{e^{k_m^Tq_n}}{\sum_{m'=1}^N e^{k_{m'}^Tq_n}}$ . The softmax function limits the magnitudes of the attention values making the network easier to train.

The attention matrix is given by 
$$A = \begin{pmatrix} 0.41 & 0.42 & 0.41 \\ 0.36 & 0.36 & 0.34 \\ 0.22 & 0.22 & 0.25 \end{pmatrix}$$

#### Question at wrong answer

Attention is formed as the dot product (or equivalent matrix multiplication  $K^TQ$ ) between the Keys and Queries.

The final attention matrix A is formed by applying the softmax function to the columns (in this case) of the matrix  $K^TQ$ . The softmax is applied to normalise the attention from  $k_m \ m \in [1,2..N]$  to each  $q_n$ ,

therefore 
$$a\left[k_{m},q_{n}\right]=\frac{e^{k_{m}^{T}q_{n}}}{\sum_{m'=1}^{N}e^{k_{m'}^{T}q_{n}}}$$

✓ Feedback for compound rule based answer (index 1)

No specific feedback

**QUESTION** 

Given the attention matrix 
$$A = \begin{pmatrix} 0.41 & 0.42 & 0.41 \\ 0.36 & 0.36 & 0.34 \\ 0.22 & 0.22 & 0.25 \end{pmatrix}$$

Given the attention matrix 
$$A=\begin{pmatrix} 0.41 & 0.42 & 0.41 \\ 0.36 & 0.36 & 0.34 \\ 0.22 & 0.22 & 0.25 \end{pmatrix}$$
 and value tokens  $V=\begin{pmatrix} 0.3 & 1.0 & 0.4 \\ 0.5 & 0.7 & 0.6 \\ 0.0 & 0.3 & 0.6 \\ 0.7 & 0.3 & 0.9 \end{pmatrix}$  determine the

self attention matrix Sa produced at the output of the attention block. (Give your answer as a  $D \times N = 4 \times 3$  matrix in the box below, with two decimal places for each value)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$                                                                                                                                                                                                                                   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|                                                                                                                                                                                                                                                                                          |
| $\boxed{\alpha\beta\infty} \boxed{\sin\cos\left[\int\sum\prod\right] \left[ !\binom{n}{k}\ln\right] \boxed{\Box} x_{\Box}} \left[ \{\},(,]\right] \left[ =\neq \leq \right] \boxed{\neg\vee} \boxed{\left( \begin{bmatrix} \Box & \Box \\ \Box & \Box \\ \end{bmatrix} \bullet \right]}$ |
| Check my answer                                                                                                                                                                                                                                                                          |

#### ✓ Detailed solution

The self attention matrix is simply produced by multiplying the attention matrix by the values.

$$Sa \ = \ V \times A = \begin{pmatrix} 0.3 & 1.0 & 0.4 \ 0.5 & 0.7 & 0.6 \ 0.0 & 0.3 & 0.6 \ 0.7 & 0.3 & 0.9 \end{pmatrix} \times \begin{pmatrix} 0.41 & 0.42 & 0.41 \ 0.36 & 0.36 & 0.34 \ 0.22 & 0.22 & 0.25 \end{pmatrix}$$

This results in an output Self attention matrix with size  $D \times N$ 

$$Sa = \begin{pmatrix} 0.57 & 0.57 & 0.56 \ 0.59 & 0.59 & 0.59 \ 0.24 & 0.24 & 0.25 \ 0.59 & 0.60 & 0.61 \end{pmatrix}$$

□ Question at wrong answer

The self attention matrix is simply produced by multiplying the attention matrix by the values.  $Sa = V \times A$ 

☑ Feedback for compound rule based answer (index 1)

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 114532

**QUESTION** 

A vision transformer has an input image size of  $H \times W = 64 \times 64$  and will use a patch size of  $P \times P = 16 \times 16$ , determine the length of the input token sequence. (Give your answer in the box below)

| $\begin{array}{ c c c c c c c c c c c c c c c c c c c$ |
|--------------------------------------------------------|
| 16                                                     |
| Check my answer                                        |

#### ✓ Detailed solution

In order to determine the the total number N of patches of size  $P \times P$  that will span the input image of size  $H \times W$ , we can simply divide each image dimension by the corresponding patch dimension, to determine how many patches will fit into the image. As both dimensions are the same this will be  $\frac{64}{16} = 4$ 

More formally we could use Relationship 5 from the guide to convolutional arithmetic (https://arxiv.org/pdf/1603.07285) to perform this calculation. For an image with no zero padding p=0 and any input size i=64, stride s=16 (assumed to be the same as patch size, so we have non-overlapping patches), and kernel size k=16 (the patch size is our kernel size). This relationship has the formula to calculate the output size o (This will be same as the number of

patches in one dimension of our input, as the image is square, it will have the same no of patches in each dimension).

$$o = \lfloor \frac{i-k}{s} \rfloor + 1 = \lfloor \frac{64-16}{16} \rfloor + 1 = 4$$

The total number of patches will be equal to this number squared

$$N = 4^2 = 16$$

The number of input tokens N is the same as the number of patches, with each patch being converted into a token. (The tokens are formed from the patches after being passed through a linear layer which changes the dimensionality to D)

#### □ Question at wrong answer

In order to determine the the total number N of patches of size  $P \times P$  that will span the input image of size  $H \times W$ , we can simply divide each image dimension by the corresponding patch dimension, to determine how many patches will fit into the image.

More formally we could use Relationship 5 from the guide to convolutional arithmetic (https://arxiv.org/pdf/1603.07285) to perform this calculation. For an image with no zero padding p=0 and any input size i=64, stride s=16 (assumed to be the same as patch size, so we have non-overlapping patches), and kernel size k=16 (the patch size is our kernel size). This relationship has the formula to calculate the output size o (This will be same as the number of patches in one dimension of our input, as the image is square, it will have the same no of patches in each dimension).

$$o = \lfloor \frac{i-k}{s} \rfloor + 1$$

The number of input tokens N is the same as the number of patches, with each patch being reshaped in to a token. (The tokens are formed

from the patches after being passed through a linear layer which changes the dimensionality to D)

✓ Feedback when answer is equal to 16

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 114544

QUESTION

A typical transformer encoder block is comprised of a multi-ahead attention block and Multi-layer perceptron block with additional Normalisation layers and residual connections.

Given the transformer uses a dimension of D=256 for the input embedding tokens and there are H=8 attention heads . Calculate the number of parameters in the multi-head attention block encoder block. (We assume bias weights are not used in the linear layers associated with Queries, Keys and Values of the attention blocks, but is used in the final linear layer of the multi-head attention block). Give your answer in the box below.

#### ✓ Detailed solution

We note that in multi-head self attention the input dimension of each token is split across the H heads with each receiving part of the token of dimension  $\frac{D}{H}$ . Each individual attention block contains 3 linear (fully connected) layers that convert the input to Keys, Queries and values. We also note that there is a final linear layer that operates on the concatenated outputs of the self attention blocks. (The other operations in the multi-head attention (multiplication, softmax, concatenation) do not contribute to the number of parameters.)

We will first calculate the parameters associated with the self attention mechanism. This contains parameters associated with the linear (fully connected, no activation) layers that convert the input tokens into Keys, Query's and values. These layers have  $\frac{D}{H}$  neurons (as the output of the attention block is the same size as the input) and there is a multiplicative weight  $w_{ij}$  connecting each  $i^{th}$  input to the  $j^{th}$  neuron (every input is connected to each neuron). Therefore there are  $\frac{D}{H} \times \frac{D}{H}$  multiplicative weights due to each of the queries, keys and values, (as noted in the question bias weights are not included in the attention block)

The number of parameters in the self attention mechanism will therefore be given by  $N_A=3\times\frac{D}{H}\times\frac{D}{H}=3\cdot\frac{256}{^{\bm Q}}\cdot\frac{256}{^{\bm Q}}=3072$ 

Secondly we consider the output linear layer. This operates on the full D dimensions (same input and output size) and therefore contains

 $N_L=D \times D+D=256^2+256=65792$  parameters, the additional D is included for the bias parameters of this layer.

The total number of parameters in the multi-head attention will be given by:

 $N_{MHA}=H\times N_A+N_L$  for H heads each with 3 linear layers due to Keys queries and values plus the parameters from the output linear layer.

$$N_{MHA} = H \times 3 \times \frac{D}{H} \times \frac{D}{H} + D \times D + D = 8 \cdot 3 \cdot \frac{256}{8} \frac{256}{8} + 256^2 + 256 = 90368$$

#### ✓ Question at wrong answer

We note that in multi-head self attention the input dimension of each token is split across the H heads with each receiving part of the token of dimension  $\frac{D}{H}$ . Each individual attention block contains 3 linear (fully connected) layers that convert the input to Keys, Queries and values. We also note that there is a final linear layer that operates on the concatenated outputs of the self attention blocks. (The other operations in the multi-head attention (multiplication, softmax, concatenation) do not contribute to the number of parameters.)

18/06/2026. 16:08

We will first calculate the parameters associated with the self attention mechanism. This contains parameters associated with the linear (fully connected, no activation) layers that convert the input tokens into Keys, Query's and values. These layers have  $\frac{D}{u}$  neurons (as the output of the attention block is the same size as the input) and there is a multiplicative weight  $w_{ij}$  connecting each  $i^{th}$  input to the  $j^{th}$  neuron (every input is connected to each neuron). Therefore there are  $\frac{D}{H} \times \frac{D}{H}$  multiplicative weights due to each of the queries, keys and values, (as noted in the question bias weights are not included in the attention block)

Secondly we consider the output linear layer. This operates on the full D dimensions (same input and output size) and therefore contains  $N_L = D \times D + D$  parameters, the additional D is included for the bias parameters of this layer.

✓ Feedback when answer is equal to 90368
No specific feedback

QUESTION

Determine the number of parameters used in the Multi-layer perceptron block of the transformer encoder. Assume that the multi-layer perceptron consists of one hidden layer with a multiplicative factor of r=5 increase in the dimensionality, the output layer uses the same dimensionality D=256. Assume that bias weights are used for both layers. Give your answer in the box below.

✓ Detailed solution

Determine parameters in Multi-layer perceptron:

As detailed in the question we assume the MLP has one hidden layer, this means it comprises of an input layer and 2 layers of neurons. The input will be of size D=256, the dimension of the tokens. The number of neurons in the hidden layer is increased by the multiplicative factor  $\ r=5$  so there are  $\ r\cdot D$  neurons in the hidden layer.

We can calculate the number of multiplicative weights  $N_{W1}$  from the input to the hidden layer as the number of inputs times the number of neurons in the hidden layer (as every input is connected to each neuron)

 $N_{W1}=D\cdot r\cdot D=rD^2$  . We also include bias weights, there is one weight for each neuron  $N_{B1}=r\cdot D$  .

For the output layer we similarly calculate the number of parameters due to the multiplicative weights as  $N_{W2}=r\cdot D\cdot D=r\cdot D^2$  and additionally the bias weights are given by  $N_{B2}=D$ .

For the multilayer perceptron the total number of parameters is then given by

$$N_{MLP} = N_{W1} + N_{B1} + N_{W2} + N_{B2} = r \cdot D^2 + r \cdot D + r \cdot D^2 + D = 2r \cdot D^2 + D \left(r + 1\right)$$

$$N_{MLP} = 5 \cdot 256 (2 \cdot 256 + 1) = 656896$$

#### ■ Question at wrong answer

As detailed in the question we assume the MLP has one hidden layer, this means it comprises of an input layer and 2 layers of neurons. The input will be of size D, the dimension of the tokens. The number of

neurons in the hidden layer is increased by the multiplicative factor  $\ r$  so there are  $\ r \cdot D$  neurons in the hidden layer.

We can calculate the number of multiplicative weights from the input to the hidden layer as the number of inputs times the number of neurons in the hidden layer (as ever input is connected to each neuron). We also include bias weights, there is one weight for each neuron.

For the output layer we follow a similar procedure to calculate the multiplicative weights but now the inputs and output sizes are reversed from the first calculation (and we must not forget the bias parameters).

✓ Feedback when answer is equal to 656896

100% of points

QUESTION

Finally determine the number of parameters due to the Normalisation layers.

#### ✓ Detailed solution

The transformer encoder (and transformers in general) use Layer normalisation rather than batch normalisation. Often when training transformer architectures small batch sizes are used (due to large model sizes), and secondly the batch size may be inconsistent in size (especially when processing text), so for these reasons batch normalisation is not used.

The layer normalisation function applies normalisation over the features of the input (typically in a transformer the normalisation is applied over the dimensions of each token).

In the pytorch implementation (pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) . The normalisation for each element (of a token) is calculated as

$$y = \frac{x - E\left[x\right]}{\sqrt{Var\left[x\right] + \epsilon}} \cdot \gamma + \beta$$

where x is the input element value, y is the output element value,  $E\left[x\right]$  and  $Var\left[x\right]$  are the mean and standard deviations over the elements of the token. The parameters  $\gamma$  and  $\beta$  are learned scaling

and offset parameters with a pair of parameters for each element of the token.

Therefore the number of parameters due to each normalisation layer will be  $N_{Norm1}=2D$ , (D parameters due to  $\gamma$  and  $\beta$ )

As there are two normalisation layers in the transformer encoder and both have the same dimensions.

$$N_{Norm} = N_{Norm1} + N_{Norm2} = 2D + 2D = 4D = 1024$$

#### □ Question at wrong answer

The transformer encoder (and transformers in general) use Layer normalisation rather than batch normalisation. Often in transformer architectures there are small batch sizes, and the batch size may be inconsistent in size (especially when processing text), so for these reasons batch normalisation is not used.

The layer normalisation function applies normalisation over the features of the input (typically in a transformer the normalisation is applied over the dimensions of each token).

In the pytorch implementation (pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) . The normalisation for each element (of a token) is calculated as

$$y = \frac{x - E\left[x\right]}{\sqrt{Var\left[x\right] + \epsilon}} \cdot \gamma + \beta$$

where x is the input element value, y is the output element value, E[x] and Var[x] are the mean and standard deviations over the elements of the token. The parameters  $\gamma$  and  $\beta$  are learned scaling and offset parameters with one of each learned for each element of the token.

✓ Feedback when answer is equal to 1024

\*\*No specific feedback\*\*

100% of points

# Exercise 114604

QUESTION

Determine the values for positional encoding (PE) for each element of the token at pos=239 (in a sequence of input tokens) with token dimension D=4 given the positional embedding equations:

$$PE_{(pos,2i)} = \sin\left(\frac{pos}{10000\frac{2i}{D}}\right) \ PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000\frac{2i}{D}}\right)$$

Give your answer as a  $D \times 1$  matrix of PE values. (Note that the calculated PE values should be returned in the normalised range [-1,1], so you can convert from radians by dividing by  $2\pi$ , or from degrees by dividing by 360)

$$\begin{array}{c|ccccccccccccccccccccccccccccccccccc$$

# Check my answer

#### ✓ Detailed solution

To generate the positional encodings PE for the token at position pos = 239 we use the equations in pairs given starting at index i = 0.

For i = 0:

The  $\sin$  term will gives us the encoding for the first element of the token  $PE\left(pos,2i\right)=PE\left(pos,0\right)$ .

We then calculate

$$PE(pos,0) = \sin\left(\frac{pos}{10000^{\frac{2i}{D}}}\right) = \sin\left(\frac{239}{10000^{\frac{2\cdot0}{4}}}\right) = \sin\left(\frac{239}{1}\right) = 0.237$$

The costerm gives us the PE for the second element of the token  $PE\left(pos,2i+1\right)=PE\left(pos,1\right)$ 

$$PE\left(pos,1\right) = \cos\left(\frac{pos}{10000^{\frac{2i}{D}}}\right) = \cos\left(\frac{239}{10000^{\frac{2\cdot 0}{4}}}\right) = \cos\left(\frac{239}{1}\right) = 0.972$$

For i = 1 we repeat the procedure getting the next two terms.

$$PE(pos,2i) = PE(pos,2)$$

$$PE\left(pos,2\right)=\sin\left(\frac{pos}{10000\frac{2i}{D}}\right)=\sin\left(\frac{239}{10000\frac{2\cdot 1}{4}}\right)=0.683$$

$$PE(pos,2i+1) = PE(pos,3)$$

$$PE\left(pos,3\right) = \cos\left(\frac{pos}{10000\frac{2i}{D}}\right) = \cos\left(\frac{239}{10000\frac{2\cdot 1}{4}}\right) = -0.731$$

Finally we normalise all the calculated (radian) values by dividing by  $2\cdot\pi$ 

$$PE_{norm} = \begin{pmatrix} \frac{0.237}{2.7} \ \frac{0.972}{2.7} \ \frac{0.683}{2.7\pi} \ \frac{-0.731}{2.\pi} \end{pmatrix} = \begin{pmatrix} 0.04 \ 0.15 \ 0.11 \ -0.12 \end{pmatrix}$$

#### Question at wrong answer

To generate the positional encodings PE for the token at position pos=239 we use the equations in pairs given starting at index i=0.

For i = 0:

The  $\sin$  term will gives us the encoding for the first element of the token  $PE\left(pos,2i\right)=PE\left(pos,0\right)$ .

We then calculate 
$$PE\left(pos,0\right)=\sin\left(\frac{pos}{10000^{\frac{2i}{D}}}\right)=\sin\left(\frac{239}{10000^{\frac{2\cdot 0}{4}}}\right)$$

The costerm gives us the PE for the second element of the token  $PE\left(pos,2i+1\right)=PE\left(pos,1\right)$ 

$$PE\left(pos,1\right)=\cos\left(\frac{pos}{10000^{\frac{2i}{D}}}\right)=\cos\left(\frac{239}{10000^{\frac{2\cdot0}{4}}}\right)$$

For i = 1 we repeat the procedure getting the next two terms.

Don't forget to normalise all the calculated (radian) values by dividing by  $2\cdot \pi$ 

Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 0.04 \\ 0.15 \\ 0.11 \\ 0.12 \end{pmatrix}$ 

<u>View exercise</u> | <u>Edit exercise</u>

## Exercise 130261

QUESTION

A typical Transformer Encoder block consists of a Multi-head attention layer followed by a MLP (Multi-Layer Perceptron).

The MLP block typically increases the dimensionality of the input by a ratio of r=4 in the hidden layer and then projects it back to the original dimension D=64 Calculate the number of parameters in the MLP block (use bias parameters in linear layers within MLP). Give your answer in the box below.

# Check my answer

#### ✓ Detailed solution

The MLP has two linear layers, firstly an expansion layer that increases the size of the embedding by r=4 and a projection layer that reduces in back down to D=64.

The expansion layer has an input size of D=64 and an output size of  $4D=4\cdot 64$ 

The number of weights  $w_e$  in this first expansion layer is simply calculated as the number of inputs by the number of neurons (which is the same as the number of outputs).  $w_e = 64 \times 4 \cdot 64 = 16384$ . The bias parameters associated with the 1st layer is equal to the number of neurons  $b_e = 4 \times 64 = 256$ .

In the second projection layer the input is of size  $4D=4\cdot 64$  and an output size of D=64.

The number of weights  $w_p$  in the projection layer is again calculated as the number of inputs by the number of neurons and will be

 $w_p=4\cdot 64\times 64=16384.$  The bias parameters associated with the 2nd layer are equal to the number of neurons  $b_p=64.$ 

The total number of parameters will be given by

$$P_T = w_e + b_e + w_p + b_p = 33088$$

#### Question at wrong answer

The MLP has two linear layers, firstly an expansion layer that increases the size of the embedding by r=4 and a projection layer that reduces in back down to D=64.

The expansion layer has an input size of D=64 and an output size of  $4D=4\cdot 64$ 

The number of weights  $w_e$  in this first expansion layer is simply calculated as the number of inputs by the number of neurons (which is the same as the number of outputs).  $w_e = D \times 4D$ . The bias parameters associated with the 1st layer is equal to the number of neurons  $b_e = 4 \times D$ .

In the second projection layer the input is of size 4D and an output size of D.

The number of weights  $w_p$  in the projection layer is again calculated as the number of inputs by the number of neurons and will be

 $w_p=4D\times D$  . The bias parameters associated with the 2nd layer are equal to the number of neurons  $b_p=64$  .

The total number of parameters will be given by

$$P_T = w_e + b_e + w_p + b_p$$

☑ Feedback when answer is equal to 33088

100% of points

<u>View exercise</u> | <u>Edit exercise</u>

# Exercise 130280

**QUESTION** 

A vision transformer often uses a Convolution layer to create the patch embeddings directly from the image (instead of slicing the image and passing patches through a linear layer). Given an image with C=3 channels (RGB), a patch spatial size of  $P\times P=32\times 32$  and an output embedding dimension D=512. Determine the required configuration for the convolution layer, give your answer as a 1 x 4 matrix with values for the (Kernel Size, Stride, Padding and no. of kernels)

#### ✓ Detailed solution

The vision transformer requires the input image to be divided into non-overlapping patches, that are flattened and input to a linear layer to convert them to an embedding of size D. A convolution layer can perform the same function if the convolution kernel size is the same

as the patch size (to select the patches). A convolution stride the same as the patch size is used to avoid overlapping, no (zero) padding is required as we only want to select the patch region. The number of kernels can be set to be equal to D so the output feature map will have the same number of channels as the embedding dimension. Each 1 x 1 x D vector of the output feature map is the embedding of a single patch.

From the information in the question, the patch size is  $P \times P = 32 \times 32$ , so our kernel size k = P = 32. The stride is also the same as the patch size s = P = 32, and no zero padding is require p = 0. The number of kernels N is given by required size of the embedding N = D = 512. So the required 1 x 4 vector will be  $[k,s,p,N] = [P,P,0,D] = (32 \quad 32 \quad 0 \quad 512)$ 

#### 

The vision transformer requires the input image to be divided into non-overlapping patches, that are flattened and input to a linear layer to convert them to an embedding of size D. A convolution layer can perform the same function if the convolution kernel size is the same as the patch size (to select the patches). A convolution stride the same as the patch size is used to avoid overlapping, no (zero) padding is required as we only want to select the valid patch region. The number of kernels can be set to be equal to D so the output feature map will have the same number of channels as the embedding dimension.

 $\bullet$  Feedback when answer is algebraically Equivalent to  $\begin{pmatrix} 32 & 32 & 0 & 512 \end{pmatrix}$ 

100% of points