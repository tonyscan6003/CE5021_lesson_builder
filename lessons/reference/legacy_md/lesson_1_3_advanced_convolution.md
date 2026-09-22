# lesson_1_3_advanced_convolution

Deep Learning for Computer vision 

Advanced Convolution Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Introduction 

- Convolutional Neural Network (CNN). 

- Convolution layers are Main Building block in Deep Neural Networks (for imaging). 

   - Provide feature detection. 

- Convolution Operations 

   - (a) Example of a Deep neural Network with Convolutional and Fully Connected Layers 

- Operate across multiple channels 

- Use multiple kernels 

- 1 x 1 Convolution and Bottlenecks 

©Dr. Tony Scanlan 

# Multiple Channels 

- Have discussed Convolution • 2D (Greyscale) images & Kernel 

- Colour Images Typically RGB 

   - Require kernel to have same number of channels as Image 

   - Convolution Image and Kernel results in a “Feature Map” with one channel 

- Deep layers CNNs • Multiple input channels 

   - Kernels must have same no. channels 

   - Output sizes governed by same relationships as 2D conv (e.g Valid, Same, Full, stride>1) 

©Dr. Tony Scanlan 

### Example Conv. Multiple Input Channels 

- RGB Image with Kernel. • Each Kernel (size k x k) has same number of channels C as input. 

   - Operates as a series of regular 2D convolutions. 

   - +Additional Sum across the Channels. 

   - One output (feature) map per kernel. 

©Dr. Tony Scanlan 

# Multiple Kernels 

- Deep Convolutional Neural Networks 

   - Parallel Kernels increases number of parameters of CNN. 

      - Learns complex Features 

   - Each Kernel has the same number of channels as the input feature map (or Image). 

   - One output feature map channel generated per kernel. 

   - Multiple kernels generate multiple output feature maps. 

   - Output sizes governed by same relationships as 2D conv (e.g Valid, Same, Full, stride>1) 

©Dr. Tony Scanlan 

# Volumes 

- Common Visual Representation Deep convolutional Neural networks 

   - Don’t want to draw individual feature maps (can be hundreds in each layer) 

- Show all feature maps in layer together as a “volume”. 

   - Kernel filters may not be explicitly shown. 

   - Get operation details from inspection of Input/Output volumes. 

- E.g. N kernels of size k x k x C 

©Dr. Tony Scanlan 

# Convolution Parameters 

- Calculating number of Parameters used by inspection. 

   - Number of channels in Output Feature Maps 

      - => Number of Kernels =N (as one kernel per Feature map) 

   - Number of channels in input Feature maps 

      - => Kernels have same number of channels C 

   - Spatial Size of kernel 𝑘× 𝑘 

      - => Each kernel channel has 𝑘× 𝑘 parameters 

- Calculate parameters 

   - 𝑁× 𝑘× 𝑘× 𝐶 

- Include Bias for each output feature map 

   - 𝑁× 𝑘× 𝑘× 𝐶+ 𝑁 

©Dr. Tony Scanlan 

# Convolution Operations 

- Calculating the Number of Operations (Multiplications) by inspection 

   - Already Calculated No. parameters 

         - 𝑁× 𝑘× 𝑘× 𝐶 

   - Number of multiplications depends on the number of times the kernels are multiplied against each region of the image. 

      - => This is given by the output image spatial size 𝑚<sup>′</sup> × 𝑛<sup>′</sup> 

      - Don’t need to know exact padding/stride! 

   - Total number of multiplications 

- 𝑚<sup>′</sup> × 𝑛<sup>′</sup> × 𝑁× 𝑘× 𝑘× 𝐶 

©Dr. Tony Scanlan 

# 1 x 1 convolution 

- Convolution Kernels can be any size • Typically see 3 x 3 in modern networks 

   - Smaller convolutions less parameters/computation 

- 1 x 1 convolution: • Smallest number of parameters • Input and Output Feature Maps always same size (no padding required) 

   - Use to reduce numbers of channels (bottleneck) 

©Dr. Tony Scanlan 

## Bottleneck Convolution 

#### • Normal Conv. 

   - If convolve 𝐶= 24 channel input of size 𝑚, 𝑛= 32 with a 𝑁= 12 kernels of size 𝑘= 3 (“Same” 

   - padding stride =1). • 2,654,208 Multiplications. 

- Bottleneck Conv. • Step1: Convolve 𝐶= 24 channel input of size 𝑚, 𝑛= 32 with a 𝑃= 12 kernels of size 𝑘= 1 . • 294,912 Multiplications. 

   - Step2: Convolve 𝑃= 12 channel input of size 𝑚, 𝑛= 32 with a 𝑁= 12 kernels of size 𝑘= 3 (“Same” padding stride =1). 

   - 1,327,104 Multiplications 

   - • Total Multiplications: 1,622,016
