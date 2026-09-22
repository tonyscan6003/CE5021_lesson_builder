# lesson_1_1_convolution

Deep Learning for Computer vision 

2D Convolution Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Convolution 

- Convolution Operation • Allows Modification of Image by a Kernel (Filter) 

      - (a) Example Image and Blurred Version 

- Linear Operation 

   - Every point in image has same 

   - values applied. 

- Widely used in Image processing • Blurring, Sharpening, Finding edges….. 

- Core operation in Convolutional Neural Networks 

- (b) Example Image and Edge Detection with Sobel Kernel 

©Dr. Tony Scanlan 

# 2D Convolution Operation 

      - (a) Example of Reversing (or flipping) kernel 

- Multiplication of image with reversed and shifted kernel. 

- “Reverse” the kernel • Example (b) has a <u>symmetric</u> kernel. 

- “Shifting the kernel” 

   - Slide kernel from/until just overlapping (assume additional zeros (padding) around image) 

   - Elementwise Multiplication and sum obtained at each image position. 

(b) 2D Convolution Example 

©Dr. Tony Scanlan 

# Mathematical Statement 2D Convolution 

- Multiplication of image with reversed and shifted kernel. 

- 2D Input image 𝑥 𝑚, 𝑛 and kernel ℎ 𝑚, 𝑛 

Position Shift relative to image • Convolution Equation (2D): Reversal ∞ ∞ − 𝑥∗ℎ 𝑥 𝑖, 𝑗∙ℎ 𝑖−𝑚, − 𝑗−𝑛 **(1)** 𝑚, 𝑛= ෍ ෍ 𝑗=−∞ 𝑖=−∞ Dummy Variables 

- This describes the “sliding” (multiply and sum) operation with the reversed kernel from previous slide 

. <u>Example of 2D convolution showing relationship between mathematical function and2D convolution</u> 

©Dr. Tony Scanlan 

# Zero Padding 

- Mathematically Convolution is defined from −∞ to +∞ 

- Unless otherwise stated, should slide kernel to positions where just overlaps image (pad images with required no of zeros). 

   - This will create some edge effects 

   - Not significant small kernel, big image 

- Size of output image can change from input. 

- Depending on requirements, we may choose specific amounts (or none) of padding. 

   - E.g. Want output image to be the same size as input image 

- (a) Convolution of 4 x 4 Image with 3 x 3 kernel using 

- 2 rings of zero padding, returning a 6 x 6 output. 

- (b) Convolution of 4 x 4 Image with 3 x 3 kernel using 

- 1 ring of zero padding, returning a 4 x 4 output (same size as input). 

©Dr. Tony Scanlan 

# Stride 

- Stride 

   - Moves the kernel by stride value each time. 

   - Normally stride assumed = 1. 

- Stride > 1 

   - Downsampling: Can be used to deliberately reduce the size of the image. (e.g stride 2) 

   - (a) Convolution of 6 x 6 Image with 2 x 2 kernel, stride = 2 (no zero padding),  returning a 3 x 3 output. 

- Also used to reduce computation. 

©Dr. Tony Scanlan 

# Pooling 

- Key Operation in CNNs 

- Uses a sliding window like convolution. 

- • Usually stride 2. 

   - Used to reduce size of Feature Maps. 

- Apply Max pooling or Average Pooling operations. 

a) Max Pooling with 2 x 2 operator (stride = 2), (no padding) b) Average Pooling with 2 x 2 operator (stride = 2), (no padding) 

©Dr. Tony Scanlan 

# Convolution Maths I 

- Lots of variations of convolution can be observed in CNN. 

   - Want to be able to determine output size. 

   - No universal formula, every case different [1] 

      - 𝑖= input image size 

      - 𝑘= kernel size 

      - 𝑜= ouptut image size 

      - 𝑝= padding 

      - 𝑠= stride 

- 2 Examples: 

   - “Valid” Case with no padding 

   - “Same” padding, where want 𝑜= 𝑖 

- (a) Relationship 1: For any 𝑖 and 𝑘 unit strides 𝑠= 1 and no padding 𝑝= 0 𝑜= 𝑖−𝑘+ 1 

- In this case 𝑖 =3, and 𝑘 =2 ⇒𝑜= 3 −2 + 1 = 2 

**(1)** 

(b) Relationship 3 (“Same” Padding): For any 𝑖 and 𝑘 odd unit strides 𝑠= 1 and padding 𝑝= 𝑘2Τ 𝑜= 𝑖+ 2 𝑘2Τ − 𝑘−1 **(2)** In this case 𝑖 =4, and 𝑘 =3, so 𝑝= 𝑘2Τ = 1 and must use ⇒𝑜= 4 + 2 × 1 − 3 −1 = 4 

[1] A guide to convolution arithmetic for deep learning 

©Dr. Tony Scanlan 

# Convolution Maths II 

- 2 Further Examples: 

   - “Full” Padding: Case where fully pad image in accordance with convolution theory. 

   - Downsampling: No zero padding, non-unit strides 

- (a) Relationship 4 (“Full” Padding): For any 𝑖 and 𝑘 ,  unit strides 𝑠 =1 and padding 𝑝= 𝑘−1 

𝑜= 𝑖+ 2 𝑘−1 − 𝑘−1 = 𝑖+ 𝑘−1 **(1)** In this case 𝑖 =4, and 𝑘 =3 ⇒𝑝= 𝑘−1 = 2 𝑜= 4 + 3 −1 = 6 

(b) Relationship 5 (Non-unit strides): For any 𝑖, 𝑘 and 𝑠 , and 𝑝= 0 

[1] A guide to convolution arithmetic for deep learning 

𝑖−𝑘 **(2)** 𝑜= + 1 𝑠 6−2 In this case 𝑖 =6, 𝑘 =2 and s = 2 ⇒𝑜= + 1 = 3 2 

©Dr. Tony Scanlan 

# Cross Correlation 

- Mathematically Similar 

   - Kernel Not Reversed 

   - Measures Similarity of Image and Kernel 

- However most (image processing) kernels are Symmetric. 

   - Reversal doesn’t change kernel. 

   - Convolution and Cross Correlation results will be the same.
