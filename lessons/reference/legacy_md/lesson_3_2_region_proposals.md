# lesson_3_2_region_proposals

Deep Learning for Computer vision 

Region Proposals Instructor: Dr. Tony Scanlan 

This work Licenced for re-use under: CC BY-NC-SA 4.0 

©Dr. Tony Scanlan 

# Object Detection 

- Where to look for an object in an image? 

- Sliding window & Image Pyramid. 

   - Used to find objects at different scales in an image. 

   - Computationally intensive. 

      - Applies classifier at each window position. 

      - Viola Jones Cascade rejects void windows. 

- Object Detection stage 

   - Locates candidate objects 

   - Reduces wasted computation 

   - => Region Proposals 

(c) Viola Jones Object Detection. (only valid window fully Computed) 

©Dr. Tony Scanlan 

# Region Proposals 

- Add a region proposal stage to object detection pipeline 

- Region Proposal Algorithm Characteristics: 

   - Produces less windows to classify. 

   - Finds any objects of any shape/scale. 

   - Low number of False Positives Proposals. (Precision) 

   - Low number False Negatives Proposals. (Precision) 

- (a) Object Detection pipeline with Region Proposals and Trained Classifier 

(b) True Positive, False Positive and False Negative Region Proposals (Original Image<sup>(i)</sup> ) 

(i) “Lake and Boats in the Inagh Valley”, Author: M. P. Whyte, License: **CC BY-SA 4.0** 

### ©Dr. Tony Scanlan 

## Objectness [1] (Multi Scale Saliency) 

- Objectness: A Measure that quantifies how likely it is for an image window to cover an object of any class. 

   - Use 5 Image cues that can be combined to determine if a proposed window is likely to contain an image. 

      - No single cue works perfectly for all cases. 

• 1. Multi Scale Saliency 

- Salient pixels stand out from their surroundings. 

- A saliency map must be generated at different scales. 

- Fourier Transform based saliency maps. 

**Multi-Scale Saliency** : The large giraffe in the original image (a) appears as a blob in the saliency map for a high scale (b), while the tiny airplane appears in the map for a low scale (c) Numerous cars in the original image (d) are not salient at any scale as shown in the saliency maps for 2 scales in (e) and (f). The contour of the building appears more salient than the cars. [1] 

[1] Alexe, B., Deselaers, T. and Ferrari, V., 2012. Measuring the objectness of image <u>windows.</u> _IEEE transactions on pattern analysis and machine intelligence_ , _34_ (11), pp.2189-2202. 

©Dr. Tony Scanlan 

# Objectness (Colour Contrast) 

- 2. Colour Contrast 

   - There is a high degree of colour contrast between an object an it’s surroundings. 

   - • Obtained by comparing histograms of pixels (in LAB colour space) from inside and candidate windows and a ring of pixels just outside the window. 

**Colour Contrast** : The windows containing the objects (cyan) have high color contrast with their surrounding ring (yellow) in images (a) and (b). The color contrast for the object in image (c) is much lower and the object is not detected. [1] 

- Detection may not occur if objects have similar colour to their surroundings. 

©Dr. Tony Scanlan 

## Objectness (Edge Density) 

- 3. Edge Density. 

   - Measure the density of edges in an inner ring of the proposed region. 

   - Objects will have a large amount of edges in their boundary compared to a uniform region. 

   - Use canny algorithm to produce edge. 

   - Note: Certain Regions such as water/trees produce lots of edges => False positive. 

**Edge Density:** Given images (a) and (b) the cyan windows covering the bus and the aeroplane have a high edge density. In image (c) the cyan window along with many other windows covering the water also have a large number of edges, however any windows covering the sail boats will see few edges by comparison. False positive region proposals result. (Ring of edge density  calculated between cyan and yellow boxes) [1] 

©Dr. Tony Scanlan Objectness (SuperPixel Straddling) 

- 4. Superpixel Straddling 

   - Superpixels [2] segment an image into small regions of colour and texture. 

   - A property of superpixels is that they do not straddle an objects boundary. 

   - The cue is based on the degree to which superpixels straddle the boundary of the proposed window. 

      - E.g. W3 is a better window than W1 

- (a) Dog Image divided into superpixels 

- (b) Ship Image divided into Large Superpixels [1] 

[2] Ren, X. and Malik, J., 2003, October. Learning a <u>classification model for segmentation. In ICCV (p. 10). IEEE.</u> 

©Dr. Tony Scanlan Objectness (Location and Size) 

- 4 (cont). Superpixel Straddling • Best individual cue for determining good region proposals. 

   - Segmentation is a way to define region proposals. 

- 5. Location and size. 

   - Based on real image statistics some proposed regions are more likely to contain an object than others. 

**Superpixel Stradding** : The cyan windows in (a) and (b) are computed from segmentations (d) and (e). Failure: Segmentation produces superpixels (f) not preserving the boundaries of the small objects in (c) resulting in False negatives for the proposed windows. 

- E.g. Windows in centre of image. 

©Dr. Tony Scanlan 

# Region Proposals with Objectness 

- Complex to train region proposal generator with objectness cues. 

- Windows are generated at random. 

   - More candidate windows leads to better region proposals. (All objects found) 

   - Slower to run. 

- Exhaustive search algorithms would be even slower generation 100k windows for large images. 

- Best candidate region proposals are selected based on combination of cues. 

**Objectness Results** : The cyan windows show the actual object bounding boxes. The yellow windows are the region proposals generated by the Objectness algorithm for different numbers of candidate windows. 

©Dr. Tony Scanlan 

# Selective Search & Image Segmentation 

- Based on Image Segmentation. 

   - If an image can be segmented then region proposals become much easier to identify. 

[2] Uijlings, J.R., Van De Sande, K.E., Gevers, T. and Smeulders, A.W., 2013. Selective search for object <u>recognition.</u> _International journal of computer vision_ , _104_ (2), pp.154-171 

<!-- Start of picture text -->
(b)<br>(a)<br><!-- End of picture text -->

**(c)** 

Image of Kittens<sup>(ii)</sup> (a), Segmented image (b) and region proposals (c). 

(ii) <u>Selective Search for Object Recognition (Slides)</u> 

©Dr. Tony Scanlan 

# Region Proposals & Scale 

- Creates region proposals at different scales in the image. 

   - Small to Large. 

- Start by over-segmenting the image. 

   - Result is that only the very smallest objects will be made of just one color. 

   - Many (small) candidate regions at this scale. 

**(b)** 

**(a)** 

**(c)** 

Input Image (a), Over-segmented image (b) and region proposals (c). [2] 

©Dr. Tony Scanlan 

# Combining Regions 

- Iteratively Combine “similar” segmented regions. 

- • This creates larger region proposals. (i.e. different object scales) 

(a) Iterative combinations of segmented regions and associated region proposals. (green boxes show successful object detection) [2] 

©Dr. Tony Scanlan 

# Region Similarity 

- Similarity of regions is based on: 

   - Colour Similarity (in original image). 

      - Histograms 

   - Texture Similarity (in original image). 

      - Gradients with a Hog like Descriptor. 

   - Size similarity. 

      - Similar sized regions 

   - Shape Compatibility. 

      - Shapes that fit together may form part of larger objects. 

- (a) Similar size regions [2] 

- (b) Shape Compatible regions [2] 

©Dr. Tony Scanlan 

# Object Recognition with Selective Search 

- SVM Trained (supervised) with region proposals & ground truth. • Use partially overlapping region proposals to provide difficult negative examples 

   - Retraining phase with false positives. 

- (a) Training the Object recognition pipeline with selective search [2].
