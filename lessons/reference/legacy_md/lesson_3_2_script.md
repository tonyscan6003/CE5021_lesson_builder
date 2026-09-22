# lesson_3_2_script

# Region Proposals 

## Slide 1: Title Slide 

<mark>In this video lesson, we will return to the topic of object detection and discuss region proposals, which are methods of determining the location of objects in an image and are an alternative to using sliding windows and image pyramids.</mark> 

## Slide 2: <mark>Object Detection</mark> 

<mark>In the object detection task, we have the problem of where to look for an object in an image. Normally we don't have any information as to the location of objects of interest. In the Histogram of Gradients and Viola Jones algorithms, a sliding window is used to exhaustively search the image for specific objects. Both detectors also used an image pyramid (shown in figure b) with a sliding window to find objects at different scales in an image. In the HoG pedestrian detector, the classifier step is always applied at each detector window position. This approach is computationally inefficient when the window doesn't contain any object of interest. In comparison, the Viola Jones face detector used a cascade of classifiers to quickly reject background image regions and save on wasteful computation. In general, for the object detection task, it is preferable to have an object detection stage, which can identify the location of candidate objects, and only send these candidate regions to the classifier. This approach will minimise the wasted computation. In the methods shown in this lesson, the goal is to directly find different sized regions of an image containing an object. These are called region proposals.</mark> 

## Slide 3: <mark>Region Proposals</mark> 

<mark>As an alternative to the sliding window based approaches, region proposal based algorithms can be used to find objects in images prior to classifying the objects. Region proposals are selected regions of the input image that are very likely to contain an object. As shown in figure (a), region proposals (the blue boxes) are drawn around objects of interest by the region proposal generator. On this slide, we will discuss the characteristics of a good region proposal algorithm. Ideally, the region proposal algorithm will generate far less images to be sent to the classifier than the sliding window and image pyramid approach.  The region proposal algorithm should also be able to find any type of objects of any size or scale in an image. The region proposal algorithm should also generate a low number of false positive proposal regions, as this helps reduce the number of calls to the classifier, as well as the overall precision of the object detector. Ideally the number of false negative regions is low, as this corresponds to missed objects that the classifier cannot recover. A low number of false negatives contributes to the recall performance of the object detection system.</mark> 

## Slide 4: <mark>Objectness [1] (Multi Scale Saliency)</mark> 

<mark>The first region proposal algorithm that we will discuss is objectness. Objectness is a measure that quantifies how likely it is for an image window to cover an object of any class. The complete algorithm uses 5 image cues that can be combined to determine if a proposed region is likely to contain an image. No single cue works perfectly for all cases, for this reason it was necessary to combine several cues to improve the robustness of the algorithm. The first cue is called multiscale saliency; salient pixels are those that stand out from their surroundings. A saliency map can be generated at different scales using the Fourier transform. We will not include the detail of how this map is generated but, instead, look at the results in the figure. The large giraffe in the image (a) appears as a large blob in the high or coarse scale and in figure (b), the giraffes head and small</mark> 

<mark>aeroplane in the background appear in the fine or low scale saliency map. Thus, both the large and small object are found, which is what we want. In contrast in the street scene in image (d), the cars in the image are not particularly visible at either of the scales shown figures (e) and (f).</mark> 

## Slide 5: Objectness (Colour Contrast) 

<mark>On this slide, we will look at colour contrast as another cue to find objects. There is often a high degree of colour contrast between an object and its surroundings. As seen in image (a), the sheep are white and the field is green. This difference in contrast is detected by comparing the histograms in the LAB colour space from inside the candidate window (that is inside cyan box) to the ring of pixels from just outside the window between yellow and cyan boxes. If the histograms are sufficiently different, an object is detected inside the proposed region. As shown in the figure, the sheep and train can be easily detected. However, the bird on the branch in figure (c) is difficult to detect due to similarity in colour between the object and its surroundings.</mark> 

## Slide 6: <mark>Objectness (Edge Density)</mark> 

<mark>Another cue to find objects in the objectness algorithm is Edge density. In this cue, the image is first passed into the canny edge detector. The density of edges is measured in the ring between the cyan and yellow boxes. This cue operates by assuming that objects should have many edges in their outer ring compared to a uniform region. As seen in the figure, the region around the bus and plane contain many edges and an object can assumed to be detected. However, as shown in the image with the sailboat, the ripples on the water create a false positive as they produce many edges similar to a true object.</mark> 

## Slide 7: <mark>Objectness (SuperPixel Straddling)</mark> 

<mark>The forth cue is based superpixel straddling. Superpixels are a type of segmentation where pixels are grouped by similarity in colour and texture. The figure (a) shows the image of a dog divided into superpixels. The key property of superpixels is that they do not straddle an objects boundary, as can be seen in the image of the dog. While superpixels divide up regions of the dog's head and body, the edges of the pixels always align with the border of the dog. The cue to detect if the proposed regions include an object, use this key property of superpixels. For each proposed window, the amount of a superpixel that appears inside and outside the boundary is compared. If a lot of the superpixel is outside the region then this is likely to cut across an object rather than contain it. As shown in figure (b), the window w3 is a better window than w1 for finding the ship as w3 straddles the light green superpxiels at the bottom of the image.</mark> 

## Slide 8: Objectness (Location and Size) 

<mark>The segmentation based superpixel straddling is the best performing cue in the objectness algorithm. As we will shortly see, segmentation is also used as the basis of the selective search region proposal algorithm. The figure on this page shows more examples of objects detected using superpixel straddling. However, as shown in figure (c) and (f), the boundaries of small objects may not be preserved and false negatives can occur. The final cue is the location and size of the proposed region itself. These are initially randomly generated. So regions that are more likely to contain an object, such as square windows in the centre of an image, are retained while small windows in the corner of images are rejected.</mark> 

## Slide 9: Region Proposals with Objectness 

<mark>Over the last few slides, we have looked at various cues that can be used to determine that an object may be present in a region of an image. It is complex to train a region proposal generator with these objectness cues. However, the results as shown on the figure are impressive. As mentioned already, the initial region proposal are generated at random and the regions likely to contain an object are selected based on their score with a combination of the cues. By generating more random candidate windows, the overall performance of the algorithm is improved as can be seen in the figure. This does increase the computational overhead, but an exhaustive search would be even slower and could contain 100k windows for large images.</mark> 

## Slide 10: Selective Search & Image Segmentation 

<mark>The second region proposal algorithm we will look at is selective search. This is based on image segmentation. As shown in the figure, if the image of kittens can segmented initially, we can just determine the region proposal windows from the segmented image. We no longer have to generate random windows and evaluate them to see if they contain an object or not, as in the objectness method.</mark> 

## Slide 11: Region Proposals & Scale 

<mark>The selective search method creates region proposals at different scales in the image. Starting with small regions and working up to larger regions. The algorithm starts by over segmenting the image. The image of the girl and the TV in figure (a) is over segmented into image (b). The result of this oversegmentation is that the very small objects will be made of just one colour. As shown in figure (c), there are many small candidate regions at this scale, many of which won't contain an object.</mark> 

## Slide 12: Combining Regions 

<mark>After the initial over segmentation of the image, the algorithm then iteratively combines "similar" segmented regions to create larger regions. We will see on the next slides how the similarity between regions is determined. The figure (a) on the slide shows the iterative combination process. It can be seen as the smaller regions are joined, complete objects are found. The green boxes show where the classifier has later positively identified an object. The blue boxes are region proposals, which are drawn around each segmented region. These may contain an object or a partial object.</mark> 

## Slide 13: Region Similarity 

<mark>The similarity of regions is based on a number of measures. The colour similarity of the segmented regions in the original image are compared. Histograms of the pixel intensity in the segmented regions are used to check for colour similarity. Texture similarity of the regions is compared with a HOG gradients like descriptor. As shown in figure (a) and (b), the size and shape compatibility of regions is also assessed as part of determining if two regions are similar and should be joined. A training procedure is required to train the selective search algorithm to produce good regions that correspond to actual objects.</mark> 

## Slide 14: Object Recognition with Selective Search 

<mark>This final slide shows how we can train an object recognition system that includes selective search. The training dataset and procedure for training this object recognition system is a little more complex than what we have seen previously. We will use a support vector machine as the classifier in our system. As shown on the left hand side of figure (a), we are supplied a set of labelled training images with bounding boxes drawn around the location of the objects. These ground truth images are used to generate positive examples (that is, regions fully containing the object) to send to the SVM. Also shown on the left-hand side of figure (a) is the selective search, which generates a range of region proposals. The region proposals from the selective search that only overlap 20-50% with</mark> 

<mark>the ground truth are retained and used to create difficult negative examples for SVM training. The SVM is trained with these positive and negative examples. After training, a search is performed for any false positives and these are added to the training dataset and the algorithm retrained to improve its performance.</mark>
