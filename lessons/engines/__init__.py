"""Figure engines for the CE6021 lessons.

REGISTRY lists every engine module in course order together with the lesson
it belongs to and the title shown in the Streamlit examples app.
"""

REGISTRY = [
    # (module path relative to this package, lesson id, title)
    ("s1_images.digital_images",         "01_01", "1.1 Digital images"),
    ("s1_images.convolution",            "01_02", "1.2 Convolution"),
    ("s1_images.sampling_interpolation", "01_03", "1.3 Sampling & interpolation"),
    ("s1_images.linear_nonlinear",       "01_04", "1.4 Linear & non-linear filters"),
    ("s1_images.morphology",             "01_05", "1.5 Morphology"),
    ("s1_images.fourier_1d",             "01_06", "1.6 Fourier series & DFT"),
    ("s1_images.fourier_2d",             "01_07", "1.7 2D DFT"),
    ("s2_edges.edge_detection",          "02_01", "2.1 Edge detection"),
    ("s2_edges.canny_hough",             "02_02", "2.2 Canny & Hough"),
    ("s2_edges.corners",                 "02_03", "2.3 Harris corners"),
    ("s2_edges.region_segmentation",     "02_04", "2.4 Region segmentation"),
    ("s2_edges.clustering_segmentation", "02_05", "2.5 Clustering segmentation"),
    ("s3_features.keypoints_log",        "03_01", "3.1 Keypoints & LoG"),
    ("s3_features.scale_space",          "03_02", "3.2 Scale space"),
    ("s3_features.descriptors",          "03_03", "3.3 Descriptors & matching"),
    ("s3_features.sift",                 "03_04", "3.4 SIFT"),
    ("s3_features.image_pyramids",       "03_05", "3.5 Image pyramids"),
    ("s4_detection.intro_detection",     "04_01", "4.1 Object detection"),
    ("s4_detection.machine_learning",    "04_02", "4.2 Machine learning"),
    ("s4_detection.classifiers",         "04_03", "4.3 Classifiers"),
    ("s4_detection.template_matching",   "04_04", "4.4 Template matching"),
    ("s4_detection.viola_jones",         "04_05", "4.5 Viola-Jones"),
    ("s4_detection.hog",                 "04_06", "4.6 HOG"),
    ("s5_faces.intro_fr",                "05_01", "5.1 Facial recognition"),
    ("s5_faces.pca",                     "05_02", "5.2 PCA"),
    ("s5_faces.eigenfaces",              "05_03", "5.3 Eigenfaces"),
    ("s5_faces.lda",                     "05_04", "5.4 LDA"),
    ("s5_faces.fisherfaces",             "05_05", "5.5 Fisherfaces"),
    ("s6_bof.bag_of_features",           "06_01", "6.1 Bag of features"),
    ("s6_bof.classification_retrieval",  "06_02", "6.2 Classification & retrieval"),
]
