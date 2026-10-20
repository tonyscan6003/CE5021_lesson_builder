"""Figure engines for the CE5021 lessons.

REGISTRY lists every engine module in course order together with the lesson
it belongs to and the title shown in the Streamlit examples app.

Only the lessons that have been built appear here; see COURSE_PLAN.md for the
full 17-lesson map and the build order.
"""

REGISTRY = [
    # (module path relative to this package, lesson id, title)
    ("s1_convnets.convolution",          "01_01", "1.1 2D convolution"),
    ("s1_convnets.fc_layers",            "01_02", "1.2 Fully connected layers"),
    ("s1_convnets.advanced_convolution", "01_03", "1.3 Advanced convolution"),
    ("s1_convnets.cnns",                 "01_04", "1.4 Convolutional neural networks"),
    ("s1_convnets.training",             "01_05", "1.5 Training neural networks"),
    ("s1_convnets.regularisation",       "01_06", "1.6 Overfitting and regularisation"),
    ("s2_classification.intro_classification", "02_01", "2.1 Image classification"),
    ("s2_classification.architectures",  "02_02", "2.2 Architectures"),
    ("s2_classification.resnets",        "02_03", "2.3 Residual networks"),
    ("s2_classification.transfer",       "02_04", "2.4 Transfer learning"),
    ("s2_classification.visualisation",  "02_05", "2.5 Visualisation & adversarial attack"),
    ("s3_detection.intro_detection",     "03_01", "3.1 Object detection metrics"),
    ("s3_detection.region_proposals",    "03_02", "3.2 Region proposals"),
    ("s3_detection.two_stage",           "03_03", "3.3 Two-stage detectors"),
]
