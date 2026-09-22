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
]
