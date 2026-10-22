"""Colours, fonts and sizes of the Q&A decks.

The colours are those of the lesson pages (lessons/build/style.css) and of the
figure palette (lessons/engines/_common.py, Okabe-Ito), so that the decks, the
HTML lessons and the figures look like one course.
"""
THEME = {
    # colours (hex, no '#')
    "ink": "1F2933",            # body text       (style.css --ink)
    "muted": "5F6B7A",          # captions        (--muted)
    "accent": "0072B2",         # accent          (--accent; C["blue"])
    "accent_dark": "005A8C",
    "accent_soft": "E8F1F8",    # card fill       (--accent-soft)
    "accent_light": "9CCBEA",
    "bg_alt": "F6F8FA",         # code background (--bg-alt)
    "dark": "1F2933",           # title-slide background
    "muted_dark": "B8C2CC",
    "white": "FFFFFF",
    "epoch": "0072B2",          # per-epoch schedulers, as in the figures (C["blue"])
    "batch": "A8457F",          # per-batch schedulers: a darker C["purple"] for text
    # fonts (shipped with Office)
    "head_font": "Calibri",
    "body_font": "Calibri",
    "code_font": "Courier New",
    # sizes (pt)
    "title": 34,
    "body": 15,
    "code": 13,
    "small": 11,
}
