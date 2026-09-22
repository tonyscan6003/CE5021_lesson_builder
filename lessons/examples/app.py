"""CE6021 interactive examples.

Run from the repository root:

    streamlit run lessons/examples/app.py

Every lesson engine that defines a `demo()` function becomes a page.
"""
import importlib
import sys
from pathlib import Path

import streamlit as st

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from lessons.engines import REGISTRY  # noqa: E402

st.set_page_config(page_title="CE6021 examples", layout="wide")

pages = []
for modpath, lesson_id, title in REGISTRY:
    mod = importlib.import_module(f"lessons.engines.{modpath}")
    if hasattr(mod, "demo"):
        pages.append(st.Page(mod.demo, title=title, url_path=lesson_id))

if not pages:
    st.error("No demos registered.")
else:
    st.navigation(pages).run()
