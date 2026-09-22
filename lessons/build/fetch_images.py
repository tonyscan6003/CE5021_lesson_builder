#!/usr/bin/env python3
"""Fetch source images from Wikimedia Commons and record their credits.

    python lessons/build/fetch_images.py            # fetch everything in data/sources.yaml
    python lessons/build/fetch_images.py --only lion cat
    python lessons/build/fetch_images.py --search "Lion d'Afrique"   # find a file title

data/sources.yaml lists each image key:

    lion:
      commons: "Lion d'Afrique.jpg"      # exact File: title on Commons
      width: 1024                        # requested thumbnail width
      author: "Clément Bardot"           # optional override when Commons has no Artist
      note: "old slide 1.1 (viii)"       # optional, provenance only
    brick:
      file: brick.png                    # a non-Commons image already in data/
      title: "Bricks 25"
      author: "CC0Textures"
      licence: "CC0 1.0"
      licence_url: https://creativecommons.org/publicdomain/zero/1.0/
      source_url: https://cc0textures.com

Output: data/credits.yaml, keyed like sources.yaml, with file, title, author,
licence, licence_url and source_url. Engines load images by key and the site
builder prints the credits under every figure and at the foot of each page.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

LESSONS = Path(__file__).resolve().parents[1]
DATA = LESSONS / "data"
SOURCES = DATA / "sources.yaml"
CREDITS = DATA / "credits.yaml"
API = "https://commons.wikimedia.org/w/api.php"
HDR = {"User-Agent": "CE6021-lesson-builder/0.1 (scanlantg@gmail.com)"}


def api(**params) -> dict:
    params["format"] = "json"
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=HDR)
    return json.load(urllib.request.urlopen(req, timeout=60))


def strip_html(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(s or ""))).strip()


def wikitext_field(title: str, field: str) -> str:
    """Fallback: read |author= (etc.) from the file page's wikitext."""
    d = api(action="query", titles="File:" + title, prop="revisions", rvprop="content", rvslots="main")
    p = next(iter(d["query"]["pages"].values()))
    try:
        text = p["revisions"][0]["slots"]["main"]["*"]
    except (KeyError, IndexError):
        return ""
    m = re.search(rf"\|\s*{field}\s*=\s*(.+)", text, re.I)
    if not m:
        return ""
    val = m.group(1).split("\n")[0]
    val = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", val)      # [[link|text]] -> text
    val = re.sub(r"\{\{[^}]*\}\}", "", val)                           # drop templates
    return strip_html(val).strip(" |")


def commons_info(title: str, width: int) -> dict:
    d = api(action="query", titles="File:" + title, prop="imageinfo",
            iiprop="url|extmetadata|size", iiurlwidth=width)
    p = next(iter(d["query"]["pages"].values()))
    if "imageinfo" not in p:
        raise KeyError(f"not found on Commons: File:{title}")
    ii = p["imageinfo"][0]
    m = ii["extmetadata"]
    g = lambda k: strip_html(m.get(k, {}).get("value", ""))  # noqa: E731
    author = g("Artist") or g("Credit") or wikitext_field(title, "author")
    author = re.sub(r"^(Taken (?:&|and) submitted by|Taken by the uploader,?|Photo by|Photograph by)\s*", "",
                    author, flags=re.I).strip()
    return dict(
        title=g("ObjectName") or Path(title).stem.replace("_", " "),
        author=author,
        licence=g("LicenseShortName"),
        licence_url=g("LicenseUrl"),
        source_url=ii["descriptionurl"],
        download_url=ii.get("thumburl") or ii["url"],
        ext=Path(title).suffix.lower() or ".jpg",
    )


def fetch(key: str, spec: dict) -> dict:
    if "commons" in spec:
        info = commons_info(spec["commons"], int(spec.get("width", 1024)))
        ext = ".jpg" if info["ext"] in (".jpg", ".jpeg") else (".png" if info["ext"] == ".png" else ".jpg")
        target = DATA / f"{key}{ext}"
        if not target.exists():
            req = urllib.request.Request(info["download_url"], headers=HDR)
            target.write_bytes(urllib.request.urlopen(req, timeout=120).read())
        credit = dict(file=target.name, title=spec.get("title", info["title"]),
                      author=spec.get("author") or info["author"] or "unknown",
                      licence=spec.get("licence", info["licence"]),
                      licence_url=spec.get("licence_url", info["licence_url"]),
                      source_url=info["source_url"])
        if spec.get("trim_border"):
            credit["trim_border"] = True
    elif "skimage" in spec:
        credit = {k: spec[k] for k in ("title", "author", "licence", "licence_url", "source_url")}
        credit["file"] = f"skimage:{spec['skimage']}"
    elif "dataset" in spec:
        # an external dataset fetched by the engine itself (e.g. via scikit-learn)
        credit = {k: spec[k] for k in ("title", "author", "licence", "licence_url", "source_url")}
        credit["file"] = f"dataset:{spec['dataset']}"
    else:
        credit = {k: spec[k] for k in ("file", "title", "author", "licence", "licence_url", "source_url")}
        if not (DATA / credit["file"]).exists():
            raise FileNotFoundError(DATA / credit["file"])
    if credit["author"] in ("", "unknown"):
        print(f"  [WARN] {key}: no author found; add an `author:` override in sources.yaml")
    return credit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--search", help="search Commons and print candidate file titles")
    ap.add_argument("--force", action="store_true", help="re-download files")
    args = ap.parse_args()

    if args.search:
        d = api(action="query", list="search", srsearch=args.search, srnamespace=6, srlimit=8)
        for r in d["query"]["search"]:
            t = r["title"][5:]
            try:
                i = commons_info(t, 400)
                print(f"{t!r:70s} {i['author'][:30]!r:34s} {i['licence']}")
            except Exception as exc:  # noqa: BLE001
                print(f"{t!r}: {exc}")
        return

    sources = yaml.safe_load(SOURCES.read_text(encoding="utf-8"))
    credits = yaml.safe_load(CREDITS.read_text(encoding="utf-8")) if CREDITS.exists() else {}
    keys = args.only or list(sources)
    for key in keys:
        spec = sources[key]
        if args.force and "commons" in spec:
            for f in DATA.glob(f"{key}.*"):
                f.unlink()
        print(f"{key}: ", end="", flush=True)
        try:
            credits[key] = fetch(key, spec)
            print(f"{credits[key]['file']}  {credits[key]['author']}  {credits[key]['licence']}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {exc}")
            sys.exit(1)
    # drop credits for keys removed from sources
    credits = {k: v for k, v in credits.items() if k in sources}
    CREDITS.write_text(yaml.safe_dump(credits, allow_unicode=True, sort_keys=True), encoding="utf-8")
    print(f"wrote {CREDITS}")


if __name__ == "__main__":
    main()
