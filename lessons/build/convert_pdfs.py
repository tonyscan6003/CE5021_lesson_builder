#!/usr/bin/env python3
"""Convert a folder of typeset PDFs to markdown.

Used to turn lecture slide decks and narration scripts into the plain-text
reference under `lessons/reference/legacy_md/`, which is the source material
for the lesson prose. Extracted from the old `grader_tool/main.py` so the
conversion survives that tool's removal.

Text-layer extraction only, no OCR: it is meant for typeset PDFs such as
exported slides, not for scans or handwriting.

    python3 lessons/build/convert_pdfs.py \\
        --input-dir path/to/pdfs --output-dir lessons/reference/legacy_md [--force]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def convert_pdf(pdf_path: Path, out_dir: Path) -> Path:
    """Convert one typeset PDF to markdown and return the written path.

    Uses pymupdf4llm for layout-aware markdown (headings, lists, tables) and
    falls back to plain per-page text from PyMuPDF if it is not installed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{pdf_path.stem}.md"
    try:
        import pymupdf4llm
        body = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=False, show_progress=False)
    except ImportError:
        import fitz
        doc = fitz.open(str(pdf_path))
        body = "\n\n---\n\n".join(
            f"## Page {i}\n\n{page.get_text().strip()}"
            for i, page in enumerate(doc, start=1)
        )
        doc.close()
    body = re.sub(r"\n{3,}", "\n\n", body).strip()      # collapse blank-line runs
    md_path.write_text(f"# {pdf_path.stem}\n\n{body}\n", encoding="utf-8")
    return md_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", required=True, dest="input_dir",
                    help="folder containing the PDFs")
    ap.add_argument("--output-dir", required=True, dest="output_dir",
                    help="folder to write <stem>.md files into")
    ap.add_argument("--force", action="store_true",
                    help="overwrite markdown files that already exist")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not in_dir.is_dir():
        sys.exit(f"[ERROR] input directory not found: {in_dir}")
    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"[ERROR] no PDFs found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Converting {len(pdfs)} PDF(s): {in_dir} -> {out_dir}")
    ok = failed = 0
    for pdf in pdfs:
        target = out_dir / f"{pdf.stem}.md"
        if target.exists() and not args.force:
            print(f"  skip (exists) {target.name}")
            ok += 1
            continue
        print(f"  {pdf.name} ...", end=" ", flush=True)
        try:
            convert_pdf(pdf, out_dir)
            print("done")
            ok += 1
        except Exception as exc:                        # noqa: BLE001 - report and continue
            print(f"FAILED: {exc}")
            failed += 1
    print(f"\nConversion complete: {ok} succeeded, {failed} failed.")


if __name__ == "__main__":
    main()
