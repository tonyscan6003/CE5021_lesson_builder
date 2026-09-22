"""Shared handwritten-answer-box HTML builders for 'exam' mode.

'exam' mode renders a printed, pen-and-paper exam paper (as opposed to
'blank', which is the digital/D2L-quiz variant with plain `___` text blanks).
These builders produce boxed answer areas sized for handwriting instead.

Raw HTML block/inline elements pass through markdown-it unprocessed (see
build_paper.py's _protect_math and build_exam.py's header rendering), so
these are plain inline-styled <div>/<span> elements, not markdown syntax.
"""

_BORDER = "1.5px solid #333"
# Keeps a box from being split across a PDF page boundary (which otherwise
# strands a lone row on the next page with the rest of it blank).
_NO_BREAK = "break-inside:avoid; page-break-inside:avoid;"
MIN_ROW_PT = 24


def answer_box(height_pt: int = MIN_ROW_PT) -> str:
    """A single full-width boxed answer area (block-level, own line)."""
    return (
        f'<div style="border:{_BORDER}; border-radius:4px; '
        f'min-height:{height_pt}pt; width:100%; margin:6px 0; '
        f'box-sizing:border-box; {_NO_BREAK}"></div>'
    )


def ruled_box(rows: int, row_height_pt: int = MIN_ROW_PT) -> str:
    """A full-width boxed area with `rows` ruled lines, for a matrix answer."""
    rows = max(rows, 1)
    inner = ''.join(
        f'<div style="height:{row_height_pt}pt;'
        + ('border-bottom:1px solid #999;' if i < rows - 1 else '')
        + '"></div>'
        for i in range(rows)
    )
    return (
        f'<div style="border:{_BORDER}; border-radius:4px; '
        f'width:100%; margin:6px 0; box-sizing:border-box; {_NO_BREAK}">{inner}</div>'
    )


def inline_box(width_pt: int = 160, height_pt: int = MIN_ROW_PT) -> str:
    """A short inline boxed answer area, for pairing with a label on one line
    (e.g. several short values side by side: $G_x =$ [box], $G_y =$ [box])."""
    return (
        f'<span style="display:inline-block; border:{_BORDER}; border-radius:4px; '
        f'min-height:{height_pt}pt; min-width:{width_pt}pt; vertical-align:middle; '
        f'margin:0 4px; box-sizing:border-box;"></span>'
    )


def cell_box(height_pt: int = MIN_ROW_PT) -> str:
    """A compact boxed answer area sized to fit inside a markdown table cell."""
    return (
        f'<span style="display:inline-block; border:{_BORDER}; border-radius:3px; '
        f'min-height:{height_pt}pt; min-width:56pt; vertical-align:middle;"></span>'
    )
