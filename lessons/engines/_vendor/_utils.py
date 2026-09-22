"""Shared formatting helpers for markdown/LaTeX output."""

import numpy as np


def fmt(v: int) -> str:
    return str(int(v))


def cell(v: int) -> str:
    """Format an integer for a markdown table cell."""
    return f'${int(v)}$'


def vec_list_str(arr) -> str:
    """Format 1D array as [-2, -2, -1]."""
    return '[' + ', '.join(fmt(v) for v in arr) + ']'


def index_range_str(start: int, end: int) -> str:
    """Format index set as [start, ..., end]."""
    return '[' + ', '.join(str(i) for i in range(start, end + 1)) + ']'


def pmatrix(arr) -> str:
    """Convert numpy array to LaTeX pmatrix.
    1-D array → row vector; 2-D array → matrix.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        inner = ' & '.join(fmt(v) for v in arr)
    else:
        rows = [' & '.join(fmt(v) for v in row) for row in arr]
        inner = r' \\ '.join(rows)
    return r'\begin{pmatrix}' + inner + r'\end{pmatrix}'


def md_table(headers: list, rows: list) -> str:
    """Build a markdown pipe table."""
    lines = [
        '| ' + ' | '.join(str(h) for h in headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |',
    ]
    for row in rows:
        lines.append('| ' + ' | '.join(str(c) for c in row) + ' |')
    return '\n'.join(lines)


def product_sum_latex(flat_a, flat_b) -> str:
    """Return LaTeX string for element-wise product sum: a0·b0 + a1·b1 + ..."""
    def wrap(v: int) -> str:
        return f'({v})' if v < 0 else str(v)

    terms = [f'{wrap(int(a))} \\cdot {wrap(int(b))}' for a, b in zip(flat_a, flat_b)]
    return ' + '.join(terms)
