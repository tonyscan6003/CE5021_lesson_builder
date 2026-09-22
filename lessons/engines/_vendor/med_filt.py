"""Generator for 3×3 median filter questions (no zero-padding).

Given an N×M image, apply a 3×3 median filter with valid (no) padding,
producing an (N-2)×(M-2) output.

Modes:
    'blank'  – question only
    'answer' – question + answer matrix + hint
    'full'   – question + step-by-step solution showing sorted windows
"""
import numpy as np
from ._utils import pmatrix, md_table
from ._exam_boxes import ruled_box

# Mark allocation per step — must sum to 10
MARKS = {
    'sorted_windows': 5,  # sorted patch values for each window
    'medians': 5,         # median values and final output
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, rows=4, cols=4, val_lo=1, val_hi=9):
    rng = np.random.default_rng(seed)
    image = rng.integers(val_lo, val_hi + 1, size=(rows, cols)).astype(int)
    return {
        'image': image,
        'rows': rows,
        'cols': cols,
    }


def compute(params):
    img = params['image']
    rows, cols = params['rows'], params['cols']
    out_r = rows - 2
    out_c = cols - 2
    result = np.zeros((out_r, out_c), dtype=int)
    windows = {}
    for i in range(out_r):
        for j in range(out_c):
            patch = img[i:i + 3, j:j + 3].flatten()
            sorted_patch = sorted(patch.tolist())
            median = sorted_patch[4]  # 5th element (0-indexed: 4) of 9
            result[i, j] = median
            windows[(i, j)] = {
                'values': patch.tolist(),
                'sorted': sorted_patch,
                'median': median,
                'position': ('top-left', 'top-right', 'bottom-left', 'bottom-right')[
                    i * out_c + j] if out_r == 2 and out_c == 2 else f'({i},{j})',
            }
    return {'result': result, 'windows': windows, 'out_r': out_r, 'out_c': out_c}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    out_r, out_c = c['out_r'], c['out_c']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Median Filter'),
        '',
        f'Apply a $3 \\times 3$ median filter (no zero-padding) to the image',
        '',
        f'$$I = {pmatrix(p["image"])}$$',
        '',
        f'Give your answer as a ${out_r} \\times {out_c}$ matrix.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `_______________`')

    elif mode == 'exam':
        lines += [
            f'**Part (a):** For each of the ${out_r} \\times {out_c}$ window'
            ' positions, list the 9 sorted patch values and identify the median.',
            '',
            ruled_box(out_r * out_c, row_height_pt=28),
            '',
            '**Part (b):** Give the final output matrix.',
            '',
            ruled_box(out_r),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines.append(
            f'A $3\\times 3$ window slides over all valid positions '
            f'(no padding), giving a ${out_r}\\times{out_c}$ output.'
        )
        lines.append('')
        lines.append('**Part (a) - Sorted windows and medians.**')
        lines.append('')
        for (i, j), w in c['windows'].items():
            vals_str = ' '.join(str(v) for v in w['values'])
            sorted_str = ' '.join(str(v) for v in w['sorted'])
            pos = w['position']
            lines += [
                f'**Position {pos}** — $3\\times 3$ patch values: $\\{{{vals_str}\\}}$',
                '',
                f'Sorted: $[{sorted_str}]$ — median (5th value) $= {w["median"]}$',
                '',
            ]
        lines += [_mark(MARKS['sorted_windows']), '']
        lines += [
            '**Part (b) - Result.**',
            '',
            f'$$O = {pmatrix(c["result"])}$$',
            '',
            _mark(MARKS['medians']),
        ]

    elif mode == 'answer':
        lines += [
            '**Answer:**',
            '',
            f'$$O = {pmatrix(c["result"])}$$',
            '',
            '> **Hint:** Slide a $3\\times 3$ window across the image (valid positions only).'
            ' At each position collect the 9 pixel values, sort them, and take the'
            ' 5th value (the median).',
        ]

    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines.append(
            f'A $3\\times 3$ window slides over all valid positions '
            f'(no padding), giving a ${out_r}\\times{out_c}$ output.'
        )
        lines.append('')
        for (i, j), w in c['windows'].items():
            vals_str = ' '.join(str(v) for v in w['values'])
            sorted_str = ' '.join(str(v) for v in w['sorted'])
            pos = w['position']
            lines += [
                f'**Position {pos}** — $3\\times 3$ patch values: $\\{{{vals_str}\\}}$',
                '',
                f'Sorted: $[{sorted_str}]$ — median (5th value) $= {w["median"]}$',
                '',
            ]
        if marks:
            lines += [_mark(MARKS['sorted_windows']), '']
        lines += [
            f'**Result:**',
            '',
            f'$$O = {pmatrix(c["result"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['medians'])]

    return '\n'.join(lines)
