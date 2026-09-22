"""Generator for integral image questions.

Given a 5×5 image, compute the integral image and use it to find the sum
of the centre 3×3 region via the A − B − C + D formula.

Modes:
    'blank'  – question only
    'answer' – question + integral image + sum + hint
    'full'   – question + step-by-step construction + sum derivation
"""
import numpy as np
from ._utils import pmatrix

# Mark allocation per step — must sum to 10
MARKS = {
    'row_pass': 4,      # cumulative column sums (I1)
    'column_pass': 4,   # integral image (II)
    'verification': 2,  # region sum using A-B-C+D
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, rows=5, cols=5, val_lo=1, val_hi=6):
    rng = np.random.default_rng(seed)
    image = rng.integers(val_lo, val_hi + 1, size=(rows, cols)).astype(int)
    return {'image': image, 'rows': rows, 'cols': cols}


def compute(params):
    I = params['image'].astype(int)
    rows, cols = params['rows'], params['cols']

    # Intermediate: cumulative column sums
    I1 = np.cumsum(I, axis=0).astype(int)

    # Integral image: cumulative row sums of I1
    II = np.cumsum(I1, axis=1).astype(int)

    # Centre 3×3 region: rows 1-3, cols 1-3 (0-indexed)
    r1, r2, c1, c2 = 1, 3, 1, 3   # inclusive bounds of centre region
    A = int(II[r2, c2])
    B = int(II[r1 - 1, c2])
    C = int(II[r2, c1 - 1])
    D = int(II[r1 - 1, c1 - 1])
    region_sum = A - B - C + D

    # Cross-check
    direct_sum = int(np.sum(I[r1:r2 + 1, c1:c2 + 1]))
    assert region_sum == direct_sum

    return {
        'I1': I1, 'II': II,
        'A': A, 'B': B, 'C': C, 'D': D,
        'region_sum': region_sum,
        'A_pos': (r2, c2), 'B_pos': (r1 - 1, c2),
        'C_pos': (r2, c1 - 1), 'D_pos': (r1 - 1, c1 - 1),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    A, B, C, D = c['A'], c['B'], c['C'], c['D']

    lines = [
        f'## Question{num_str}: Integral Image',
        '',
        'Obtain the integral image of',
        '',
        f'$$I = {pmatrix(p["image"])}$$',
        '',
        'Then use the integral image to find the **sum of pixels in the centre $3\\times 3$ region**'
        ' (rows 2–4, columns 2–4, 1-indexed). Express your answer in the form $A-B-C+D$'
        ' where $A, B, C, D$ are values from the integral image.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Integral image:** `_______________`',
            '',
            '**Sum (A−B−C+D):** `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            '**Integral image:**',
            '',
            f'$$I_I = {pmatrix(c["II"])}$$',
            '',
            f'**Sum:** $A-B-C+D = {A}-{B}-{C}+{D} = {c["region_sum"]}$',
            '',
            '> **Hint:** Build the integral image in two passes: first cumulative-sum'
            ' the **columns** of $I$, then cumulative-sum the **rows** of that result.'
            ' The region sum is $A - B - C + D$ where $A$ is the bottom-right value,'
            ' $B$ is directly above the region, $C$ is directly to the left, and $D$'
            ' is above-left (corner) of the region in the integral image.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Column cumulative sums.**',
            '',
            'Sum each column from top to bottom to form intermediate image $I_1$:',
            '',
            f'$$I_1 = {pmatrix(c["I1"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['row_pass'])]
        lines += [
            '',
            '**Step 2 – Row cumulative sums.**',
            '',
            'Sum each row of $I_1$ from left to right to produce the integral image $I_I$:',
            '',
            f'$$I_I = {pmatrix(c["II"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['column_pass'])]
        lines += [
            '',
            '**Step 3 – Extract region sum using four corners.**',
            '',
            'For the centre $3\\times 3$ region (rows 2–4, cols 2–4, 1-indexed):',
            '',
            '| Symbol | Position (0-indexed) | Value |',
            '|--------|----------------------|-------|',
            f'| $A$ | row {c["A_pos"][0]}, col {c["A_pos"][1]} (bottom-right of region) | ${A}$ |',
            f'| $B$ | row {c["B_pos"][0]}, col {c["B_pos"][1]} (above region, same col) | ${B}$ |',
            f'| $C$ | row {c["C_pos"][0]}, col {c["C_pos"][1]} (left of region, same row) | ${C}$ |',
            f'| $D$ | row {c["D_pos"][0]}, col {c["D_pos"][1]} (above-left corner) | ${D}$ |',
            '',
            f'$$\\text{{Sum}} = A - B - C + D = {A} - {B} - {C} + {D} = {c["region_sum"]}$$',
            '',
            '*(Direct verification: summing the 9 values in the centre $3\\times 3$ patch'
            f' also gives {c["region_sum"]} ✓)*',
        ]
        if marks:
            lines += ['', _mark(MARKS['verification'])]

    return '\n'.join(lines)
