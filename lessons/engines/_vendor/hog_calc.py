"""Generator for HOG descriptor vector length questions.

Given cell size, block dimensions (in cells), number of orientation bins,
and the number of overlapping blocks, compute the total descriptor length.

Length = n_blocks_x × n_blocks_y × cells_per_block_x × cells_per_block_y × n_orientations

Modes:
    'blank'  – question only
    'answer' – question + answer + hint
    'full'   – question + step-by-step derivation
"""
import numpy as np
from ._exam_boxes import answer_box

# Mark allocation per step — must sum to 10
MARKS = {
    'gradients': 4,  # short vector per block
    'histogram': 4,  # total descriptor length computation
    'normalised': 2, # normalised result note
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


_ORIENT_CHOICES = [6, 8, 9, 10, 12]


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    cell_px = int(rng.choice([4, 5, 6, 8]))          # pixels per cell (side)
    cpb = int(rng.integers(2, 8))                      # cells per block (square)
    n_orient = int(rng.choice(_ORIENT_CHOICES))
    n_blocks_x = int(rng.integers(8, 25))
    n_blocks_y = int(rng.integers(8, 25))
    return {
        'cell_px': cell_px,
        'cpb': cpb,
        'n_orient': n_orient,
        'n_blocks_x': n_blocks_x,
        'n_blocks_y': n_blocks_y,
    }


def compute(params):
    p = params
    short_vec = p['cpb'] * p['cpb'] * p['n_orient']
    length = p['n_blocks_x'] * p['n_blocks_y'] * short_vec
    return {'short_vec': short_vec, 'length': length}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: HOG Descriptor Vector Length'),
        '',
        'A Histogram of Oriented Gradients descriptor is configured as follows:',
        '',
        f'- Cell size: ${p["cell_px"]} \\times {p["cell_px"]}$ pixels',
        f'- Block size: ${p["cpb"]} \\times {p["cpb"]}$ cells',
        f'- Orientation bins per cell: ${p["n_orient"]}$',
        f'- Overlapping blocks: ${p["n_blocks_x"]} \\times {p["n_blocks_y"]}$',
        '',
        'What is the **total length** of the descriptor vector?',
        '',
    ]

    if mode == 'blank':
        lines.append('**Length =** `_______________`')

    elif mode == 'exam':
        lines += [
            '**Part (a):** Short vector length per block $=$',
            '',
            answer_box(),
            '',
            '**Part (b):** Total descriptor length $=$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Short vector length per block.**',
            '',
            f'Each block contains ${p["cpb"]} \\times {p["cpb"]} = {p["cpb"]**2}$ cells.'
            f' Each cell has a histogram of length ${p["n_orient"]}$, so:',
            '',
            f'$$\\text{{short vector}} = {p["cpb"]} \\times {p["cpb"]} \\times {p["n_orient"]}'
            f' = {c["short_vec"]}$$',
            '',
            _mark(MARKS['gradients']),
        ]
        lines += [
            '',
            '**Part (b) - Total descriptor length.**',
            '',
            f'There are ${p["n_blocks_x"]} \\times {p["n_blocks_y"]} = {p["n_blocks_x"]*p["n_blocks_y"]}$'
            ' overlapping blocks, so:',
            '',
            f'$$\\text{{Length}} = {p["n_blocks_x"]} \\times {p["n_blocks_y"]}'
            f' \\times {c["short_vec"]} = {c["length"]}$$',
            '',
            _mark(MARKS['histogram'] + MARKS['normalised']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $\\text{{Length}} = {c["length"]}$',
            '',
            '> **Hint:** Each cell contributes one histogram with length = number of'
            ' orientation bins. Each block concatenates all its cell histograms giving'
            ' a per-block vector of length (cells per block)$^2 \\times$ orientations.'
            ' Multiply by the total number of overlapping blocks for the full descriptor.'
            ' Note: the cell pixel size affects feature resolution, not vector length.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Short vector length per block.**',
            '',
            f'Each block contains ${p["cpb"]} \\times {p["cpb"]} = {p["cpb"]**2}$ cells.'
            f' Each cell has a histogram of length ${p["n_orient"]}$, so:',
            '',
            f'$$\\text{{short vector}} = {p["cpb"]} \\times {p["cpb"]} \\times {p["n_orient"]}'
            f' = {c["short_vec"]}$$',
            '',
            '*(The cell pixel size affects feature resolution, not the vector length.)*',
        ]
        if marks:
            lines += ['', _mark(MARKS['gradients'])]
        lines += [
            '',
            '**Step 2 – Total descriptor length.**',
            '',
            f'There are ${p["n_blocks_x"]} \\times {p["n_blocks_y"]} = {p["n_blocks_x"]*p["n_blocks_y"]}$'
            ' overlapping blocks, so:',
            '',
            f'$$\\text{{Length}} = {p["n_blocks_x"]} \\times {p["n_blocks_y"]}'
            f' \\times {c["short_vec"]} = {c["length"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['histogram']), '', _mark(MARKS['normalised'])]

    return '\n'.join(lines)
