"""Generator for separable 2D convolution questions.

Factorises a kernel K = v * h^T into a column vector v and row vector h,
then applies them sequentially with one ring of zero-padding so the output
is the same size as the input image.

Modes:
    'blank'  – question only
    'answer' – question + answer matrix + hint
    'full'   – question + factorisation + step-by-step convolution
"""
import numpy as np
from scipy.signal import convolve2d
from ._utils import pmatrix
from ._exam_boxes import ruled_box

# Mark allocation per step — must sum to 10
MARKS = {
    'row_conv': 4,  # column kernel convolution (intermediate)
    'col_conv': 4,  # row kernel convolution
    'result': 2,    # final output
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, img_size=3, val_lo=-3, val_hi=3):
    rng = np.random.default_rng(seed)

    def _nonzero_vec(n):
        while True:
            v = rng.integers(val_lo, val_hi + 1, size=n)
            if np.any(v != 0):
                return v

    v = _nonzero_vec(3)
    h = _nonzero_vec(3)
    image = rng.integers(0, 5, size=(img_size, img_size))

    return {
        'v': v.astype(int),
        'h': h.astype(int),
        'kernel': np.outer(v, h).astype(int),
        'image': image.astype(int),
        'img_size': img_size,
    }


def compute(params):
    I = params['image']
    v = params['v'].reshape(-1, 1)
    h = params['h'].reshape(1, -1)

    I_padded = np.pad(I, 1, mode='constant').astype(int)
    I_int = convolve2d(I_padded, v, mode='valid').astype(int)
    O = convolve2d(I_int, h, mode='valid').astype(int)

    # Verify against full 2D convolution
    O_check = convolve2d(np.pad(I, 1, mode='constant'),
                         params['kernel'], mode='valid').astype(int)
    assert np.allclose(O, O_check), "Separable result mismatch"

    return {
        'I_padded': I_padded,
        'I_int': I_int,
        'result': O,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    n = p['img_size']

    v_str = pmatrix(p['v'])
    h_str = pmatrix(p['h'].reshape(1, -1))
    k_str = pmatrix(p['kernel'])
    v_col = pmatrix(p['v'].reshape(-1, 1))

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Separable Convolution'),
        '',
        f'Perform separable convolution between the ${n}\\times{n}$ image',
        '',
        f'$$I = {pmatrix(p["image"])}$$',
        '',
        'and the kernel',
        '',
        f'$$K = {k_str}$$',
        '',
        f'Use one ring of zero-padding so the output $O$ is the same size as $I$.',
        f'Give your answer as a ${n}\\times{n}$ matrix.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `_______________`')

    elif mode == 'exam':
        lines += [
            '**Part (a):** Intermediate result $I_{\\text{int}}$'
            ' (after convolving the zero-padded image with the column kernel $v$).',
            '',
            ruled_box(c['I_int'].shape[0]),
            '',
            '**Part (b):** Final output $O$'
            ' (after convolving $I_{\\text{int}}$ with the row kernel $h$).',
            '',
            ruled_box(c['result'].shape[0]),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Intermediate result.**',
            '',
            f'Inspect $K$: each row is a scalar multiple of the others, confirming $K$ is separable.',
            '',
            f'$$K = {k_str} = {v_col} \\times {h_str}$$',
            '',
            'So $v = ' + v_str + r'$ (column) and $h = ' + h_str + r'$ (row).',
            '',
            f'$$I_{{\\text{{pad}}}} = {pmatrix(c["I_padded"])}$$',
            '',
            f'Slide the flipped $v$ down each column (valid positions):',
            '',
            f'$$I_{{\\text{{int}}}} = I_{{\\text{{pad}}}} \\circledast {v_col} = {pmatrix(c["I_int"])}$$',
            '',
            _mark(MARKS['row_conv']),
        ]
        lines += [
            '',
            '**Part (b) - Final output.**',
            '',
            f'Slide the flipped $h$ along each row (valid positions):',
            '',
            f'$$O = I_{{\\text{{int}}}} \\circledast {h_str} = {pmatrix(c["result"])}$$',
            '',
            _mark(MARKS['col_conv'] + MARKS['result']),
        ]

    elif mode == 'answer':
        lines += [
            '**Answer:**',
            '',
            f'$$O = {pmatrix(c["result"])}$$',
            '',
            '> **Hint:** Factorise $K$ into a column vector $v$ and a row vector $h$'
            ' such that $K = v \\cdot h^T$. Zero-pad $I$, then convolve with $v$'
            ' (column-wise) to get an intermediate image, then convolve that with $h$'
            ' (row-wise) to get $O$. Remember to flip each 1-D kernel before sliding.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Factorise the kernel.**',
            '',
            f'Inspect $K$: each row is a scalar multiple of the others, confirming $K$ is separable.',
            '',
            f'$$K = {k_str} = {v_col} \\times {h_str}$$',
            '',
            'So $v = ' + v_str + r'$ (column) and $h = ' + h_str + r'$ (row).',
            '',
            '**Step 2 – Zero-pad the image.**',
            '',
            f'$$I_{{\\text{{pad}}}} = {pmatrix(c["I_padded"])}$$',
            '',
            '**Step 3 – Convolve with the column kernel $v$.**',
            '',
            f'Slide the flipped $v$ down each column (valid positions):',
            '',
            f'$$I_{{\\text{{int}}}} = I_{{\\text{{pad}}}} \\circledast {v_col} = {pmatrix(c["I_int"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['row_conv'])]
        lines += [
            '',
            '**Step 4 – Convolve with the row kernel $h$.**',
            '',
            f'Slide the flipped $h$ along each row (valid positions):',
            '',
            f'$$O = I_{{\\text{{int}}}} \\circledast {h_str} = {pmatrix(c["result"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['col_conv']), '', _mark(MARKS['result'])]

    return '\n'.join(lines)
