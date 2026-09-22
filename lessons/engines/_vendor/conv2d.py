"""Generator for 2D convolution questions (valid/no padding).

Convolves a kernel A with image B using valid padding (no zero-padding),
producing output of size (B_rows - A_rows + 1) x (B_cols - A_cols + 1).

The kernel is rotated 180° before sliding, consistent with the mathematical
definition of convolution (as opposed to cross-correlation).

Output modes:
    'full'   – question text + complete step-by-step solution
    'answer' – question text + final output matrix only
    'blank'  – question text only
"""

import numpy as np
from scipy.signal import convolve2d

from ._utils import (
    fmt, vec_list_str, index_range_str,
    pmatrix, md_table, product_sum_latex,
)

# Mark allocation per step — must sum to 10
MARKS = {
    'output_size': 2,  # output size determination
    'convolution': 8,  # full convolution computation
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------

def generate_params(
    seed=None,
    kernel_rows: int = 2,
    kernel_cols: int = 2,
    image_rows: int = 3,
    image_cols: int = 3,
    val_lo: int = -3,
    val_hi: int = 3,
) -> dict:
    """Return a parameter dict for one 2D convolution question."""
    rng = np.random.default_rng(seed)

    def rand_nonzero(r, c):
        while True:
            v = rng.integers(val_lo, val_hi + 1, size=(r, c))
            if np.any(v != 0):
                return v

    kernel = rand_nonzero(kernel_rows, kernel_cols)
    image = rand_nonzero(image_rows, image_cols)

    return {
        'kernel': kernel,
        'image': image,
        'kernel_rows': kernel_rows,
        'kernel_cols': kernel_cols,
        'image_rows': image_rows,
        'image_cols': image_cols,
    }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute(params: dict) -> dict:
    """Compute valid convolution result and per-position details."""
    kernel = params['kernel']
    image = params['image']

    result = convolve2d(image, kernel, mode='valid').astype(int)
    kernel_flipped = kernel[::-1, ::-1].astype(int)

    out_rows, out_cols = result.shape
    kr, kc = params['kernel_rows'], params['kernel_cols']

    position_details = []
    for i in range(out_rows):
        for j in range(out_cols):
            patch = image[i:i + kr, j:j + kc].astype(int)
            val = int(result[i, j])
            # Row-major flattened values for product formula
            patch_flat = patch.flatten().tolist()
            kf_flat = kernel_flipped.flatten().tolist()
            position_details.append({
                'i': i, 'j': j,
                'patch': patch,
                'patch_flat': patch_flat,
                'kf_flat': kf_flat,
                'val': val,
            })

    return {
        'result': result,
        'kernel_flipped': kernel_flipped,
        'position_details': position_details,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(params: dict, computed: dict, mode: str = 'full',
           question_num=None, marks=False) -> str:
    p, c = params, computed
    kr, kc = p['kernel_rows'], p['kernel_cols']
    ir, ic = p['image_rows'], p['image_cols']
    out_r = ir - kr + 1
    out_c = ic - kc + 1

    num_str = f' {question_num}' if question_num is not None else ''

    lines = [
        f'## Question{num_str}: 2D Convolution',
        '',
        f'Find the result of convolving the ${kr} \\times {kc}$ kernel',
        '',
        f'$$A = {pmatrix(p["kernel"])}$$',
        '',
        f'with the ${ir} \\times {ic}$ image',
        '',
        f'$$B = {pmatrix(p["image"])}$$',
        '',
        ('(Assuming no-padding — also called "Valid Padding".) '
         f'Give your answer as a ${out_r} \\times {out_c}$ matrix.'),
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `_______________`')
    elif mode == 'answer':
        lines += [
            '**Answer:**',
            '',
            f'$$C = {pmatrix(c["result"])}$$',
            '',
            r'> **Hint:** Rotate the kernel 180° (flip both axes). Slide it over'
            r' each valid patch of the image, multiply element-wise, and sum.'
            r' There is no padding — the output is smaller than the image.',
        ]
    else:
        lines += [
            '---',
            '',
            '### Solution',
            '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [_solution(params, computed)]
        if marks:
            lines += ['', _mark(MARKS['output_size']), '', _mark(MARKS['convolution'])]

    return '\n'.join(lines)


def _solution(params: dict, computed: dict) -> str:
    p, c = params, computed
    kr, kc = p['kernel_rows'], p['kernel_cols']
    kf = c['kernel_flipped']

    lines = [
        '**Step 1:** Flip the kernel 180° (rotate both rows and columns) before convolution:',
        '',
        f'$$A\'= {pmatrix(kf)}$$',
        '',
        ('**Step 2:** Slide the flipped kernel over the image. At each position, '
         'compute the element-wise product between the image patch and $A\'$, '
         'then sum to get the output value.'),
        '',
    ]

    for d in c['position_details']:
        i, j = d['i'], d['j']
        patch = d['patch']
        term_str = product_sum_latex(d['patch_flat'], d['kf_flat'])

        # Show which sub-image rows are involved
        row_range = f'{i}:{i + kr}'
        col_range = f'{j}:{j + kc}'

        lines += [
            f'**Position $O_{{{i}{j}}}$ — image patch $B[{row_range},\\,{col_range}]$:**',
            '',
            (f'Patch $= {pmatrix(patch)}$, '
             f'flipped kernel $A\' = {pmatrix(kf)}$'),
            '',
            f'$$O_{{{i}{j}}} = {term_str} = {d["val"]}$$',
            '',
        ]

    lines += [
        '**Result:**',
        '',
        f'$$C = {pmatrix(c["result"])}$$',
    ]

    return '\n'.join(lines)
