"""Morphological closing question generator.

Closing = dilation followed by erosion with the same SE:
    I • B = (I ⊕ B) ⊖ B

Effect: fills small holes and gaps in the foreground, smooths inner contours.
The solution shows the intermediate dilation result before the final erosion.
"""
import numpy as np
from . import _morph_utils as mu

MARKS = {
    'sequence': 2,       # correctly identify dilation → erosion
    'intermediate': 4,   # correct dilated image
    'result': 4,         # correct final (eroded) image
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    se, shape, size = mu.make_se(rng, size=3)
    for _ in range(30):
        img = mu.make_blob(rng)
        dilated = mu.dilate(img, se)
        # Ensure dilation doesn't fill entire image (closing would equal input then)
        if dilated.sum() < img.size - 2:
            return {
                'img': img.tolist(),
                'se': se.tolist(),
                'se_shape': shape,
                'se_size': size,
            }
    img = mu.make_blob(rng)
    return {
        'img': img.tolist(),
        'se': se.tolist(),
        'se_shape': shape,
        'se_size': size,
    }


def compute(params):
    img = np.array(params['img'])
    se  = np.array(params['se'])
    dilated = mu.dilate(img, se)
    result  = mu.erode(dilated, se)
    return {
        'dilated': dilated.tolist(),
        'result':  result.tolist(),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img     = np.array(prms['img'])
    se      = np.array(prms['se'])
    dilated = np.array(c['dilated'])
    result  = np.array(c['result'])
    sname   = mu.SE_SHAPE_NAMES[prms['se_shape']]
    ssize   = prms['se_size']

    lines = [
        f'## Question{num_str}: Morphological Closing',
        '',
        f'The binary image $I$ below is processed by morphological **closing**'
        f' using the {sname} ${ssize} \\times {ssize}$ structuring element $B$.',
        '',
        '**Input image $I$:**',
        '',
        mu.fmt_matrix(img),
        '',
        '**Structuring element $B$:**',
        '',
        mu.fmt_matrix(se),
        '',
        'Compute the closed image $I \\bullet B$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$I \\bullet B =$',
            '',
        ]

    elif mode == 'answer':
        lines += [
            '**Result $I \\bullet B$:**',
            '',
            mu.fmt_matrix(result),
            '',
            '> **Hint:** Closing applies **dilation then erosion** with the same SE:'
            ' $I \\bullet B = (I \\oplus B) \\ominus B$.'
            ' It fills small holes and gaps in the foreground.',
        ]

    else:  # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Closing** applies dilation followed by erosion with the same SE:',
            '',
            '$$I \\bullet B = (I \\oplus B) \\ominus B$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['sequence'])]
        lines += [
            '',
            '**Step 1 — Dilation** $(I \\oplus B)$:',
            '',
            mu.fmt_matrix(dilated, 'I \\oplus B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['intermediate'])]
        lines += [
            '',
            '**Step 2 — Erosion of dilated result** $\\ominus B$:',
            '',
            mu.fmt_matrix(result, 'I \\bullet B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
