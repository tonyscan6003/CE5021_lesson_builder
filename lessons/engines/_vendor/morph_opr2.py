"""Morphological dilation question generator.

Given a binary image I and structuring element B, compute I ⊕ B.
A pixel in the output is 1 iff B, centred there, has any overlap with the
foreground of I.
"""
import numpy as np
from . import _morph_utils as mu

MARKS = {
    'rule': 2,
    'result': 8,
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    se, shape, size = mu.make_se(rng, size=3)
    img = mu.make_blob(rng)
    dilated = mu.dilate(img, se)
    # Retry if dilation fills the entire image (uninteresting)
    for _ in range(20):
        if dilated.sum() < img.size:
            break
        img = mu.make_blob(rng)
        dilated = mu.dilate(img, se)
    return {
        'img': img.tolist(),
        'se': se.tolist(),
        'se_shape': shape,
        'se_size': size,
    }


def compute(params):
    img = np.array(params['img'])
    se  = np.array(params['se'])
    return {'result': mu.dilate(img, se).tolist()}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img    = np.array(prms['img'])
    se     = np.array(prms['se'])
    result = np.array(c['result'])
    sname  = mu.SE_SHAPE_NAMES[prms['se_shape']]
    ssize  = prms['se_size']

    lines = [
        f'## Question{num_str}: Morphological Dilation',
        '',
        f'The binary image $I$ below is processed by morphological **dilation**'
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
        'Compute the dilated image $I \\oplus B$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$I \\oplus B =$',
            '',
        ]

    elif mode == 'answer':
        lines += [
            '**Result $I \\oplus B$:**',
            '',
            mu.fmt_matrix(result),
            '',
            '> **Hint:** A pixel at position $(r,c)$ in $I \\oplus B$ is **1** if and only if'
            ' $B$, when centred at $(r,c)$, overlaps at least one foreground pixel of $I$.'
            ' Equivalently, each foreground pixel of $I$ "paints" the SE shape into the output.',
        ]

    else:  # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Dilation rule.** The dilated image $I \\oplus B$ is defined as:',
            '',
            r'$$[I \oplus B](r,c) = \begin{cases} 1 & \text{if } B \text{ centred at } (r,c)'
            r' \cap \text{foreground}(I) \neq \emptyset \\ 0 & \text{otherwise} \end{cases}$$',
            '',
            'Equivalently, for each foreground pixel $(r,c)$ in $I$, all pixels covered'
            ' by $B$ centred at $(r,c)$ are set to 1 in the output (clipped to image bounds).',
        ]
        if marks:
            lines += ['', _mark(MARKS['rule'])]
        lines += [
            '',
            '**Result:**',
            '',
            mu.fmt_matrix(result, 'I \\oplus B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
