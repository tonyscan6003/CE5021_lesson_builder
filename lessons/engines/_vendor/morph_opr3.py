"""Morphological opening question generator.

Opening = erosion followed by dilation with the same SE:
    I ∘ B = (I ⊖ B) ⊕ B

Effect: smooths contours, removes thin protrusions and isolated foreground
pixels smaller than the SE.  The solution shows the intermediate erosion
result before the final dilation.
"""
import numpy as np
from . import _morph_utils as mu
from ._exam_boxes import answer_box, ruled_box

MARKS = {
    'sequence': 2,       # correctly identify erosion → dilation
    'intermediate': 4,   # correct eroded image
    'result': 4,         # correct final (dilated) image
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    se, shape, size = mu.make_se(rng, size=3)
    for _ in range(40):
        img = mu.make_blob(rng)
        eroded = mu.erode(img, se)
        if eroded.sum() >= 4:       # enough pixels survive erosion
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
    eroded = mu.erode(img, se)
    result = mu.dilate(eroded, se)
    return {
        'eroded': eroded.tolist(),
        'result': result.tolist(),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img    = np.array(prms['img'])
    se     = np.array(prms['se'])
    eroded = np.array(c['eroded'])
    result = np.array(c['result'])
    sname  = mu.SE_SHAPE_NAMES[prms['se_shape']]
    ssize  = prms['se_size']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Morphological Opening'),
        '',
        f'The binary image $I$ below is processed by morphological **opening**'
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
        'Compute the opened image $I \\circ B$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$I \\circ B =$',
            '',
        ]

    elif mode == 'exam':
        lines += [
            '**Part (a):** State the sequence of operations for opening.',
            '',
            answer_box(),
            '',
            '**Part (b):** Compute the intermediate eroded image $I \\ominus B$.',
            '',
            ruled_box(eroded.shape[0]),
            '',
            '**Part (c):** Compute the final opened image $I \\circ B$.',
            '',
            ruled_box(result.shape[0]),
            '',
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Sequence.**',
            '',
            '$$I \\circ B = (I \\ominus B) \\oplus B$$',
            '',
            _mark(MARKS['sequence']),
        ]
        lines += [
            '',
            '**Part (b) - Erosion** $(I \\ominus B)$:',
            '',
            mu.fmt_matrix(eroded, 'I \\ominus B'),
            '',
            _mark(MARKS['intermediate']),
        ]
        lines += [
            '',
            '**Part (c) - Dilation of eroded result** $\\oplus B$:',
            '',
            mu.fmt_matrix(result, 'I \\circ B'),
            '',
            _mark(MARKS['result']),
        ]

    elif mode == 'answer':
        lines += [
            '**Result $I \\circ B$:**',
            '',
            mu.fmt_matrix(result),
            '',
            '> **Hint:** Opening applies **erosion then dilation** with the same SE:'
            ' $I \\circ B = (I \\ominus B) \\oplus B$.'
            ' It smooths the boundary and removes foreground regions smaller than $B$.',
        ]

    else:  # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Opening** applies erosion followed by dilation with the same SE:',
            '',
            '$$I \\circ B = (I \\ominus B) \\oplus B$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['sequence'])]
        lines += [
            '',
            '**Step 1 — Erosion** $(I \\ominus B)$:',
            '',
            mu.fmt_matrix(eroded, 'I \\ominus B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['intermediate'])]
        lines += [
            '',
            '**Step 2 — Dilation of eroded result** $\\oplus B$:',
            '',
            mu.fmt_matrix(result, 'I \\circ B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
