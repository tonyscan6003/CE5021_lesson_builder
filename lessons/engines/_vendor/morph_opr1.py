"""Morphological erosion question generator.

Given a binary image I and structuring element B, compute I ⊖ B.
A pixel in the output is 1 iff B, centred there, lies entirely within the
foreground of I.
"""
import numpy as np
from . import _morph_utils as mu
from ._exam_boxes import answer_box, ruled_box

MARKS = {
    'rule': 2,
    'result': 8,
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    # Keep 3×3 SE for erosion so the blob is not trivially wiped out
    se, shape, size = mu.make_se(rng, size=3)
    for _ in range(30):
        img = mu.make_blob(rng)
        eroded = mu.erode(img, se)
        if eroded.sum() >= 3:           # non-trivial result
            return {
                'img': img.tolist(),
                'se': se.tolist(),
                'se_shape': shape,
                'se_size': size,
            }
    # Fallback: accept whatever we have
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
    return {'result': mu.erode(img, se).tolist()}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img    = np.array(prms['img'])
    se     = np.array(prms['se'])
    result = np.array(c['result'])
    sname  = mu.SE_SHAPE_NAMES[prms['se_shape']]
    ssize  = prms['se_size']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Morphological Erosion'),
        '',
        f'The binary image $I$ below is processed by morphological **erosion**'
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
        'Compute the eroded image $I \\ominus B$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$I \\ominus B =$',
            '',
        ]

    elif mode == 'exam':
        lines += [
            '**Part (a):** State the erosion rule'
            ' (when is a pixel in $I \\ominus B$ equal to 1?).',
            '',
            answer_box(),
            '',
            '**Part (b):** Compute the eroded image $I \\ominus B$.',
            '',
            ruled_box(result.shape[0]),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Erosion rule.**',
            '',
            r'$$[I \ominus B](r,c) = \begin{cases} 1 & \text{if } B \text{ centred at } (r,c)'
            r' \subseteq \text{foreground of } I \\ 0 & \text{otherwise} \end{cases}$$',
            '',
            'Pixels where $B$ extends beyond the image border count as background.',
            '',
            _mark(MARKS['rule']),
        ]
        lines += [
            '',
            '**Part (b) - Result.**',
            '',
            mu.fmt_matrix(result, 'I \\ominus B'),
            '',
            _mark(MARKS['result']),
        ]

    elif mode == 'answer':
        lines += [
            '**Result $I \\ominus B$:**',
            '',
            mu.fmt_matrix(result),
            '',
            '> **Hint:** A pixel at position $(r,c)$ in $I \\ominus B$ is **1** if and only if'
            ' $B$, when centred at $(r,c)$, is entirely contained within the foreground'
            ' (all covered pixels of $I$ are 1).  Otherwise it is **0**.'
            ' Pixels where $B$ extends outside the image boundary are treated as **0**.',
        ]

    else:  # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Erosion rule.** The eroded image $I \\ominus B$ is defined as:',
            '',
            r'$$[I \ominus B](r,c) = \begin{cases} 1 & \text{if } B \text{ centred at } (r,c)'
            r' \subseteq \text{foreground of } I \\ 0 & \text{otherwise} \end{cases}$$',
            '',
            'Pixels where $B$ extends beyond the image border count as background.',
        ]
        if marks:
            lines += ['', _mark(MARKS['rule'])]
        lines += [
            '',
            '**Result:**',
            '',
            mu.fmt_matrix(result, 'I \\ominus B'),
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
