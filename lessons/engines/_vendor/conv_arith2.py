"""Convolution arithmetic: arbitrary zero padding, unit stride (Relationship 2).

Given i, k, p, s=1:  o = (i - k) + 2p + 1
"""
import numpy as np
from ._exam_boxes import answer_box

MARKS = {
    'relationship': 4,
    'result': 6,
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    i_choices = [7, 8, 9, 10, 12, 14, 16, 20, 24, 28]
    k_choices = [3, 4, 5, 6, 7]
    p_choices = [1, 2, 3]
    i = int(rng.choice(i_choices))
    k = int(rng.choice(k_choices))
    p_valid = [pv for pv in p_choices if i + 2 * pv >= k]
    p = int(rng.choice(p_valid if p_valid else [1]))
    return {'i': i, 'k': k, 's': 1, 'p': p}


def compute(params):
    i, k, p = params['i'], params['k'], params['p']
    return {'o': (i - k) + 2 * p + 1}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k, pad = prms['i'], prms['k'], prms['p']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Convolution Output Size (Zero Padding, Unit Stride)'),
        '',
        f'A ${i} \\times {i}$ input feature map is convolved with a ${k} \\times {k}$ kernel.',
        f'The convolution uses **zero padding** $p = {pad}$ (on each side) and **unit stride** ($s = 1$).',
        '',
        'What is the output size $o \\times o$?',
        '',
    ]

    if mode == 'blank':
        lines.append('$o =$ `___`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $o = {c["o"]}$, giving a ${c["o"]} \\times {c["o"]}$ output.',
            '',
            '> **Hint:** For zero padding $p$ and unit stride: $o = (i - k) + 2p + 1$.',
        ]

    elif mode == 'exam':
        lines += [
            '**Part (a):** State the relationship between output size $o$,'
            ' input size $i$, kernel size $k$ and padding $p$ for unit stride.',
            '',
            '$o =$',
            '',
            answer_box(),
            '',
            '**Part (b):** Using this relationship, calculate the output size $o$.',
            '',
            '$o =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Relationship.**',
            '',
            'For zero padding $p$ and unit stride ($s = 1$), the relationship is:',
            '',
            '$$o = (i - k) + 2p + 1$$',
            '',
            _mark(MARKS['relationship']),
        ]
        lines += [
            '',
            '**Part (b) - Calculation.**',
            '',
            f'Substituting $i = {i}$, $k = {k}$ and $p = {pad}$:',
            '',
            (f'$$o = ({i} - {k}) + 2 \\times {pad} + 1'
             f' = {i - k} + {2 * pad} + 1 = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$.',
            '',
            _mark(MARKS['result']),
        ]

    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            'For zero padding $p$ and unit stride ($s = 1$), the relationship is:',
            '',
            '$$o = (i - k) + 2p + 1$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['relationship'])]
        lines += [
            '',
            f'Substituting $i = {i}$, $k = {k}$ and $p = {pad}$:',
            '',
            (f'$$o = ({i} - {k}) + 2 \\times {pad} + 1'
             f' = {i - k} + {2 * pad} + 1 = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$.',
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
