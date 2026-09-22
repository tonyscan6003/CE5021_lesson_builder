"""Convolution arithmetic: zero padding, non-unit stride — general case (Relationship 6).

Given i, k, p, s > 1:  o = floor((i + 2p - k) / s) + 1
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
    i_choices = [7, 8, 9, 10, 12, 14, 15, 16, 20, 24, 28, 32]
    k_choices = [3, 4, 5, 6, 7]
    s_choices = [2, 3, 4]
    p_choices = [1, 2, 3]
    i = int(rng.choice(i_choices))
    k = int(rng.choice(k_choices))
    s = int(rng.choice(s_choices))
    p_valid = [pv for pv in p_choices if i + 2 * pv >= k]
    p = int(rng.choice(p_valid if p_valid else [1]))
    # ensure o >= 1
    while (i + 2 * p - k) < 0:
        p += 1
    return {'i': i, 'k': k, 's': s, 'p': p}


def compute(params):
    i, k, s, p = params['i'], params['k'], params['s'], params['p']
    return {'o': (i + 2 * p - k) // s + 1}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k, s, pad = prms['i'], prms['k'], prms['s'], prms['p']
    numerator = i + 2 * pad - k

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Convolution Output Size (Zero Padding, Non-Unit Stride)'),
        '',
        f'A ${i} \\times {i}$ input feature map is convolved with a ${k} \\times {k}$ kernel.',
        (f'The convolution uses **zero padding** $p = {pad}$ (on each side)'
         f' and **stride** $s = {s}$.'),
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
            ('> **Hint:** General case with zero padding and non-unit stride:'
             ' $o = \\lfloor (i + 2p - k) / s \\rfloor + 1$.'),
        ]

    elif mode == 'exam':
        lines += [
            '**Part (a):** State the relationship between output size $o$,'
            ' input size $i$, kernel size $k$, padding $p$ and stride $s$.',
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
            'For the general case with zero padding $p$ and non-unit stride $s$:',
            '',
            '$$o = \\left\\lfloor \\frac{i + 2p - k}{s} \\right\\rfloor + 1$$',
            '',
            _mark(MARKS['relationship']),
        ]
        lines += [
            '',
            '**Part (b) - Calculation.**',
            '',
            f'Substituting $i = {i}$, $k = {k}$, $p = {pad}$ and $s = {s}$:',
            '',
            (f'$$o = \\left\\lfloor \\frac{{{i} + 2 \\times {pad} - {k}}}{{{s}}} \\right\\rfloor + 1'
             f' = \\left\\lfloor \\frac{{{numerator}}}{{{s}}} \\right\\rfloor + 1'
             f' = {numerator // s} + 1 = {c["o"]}$$'),
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
            'For the general case with zero padding $p$ and non-unit stride $s$:',
            '',
            '$$o = \\left\\lfloor \\frac{i + 2p - k}{s} \\right\\rfloor + 1$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['relationship'])]
        lines += [
            '',
            f'Substituting $i = {i}$, $k = {k}$, $p = {pad}$ and $s = {s}$:',
            '',
            (f'$$o = \\left\\lfloor \\frac{{{i} + 2 \\times {pad} - {k}}}{{{s}}} \\right\\rfloor + 1'
             f' = \\left\\lfloor \\frac{{{numerator}}}{{{s}}} \\right\\rfloor + 1'
             f' = {numerator // s} + 1 = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$.',
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
