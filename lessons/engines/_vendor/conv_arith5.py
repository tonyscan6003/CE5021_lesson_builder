"""Convolution arithmetic: no zero padding, non-unit stride (Relationship 5).

Given i, k, s > 1, p = 0:  o = floor((i - k) / s) + 1
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
    k_choices = [2, 3, 4, 5]
    s_choices = [2, 3, 4]
    i = int(rng.choice(i_choices))
    k = int(rng.choice([kv for kv in k_choices if kv < i]))
    s = int(rng.choice(s_choices))
    # ensure o >= 1: (i - k) // s >= 0 is guaranteed when i >= k
    return {'i': i, 'k': k, 's': s, 'p': 0}


def compute(params):
    i, k, s = params['i'], params['k'], params['s']
    return {'o': (i - k) // s + 1}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k, s = prms['i'], prms['k'], prms['s']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Convolution Output Size (No Padding, Non-Unit Stride)'),
        '',
        f'A ${i} \\times {i}$ input feature map is convolved with a ${k} \\times {k}$ kernel.',
        f'The convolution uses **no zero padding** ($p = 0$) and **stride** $s = {s}$.',
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
            ('> **Hint:** For no padding and non-unit stride:'
             ' $o = \\lfloor (i - k) / s \\rfloor + 1$.'),
        ]

    elif mode == 'exam':
        lines += [
            '**Part (a):** State the relationship between output size $o$,'
            ' input size $i$, kernel size $k$ and stride $s$ for no padding.',
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
            'For no zero padding ($p = 0$) and non-unit stride $s$, the relationship is:',
            '',
            '$$o = \\left\\lfloor \\frac{i - k}{s} \\right\\rfloor + 1$$',
            '',
            _mark(MARKS['relationship']),
        ]
        lines += [
            '',
            '**Part (b) - Calculation.**',
            '',
            f'Substituting $i = {i}$, $k = {k}$ and $s = {s}$:',
            '',
            (f'$$o = \\left\\lfloor \\frac{{{i} - {k}}}{{{s}}} \\right\\rfloor + 1'
             f' = \\left\\lfloor \\frac{{{i - k}}}{{{s}}} \\right\\rfloor + 1'
             f' = {(i - k) // s} + 1 = {c["o"]}$$'),
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
            'For no zero padding ($p = 0$) and non-unit stride $s$, the relationship is:',
            '',
            '$$o = \\left\\lfloor \\frac{i - k}{s} \\right\\rfloor + 1$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['relationship'])]
        lines += [
            '',
            f'Substituting $i = {i}$, $k = {k}$ and $s = {s}$:',
            '',
            (f'$$o = \\left\\lfloor \\frac{{{i} - {k}}}{{{s}}} \\right\\rfloor + 1'
             f' = \\left\\lfloor \\frac{{{i - k}}}{{{s}}} \\right\\rfloor + 1'
             f' = {(i - k) // s} + 1 = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$.',
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
