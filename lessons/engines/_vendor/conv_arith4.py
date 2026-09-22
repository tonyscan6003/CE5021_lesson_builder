"""Convolution arithmetic: full padding, unit stride (Relationship 4).

p = k - 1, s = 1.  o = i + (k - 1)  (output is larger than input).
"""
import numpy as np

MARKS = {
    'padding': 4,
    'result': 6,
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    i_choices = [4, 5, 6, 7, 8, 10, 12, 14]
    k_choices = [2, 3, 4, 5, 6]
    i = int(rng.choice(i_choices))
    k = int(rng.choice(k_choices))
    p = k - 1
    return {'i': i, 'k': k, 'p': p, 's': 1}


def compute(params):
    i, k = params['i'], params['k']
    return {'o': i + (k - 1), 'p': k - 1}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k = prms['i'], prms['k']
    p = k - 1

    lines = [
        f'## Question{num_str}: Convolution Output Size (Full Padding, Unit Stride)',
        '',
        f'A ${i} \\times {i}$ input feature map is convolved with a ${k} \\times {k}$ kernel using',
        '**full padding** and **unit stride** ($s = 1$).',
        '',
        '(a) What zero padding $p$ must be applied on each side?',
        '(b) What is the output size $o \\times o$?',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$p =$ `___`',
            '',
            '$o =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**(a)** $p = {p}$',
            '',
            f'**(b)** $o = {c["o"]}$',
            '',
            '> **Hint:** For full padding, $p = k - 1$ and $o = i + (k - 1)$.',
        ]

    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**(a) Determine padding.**',
            '',
            'Full padding ensures every partial overlap of the kernel over the input is included,',
            'requiring:',
            '',
            f'$$p = k - 1 = {k} - 1 = {p}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['padding'])]
        lines += [
            '',
            '**(b) Determine output size.**',
            '',
            'Substituting $p = k - 1$ into $o = (i - k) + 2p + 1$:',
            '',
            (f'$$o = i + (k - 1) = {i} + ({k} - 1)'
             f' = {i} + {k - 1} = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$ (larger than the input).',
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
