"""Convolution arithmetic: half (same) padding, unit stride (Relationship 3).

k must be odd. p = floor(k/2), s=1.  o = i (output equals input).
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
    i_choices = [8, 10, 12, 14, 16, 20, 24, 28, 32]
    k_odd_choices = [3, 5, 7, 9]   # k MUST be odd for half/same padding
    i = int(rng.choice(i_choices))
    k = int(rng.choice([kv for kv in k_odd_choices if kv < i]))
    p = k // 2
    return {'i': i, 'k': k, 'p': p, 's': 1}


def compute(params):
    i = params['i']
    k = params['k']
    p = k // 2
    return {'o': i, 'p': p}


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k = prms['i'], prms['k']
    p = k // 2

    lines = [
        f'## Question{num_str}: Convolution Output Size (Half / Same Padding, Unit Stride)',
        '',
        f'A ${i} \\times {i}$ input feature map is convolved with a ${k} \\times {k}$ kernel using',
        '**half (same) padding** and **unit stride** ($s = 1$).',
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
            f'**(b)** $o = {c["o"]}$ (output equals input size)',
            '',
            (f'> **Hint:** For half/same padding, $p = \\lfloor k/2 \\rfloor$ (requires $k$ odd),'
             ' and the output size equals the input size: $o = i$.'),
        ]

    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**(a) Determine padding.**',
            '',
            'For half/same padding the required padding is $p = \\lfloor k/2 \\rfloor$ (requires $k$ odd):',
            '',
            f'$$p = \\left\\lfloor \\frac{{{k}}}{{2}} \\right\\rfloor = {p}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['padding'])]
        lines += [
            '',
            '**(b) Determine output size.**',
            '',
            'Substituting into $o = (i - k) + 2p + 1$ with $p = \\lfloor k/2 \\rfloor = (k-1)/2$:',
            '',
            (f'$$o = ({i} - {k}) + 2 \\times {p} + 1'
             f' = {i - k} + {2 * p} + 1 = {c["o"]}$$'),
            '',
            f'The output is ${c["o"]} \\times {c["o"]}$ — equal to the input size.',
        ]
        if marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
