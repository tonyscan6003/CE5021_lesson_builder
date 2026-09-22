"""Generator for Conv Layer Parameters (no padding, stride>1) questions (three parts).

Part (a): Output size [o, o, K].
Part (b): Num parameters Np = k*k*C*K.
Part (c): Num multiplications Nm = Np * o * o.

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'output_size': 3,       # output spatial size
    'params': 3,            # number of learnable parameters
    'multiplications': 4,   # number of multiplications
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    i_choices = [28, 32, 40, 48, 56, 60, 64]
    k_choices = [3, 5, 7]
    s_choices = [2, 3]
    K_choices = [32, 64, 96, 128, 256]
    C_choices = [16, 32, 48, 64, 88, 96]

    i_val = int(rng.choice(i_choices))
    k_val = int(rng.choice(k_choices))
    s_val = int(rng.choice(s_choices))
    K_val = int(rng.choice(K_choices))
    C_val = int(rng.choice(C_choices))

    return {'i': i_val, 'k': k_val, 's': s_val, 'K': K_val, 'C': C_val}


def compute(params):
    i, k, s, K, C = params['i'], params['k'], params['s'], params['K'], params['C']
    o = (i - k) // s + 1
    Np = k * k * C * K
    Nm = Np * o * o
    return {'o': o, 'Np': Np, 'Nm': Nm}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    i, k, s, K, C = p['i'], p['k'], p['s'], p['K'], p['C']

    lines = [
        f'## Question{num_str}: Convolutional Layer Parameters (No Padding, Stride > 1)',
        '',
        'A convolutional layer has the following configuration:',
        '',
        f'- Input: ${i} \\times {i} \\times {C}$ (square, $C={C}$ channels)',
        f'- Kernel size: ${k} \\times {k}$',
        f'- Stride: $s = {s}$',
        f'- Number of kernels (output channels): $K = {K}$',
        f'- Padding: $p = 0$ (no padding)',
        '',
        '**Part (a):** What is the **output size** $[o, o, K]$?',
        '',
        '**Part (b):** How many **learnable parameters** does this layer have?',
        '',
        '**Part (c):** How many **multiplications** are performed per forward pass?',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** Output size = `___`',
            '',
            '**Part (b):** $N_p =$ `___`',
            '',
            '**Part (c):** $N_m =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $[{c["o"]}, {c["o"]}, {K}]$',
            '',
            f'**Part (b):** $N_p = {c["Np"]}$',
            '',
            f'**Part (c):** $N_m = {c["Nm"]}$',
            '',
            '> **Hint:** Output size: $o = \\lfloor(i-k)/s\\rfloor+1$.'
            ' Parameters: $N_p = k \\times k \\times C \\times K$.'
            ' Multiplications: $N_m = N_p \\times o \\times o$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Output size.**',
            '',
            f'$$o = \\left\\lfloor\\frac{{i - k}}{{s}}\\right\\rfloor + 1'
            f' = \\left\\lfloor\\frac{{{i} - {k}}}{{{s}}}\\right\\rfloor + 1'
            f' = \\left\\lfloor\\frac{{{i-k}}}{{{s}}}\\right\\rfloor + 1 = {c["o"]}$$',
            '',
            f'Output shape: $[{c["o"]}, {c["o"]}, {K}]$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output_size'])]
        lines += [
            '',
            '**Part (b) - Number of parameters.**',
            '',
            f'$$N_p = k \\times k \\times C \\times K = {k} \\times {k} \\times {C} \\times {K} = {c["Np"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['params'])]
        lines += [
            '',
            '**Part (c) - Number of multiplications.**',
            '',
            f'$$N_m = N_p \\times o \\times o = {c["Np"]} \\times {c["o"]} \\times {c["o"]} = {c["Nm"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['multiplications'])]

    return '\n'.join(lines)
