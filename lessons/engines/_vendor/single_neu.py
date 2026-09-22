"""Generator for Single Neuron questions (two parts).

Part (a): Linear activation y = z = sum(xi*wi) + b
Part (b): Sigmoid activation sigma(z) = 1/(1+e^-z)

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts
"""
import math
import numpy as np


# Mark allocation per step — must sum to 10
MARKS = {
    'weighted_sum': 6,  # weighted sum z
    'activation': 4,    # sigmoid activation output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    x = [round(float(rng.uniform(-3, 3)), 1) for _ in range(3)]
    w = [round(float(rng.uniform(-1, 1)), 1) for _ in range(3)]
    b = round(float(rng.uniform(-2, 2)), 1)
    return {'x': x, 'w': w, 'b': b}


def compute(params):
    x, w, b = params['x'], params['w'], params['b']
    z = round(sum(xi * wi for xi, wi in zip(x, w)) + b, 3)
    y_linear = z
    y_sigmoid = round(1.0 / (1.0 + math.exp(-z)), 2)
    return {'z': z, 'y_linear': y_linear, 'y_sigmoid': y_sigmoid}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    x, w, b = p['x'], p['w'], p['b']

    def _term(xi, wi):
        if xi < 0 and wi < 0:
            return f'({xi}) \\cdot ({wi})'
        elif xi < 0:
            return f'({xi}) \\cdot {wi}'
        elif wi < 0:
            return f'{xi} \\cdot ({wi})'
        return f'{xi} \\cdot {wi}'

    terms = ' + '.join(_term(xi, wi) for xi, wi in zip(x, w))
    b_str = f'+ ({b})' if b < 0 else f'+ {b}'

    lines = [
        f'## Question{num_str}: Single Neuron Activation',
        '',
        'A single neuron has the following inputs, weights and bias:',
        '',
        f'- Inputs: $x_1={x[0]},\\; x_2={x[1]},\\; x_3={x[2]}$',
        f'- Weights: $w_1={w[0]},\\; w_2={w[1]},\\; w_3={w[2]}$',
        f'- Bias: $b={b}$',
        '',
        'The pre-activation value is $z = \\sum_i x_i w_i + b$.',
        '',
        '**Part (a):** Using a **linear** activation $y = z$, compute $y$. (Round to 3 d.p.)',
        '',
        '**Part (b):** Using the **sigmoid** activation $y = \\sigma(z) = \\dfrac{1}{1+e^{-z}}$,'
        ' compute $y$. (Round to 2 d.p.)',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** $y =$ `_______________`',
            '',
            '**Part (b):** $y =$ `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $y = z = {c["y_linear"]}$',
            '',
            f'**Part (b):** $y = \\sigma({c["z"]}) = {c["y_sigmoid"]}$',
            '',
            '> **Hint:** First compute $z = x_1 w_1 + x_2 w_2 + x_3 w_3 + b$.'
            ' For (a) output $y = z$ directly.'
            ' For (b) apply $\\sigma(z) = 1/(1+e^{-z})$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 - Compute the weighted sum.**',
            '',
            f'$$z = {terms} {b_str} = {c["z"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['weighted_sum'])]
        lines += [
            '',
            '**Part (a) - Linear activation.**',
            '',
            f'$$y = z = {c["y_linear"]}$$',
            '',
            '**Part (b) - Sigmoid activation.**',
            '',
            f'$$y = \\sigma(z) = \\frac{{1}}{{1 + e^{{-{c["z"]}}}}}'
            f' = {c["y_sigmoid"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['activation'])]

    return '\n'.join(lines)
