"""Generator for Softmax Activation questions.

Given a random z vector of length 5, compute e^zi for each element,
sum S = sum(e^zi), then sigma(zi) = e^zi / S.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution
"""
import math
import numpy as np
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'exp_values': 4,  # exponential values
    'sum_exp': 2,     # sum of exponentials
    'softmax': 4,     # final softmax outputs
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    z = [round(float(rng.uniform(-2, 5)), 1) for _ in range(5)]
    return {'z': z}


def compute(params):
    z = params['z']
    exp_z = [round(math.exp(zi), 4) for zi in z]
    S = round(sum(exp_z), 4)
    sigma = [round(e / S, 2) for e in exp_z]
    return {'exp_z': exp_z, 'S': S, 'sigma': sigma}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    z = p['z']

    z_vec = '\\begin{pmatrix}' + ' & '.join(str(zi) for zi in z) + '\\end{pmatrix}'
    sigma_vec = '\\begin{pmatrix}' + ' & '.join(str(s) for s in c['sigma']) + '\\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Softmax Activation',
        '',
        'Compute the **softmax** activation for the following input vector:',
        '',
        f'$$\\mathbf{{z}} = {z_vec}$$',
        '',
        'The softmax function is defined as:',
        '',
        r'$$\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}$$',
        '',
        'Give each output rounded to 2 d.p. as a $1 \\times 5$ row vector.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** $\\sigma(\\mathbf{z}) =$ `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $\\sigma(\\mathbf{{z}}) = {sigma_vec}$',
            '',
            '> **Hint:** Compute $e^{{z_i}}$ for each element, sum them to get $S$,'
            ' then divide each $e^{{z_i}}$ by $S$.',
        ]

    else:
        exp_terms = ',\\; '.join(
            f'e^{{{zi}}} = {ei}' for zi, ei in zip(z, c['exp_z'])
        )
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 - Compute exponentials.**',
            '',
            f'$$\\quad {exp_terms}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['exp_values'])]
        lines += [
            '',
            '**Step 2 - Compute the sum.**',
            '',
            f'$$S = ' + ' + '.join(str(e) for e in c['exp_z']) + f' = {c["S"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['sum_exp'])]
        lines += [
            '',
            '**Step 3 - Divide each exponential by S.**',
            '',
        ]
        for i, (zi, ei, si) in enumerate(zip(z, c['exp_z'], c['sigma'])):
            lines.append(f'$$\\sigma(z_{{{i+1}}}) = \\frac{{{ei}}}{{{c["S"]}}} = {si}$$')
            lines.append('')
        lines.append(f'$$\\sigma(\\mathbf{{z}}) = {sigma_vec}$$')
        if marks:
            lines += ['', _mark(MARKS['softmax'])]

    return '\n'.join(lines)
