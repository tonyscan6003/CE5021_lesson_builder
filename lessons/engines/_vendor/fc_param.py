"""Generator for Fully Connected Layer Parameters questions.

N inputs, M outputs. Np = N*M + M (weights + biases).

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution
"""
import numpy as np
from ._exam_boxes import answer_box


# Mark allocation per step — must sum to 10
MARKS = {
    'weights': 5,  # weight and bias counts
    'total': 5,    # total parameter formula
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    N_choices = [128, 256, 512, 768, 1024]
    M_choices = [5, 10, 20, 50, 100]
    N = int(rng.choice(N_choices))
    M = int(rng.choice(M_choices))
    return {'N': N, 'M': M}


def compute(params):
    N, M = params['N'], params['M']
    Np = N * M + M
    return {'Np': Np}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    N, M = p['N'], p['M']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Fully Connected Layer Parameters'),
        '',
        f'A fully connected layer has $N = {N}$ inputs and $M = {M}$ outputs.',
        '',
        'How many **learnable parameters** does this layer have?'
        ' (Include both weights and biases.)',
        '',
    ]

    if mode == 'blank':
        lines.append('**$N_p =$ ** `_______________`')

    elif mode == 'exam':
        lines += [
            '**Part (a):** Number of weights ($N \\times M$) $=$',
            '',
            answer_box(),
            '',
            '**Part (b):** $N_p =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Weights and biases.**',
            '',
            'The fully connected layer connects every input to every output:',
            '',
            f'- Weights: $N \\times M = {N} \\times {M} = {N*M}$',
            f'- Biases: $M = {M}$',
            '',
            _mark(MARKS['weights']),
        ]
        lines += [
            '',
            '**Part (b) - Total parameters.**',
            '',
            f'$$N_p = N \\times M + M = {N} \\times {M} + {M} = {N*M} + {M} = {c["Np"]}$$',
            '',
            _mark(MARKS['total']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $N_p = {c["Np"]}$',
            '',
            '> **Hint:** The layer has $N \\times M$ weight parameters plus $M$ bias parameters.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            'The fully connected layer connects every input to every output:',
            '',
            f'- Weights: $N \\times M = {N} \\times {M} = {N*M}$',
            f'- Biases: $M = {M}$',
        ]
        if marks:
            lines += ['', _mark(MARKS['weights'])]
        lines += [
            '',
            f'$$N_p = N \\times M + M = {N} \\times {M} + {M} = {N*M} + {M} = {c["Np"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['total'])]

    return '\n'.join(lines)
