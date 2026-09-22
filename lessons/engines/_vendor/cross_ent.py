"""Generator for Cross Entropy Loss questions.

N=5 output classes, random probability vector (Dirichlet), random true class Y_c.
L = -(1/N) * log10(p_{Y_c})

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution
"""
import math
import numpy as np
from ._exam_boxes import answer_box


# Mark allocation per step — must sum to 10
MARKS = {
    'log_probs': 6,  # computing log probability
    'loss': 4,       # final loss value
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([1.0] * 5)
    # round to 2dp and renormalise
    p = [round(float(v), 2) for v in raw]
    total = sum(p)
    p = [round(v / total, 2) for v in p]
    # fix rounding so sum = 1 by adjusting last
    diff = round(1.0 - sum(p[:-1]), 2)
    p[-1] = diff
    Y_c = int(rng.integers(0, 5))
    return {'p': p, 'Y_c': Y_c}


def compute(params):
    p = params['p']
    Y_c = params['Y_c']
    N = 5
    p_true = p[Y_c]
    # guard against 0
    if p_true <= 0:
        p_true = 0.01
    L = round(-(1.0 / N) * math.log10(p_true), 2)
    return {'L': L, 'p_true': p_true, 'N': N}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    probs = p['p']
    Y_c = p['Y_c']

    prob_vec = '\\begin{pmatrix}' + ' & '.join(str(v) for v in probs) + '\\end{pmatrix}'
    one_hot = [1 if i == Y_c else 0 for i in range(5)]
    one_hot_vec = '\\begin{pmatrix}' + ' & '.join(str(v) for v in one_hot) + '\\end{pmatrix}'

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Cross-Entropy Loss'),
        '',
        'A neural network outputs the following probability vector for $N=5$ classes:',
        '',
        f'$$\\mathbf{{p}} = {prob_vec}$$',
        '',
        f'The true class is $Y_c = {Y_c}$ (0-indexed), giving one-hot vector:',
        '',
        f'$$\\mathbf{{y}} = {one_hot_vec}$$',
        '',
        'The cross-entropy loss is:',
        '',
        r'$$L = -\frac{1}{N} \sum_{c=0}^{N-1} y_c \log_{10}(p_c)$$',
        '',
        'Compute $L$ (round to 2 d.p.).',
        '',
    ]

    if mode == 'blank':
        lines.append('**$L =$ ** `_______________`')

    elif mode == 'exam':
        lines += [
            f'**Part (a):** $\\log_{{10}}(p_{{Y_c}}) =$',
            '',
            answer_box(),
            '',
            '**Part (b):** $L =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'Since $\\mathbf{{y}}$ is one-hot, only the term for class $Y_c = {Y_c}$ survives:',
            '',
            '**Part (a) - Log probability.**',
            '',
            f'$$\\log_{{10}}(p_{{{Y_c}}}) = \\log_{{10}}({c["p_true"]}) = {round(math.log10(c["p_true"]), 4)}$$',
            '',
            _mark(MARKS['log_probs']),
        ]
        lines += [
            '',
            '**Part (b) - Loss.**',
            '',
            f'$$L = -\\frac{{1}}{{{c["N"]}}} \\cdot 1 \\cdot ({round(math.log10(c["p_true"]), 4)}) = {c["L"]}$$',
            '',
            _mark(MARKS['loss']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $L = {c["L"]}$',
            '',
            '> **Hint:** Only the term for the true class survives the one-hot multiplication.'
            f' $L = -(1/{c["N"]}) \\log_{{10}}(p_{{Y_c}}) = -(1/{c["N"]}) \\log_{{10}}({c["p_true"]})$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'Since $\\mathbf{{y}}$ is one-hot, only the term for class $Y_c = {Y_c}$ survives:',
            '',
            f'$$L = -\\frac{{1}}{{{c["N"]}}} \\cdot y_{{{Y_c}}} \\cdot \\log_{{10}}(p_{{{Y_c}}})$$',
            '',
            f'$$= -\\frac{{1}}{{{c["N"]}}} \\cdot 1 \\cdot \\log_{{10}}({c["p_true"]})$$',
            '',
            f'$$= -\\frac{{1}}{{{c["N"]}}} \\cdot ({round(math.log10(c["p_true"]), 4)})$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['log_probs'])]
        lines += [
            '',
            f'$$= {c["L"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['loss'])]

    return '\n'.join(lines)
