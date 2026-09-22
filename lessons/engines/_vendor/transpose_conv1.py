"""Generator for Transpose Convolution (stride 1) questions.

Input X (2x2), kernel K (2x2). Equivalent to padding X with k-1=1 ring of zeros
then convolving with K (valid conv) to get 3x3 output.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution
"""
import numpy as np
from scipy.signal import convolve2d
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'output_size': 4,   # padded input (output size derivation)
    'computation': 6,   # final convolution output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    while True:
        K = rng.integers(-2, 3, size=(2, 2)).tolist()
        if any(K[i][j] != 0 for i in range(2) for j in range(2)):
            break
    X = rng.integers(1, 6, size=(2, 2)).tolist()
    return {'K': K, 'X': X}


def compute(params):
    K = np.array(params['K'])
    X = np.array(params['X'])
    k = K.shape[0]  # 2
    # Pad X with k-1=1 ring of zeros -> 4x4
    p = k - 1
    X_pad = np.pad(X, p, mode='constant')
    # Valid convolution (scipy convolve2d with mode='valid')
    Y = convolve2d(X_pad, K, mode='valid')
    return {
        'X_pad': X_pad.tolist(),
        'Y': Y.tolist(),
        'pad': p,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    K = np.array(p['K'])
    X = np.array(p['X'])
    X_pad = np.array(c['X_pad'])
    Y = np.array(c['Y'])

    lines = [
        f'## Question{num_str}: Transpose Convolution (Stride 1)',
        '',
        f'Compute the **transpose convolution** (stride $s=1$) of input $X = {pmatrix(X)}$'
        f' with kernel $K = {pmatrix(K)}$.',
        '',
        'Use the equivalent direct convolution approach:',
        f'pad $X$ with $p\' = k-1 = {c["pad"]}$ ring(s) of zeros, then apply standard valid convolution.',
        '',
        'Give the $3 \\times 3$ output $Y$.',
        '',
    ]

    if mode == 'blank':
        lines.append('**$Y =$ ** `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $Y = {pmatrix(Y)}$',
            '',
            '> **Hint:** Pad the input with $k-1$ zeros on all sides, then perform standard valid convolution with $K$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'**Step 1 - Pad $X$ with $p\' = k-1 = {c["pad"]}$ ring of zeros.**',
            '',
            f'$$X_{{pad}} = {pmatrix(X_pad)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output_size'])]
        lines += [
            '',
            f'**Step 2 - Apply valid convolution with $K$.**',
            '',
            f'Convolve the $4 \\times 4$ padded input with the $2 \\times 2$ kernel:',
            f'$$Y = X_{{pad}} * K = {pmatrix(Y)}$$',
            '',
            'Output size: $(4 - 2 + 1) \\times (4 - 2 + 1) = 3 \\times 3$. ✓',
        ]
        if marks:
            lines += ['', _mark(MARKS['computation'])]

    return '\n'.join(lines)
