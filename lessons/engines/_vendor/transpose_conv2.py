"""Generator for Transpose Convolution (stride 2) questions.

Input X (2x2), kernel K (3x3), stride s=2.
Equivalent: stretch X (insert s-1=1 zeros between elements -> 3x3),
pad with p'=k-1=2 zeros -> 7x7, valid conv with K -> 5x5 output.

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
    'output_size': 4,   # stretch + pad (output size derivation)
    'computation': 6,   # final convolution output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    while True:
        K = rng.integers(-2, 3, size=(3, 3)).tolist()
        if any(K[i][j] != 0 for i in range(3) for j in range(3)):
            break
    X = rng.integers(1, 6, size=(2, 2)).tolist()
    return {'K': K, 'X': X, 's': 2}


def compute(params):
    K = np.array(params['K'])
    X = np.array(params['X'])
    s = params['s']
    k = K.shape[0]  # 3

    # Stretch X: insert s-1=1 zeros between elements -> 3x3
    xi, xj = X.shape
    si = (xi - 1) * s + 1
    sj = (xj - 1) * s + 1
    X_stretched = np.zeros((si, sj), dtype=int)
    for i in range(xi):
        for j in range(xj):
            X_stretched[i * s, j * s] = X[i, j]

    # Pad with p'=k-1=2 zeros -> 7x7
    p = k - 1
    X_pad = np.pad(X_stretched, p, mode='constant')

    # Valid convolution
    Y = convolve2d(X_pad, K, mode='valid')

    output_size = s * (xi - 1) + k  # 2*(2-1)+3 = 5
    return {
        'X_stretched': X_stretched.tolist(),
        'X_pad': X_pad.tolist(),
        'Y': Y.tolist(),
        'p': p,
        'output_size': output_size,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    K = np.array(p['K'])
    X = np.array(p['X'])
    X_stretched = np.array(c['X_stretched'])
    X_pad = np.array(c['X_pad'])
    Y = np.array(c['Y'])
    s = p['s']
    k = K.shape[0]

    lines = [
        f'## Question{num_str}: Transpose Convolution (Stride 2)',
        '',
        f'Compute the **transpose convolution** (stride $s=2$, no padding) of input'
        f' $X = {pmatrix(X)}$ with kernel $K = {pmatrix(K)}$.',
        '',
        'Output size: $o\' = s(i\'-1)+k = 2(2-1)+3 = {}$.'.format(c['output_size']),
        '',
        'Use the equivalent direct convolution approach:',
        f'1. Stretch $X$ by inserting $s-1=1$ zero(s) between elements.',
        f'2. Pad with $p\'=k-1={c["p"]}$ zeros on all sides.',
        f'3. Apply standard valid convolution with $K$.',
        '',
        f'Give the ${c["output_size"]} \\times {c["output_size"]}$ output $Y$.',
        '',
    ]

    if mode == 'blank':
        lines.append('**$Y =$ ** `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $Y = {pmatrix(Y)}$',
            '',
            '> **Hint:** Stretch the input (insert zeros between elements), pad with $k-1$ zeros,'
            ' then convolve with $K$ (valid).',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'**Step 1 - Stretch $X$ (insert $s-1=1$ zero between elements) -> $3 \\times 3$.**',
            '',
            f'$$X_{{stretched}} = {pmatrix(X_stretched)}$$',
            '',
            f'**Step 2 - Pad with $p\'=k-1={c["p"]}$ zeros -> $7 \\times 7$.**',
            '',
            f'$$X_{{pad}} = {pmatrix(X_pad)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output_size'])]
        lines += [
            '',
            f'**Step 3 - Valid convolution with $K$ -> ${c["output_size"]} \\times {c["output_size"]}$.**',
            '',
            f'$$Y = X_{{pad}} * K = {pmatrix(Y)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['computation'])]

    return '\n'.join(lines)
