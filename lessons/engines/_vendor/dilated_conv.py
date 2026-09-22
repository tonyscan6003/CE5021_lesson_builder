"""Generator for Dilated Convolution questions.

7x7 image, 3x3 kernel, dilation d=2, stride s=1, no padding.
Effective kernel size: k_hat = k + (k-1)*(d-1) = 5.
Output size: o = floor((7+0-5)/1)+1 = 3. So 3x3 output.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution showing expanded kernel
"""
import numpy as np
from scipy.signal import convolve2d
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'effective_kernel_size': 2,  # k_hat formula
    'expanded_kernel': 3,        # K_exp matrix
    'output_size': 3,            # output size derivation
    'computation': 2,            # final convolution output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    I = rng.integers(0, 6, size=(7, 7)).tolist()
    while True:
        K = rng.integers(-2, 3, size=(3, 3)).tolist()
        if any(K[i][j] != 0 for i in range(3) for j in range(3)):
            break
    return {'I': I, 'K': K, 'd': 2, 's': 1, 'p': 0}


def compute(params):
    I = np.array(params['I'])
    K = np.array(params['K'])
    d = params['d']
    k = K.shape[0]  # 3

    # Effective kernel size
    k_hat = k + (k - 1) * (d - 1)  # 5

    # Expand K by inserting d-1=1 zeros between elements -> 5x5
    K_exp = np.zeros((k_hat, k_hat), dtype=int)
    for i in range(k):
        for j in range(k):
            K_exp[i * d, j * d] = K[i, j]

    # Output size
    o = (I.shape[0] - k_hat) // params['s'] + 1  # 3

    # Compute output using expanded kernel (valid conv)
    O = convolve2d(I, K_exp, mode='valid')

    return {
        'K_exp': K_exp.tolist(),
        'O': O.tolist(),
        'k_hat': k_hat,
        'o': o,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    I = np.array(p['I'])
    K = np.array(p['K'])
    K_exp = np.array(c['K_exp'])
    O = np.array(c['O'])
    d = p['d']
    k = K.shape[0]

    lines = [
        f'## Question{num_str}: Dilated Convolution',
        '',
        f'Apply a **dilated convolution** with dilation $d={d}$, stride $s=1$, no padding'
        f' to the following $7 \\times 7$ image:',
        '',
        f'$$I = {pmatrix(I)}$$',
        '',
        f'Using the $3 \\times 3$ kernel:',
        '',
        f'$$K = {pmatrix(K)}$$',
        '',
        'Give the output matrix $O$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$\\hat{k} =$ `___`,  Output size $=$ `___`',
            '',
            '**Output** $O =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Effective kernel size:** $\\hat{{k}} = {c["k_hat"]}$',
            '',
            f'**Output size:** ${c["o"]} \\times {c["o"]}$',
            '',
            f'**Output:** $O = {pmatrix(O)}$',
            '',
            '> **Hint:** Effective kernel size: $\\hat{{k}} = k + (k-1)(d-1)$.'
            ' Expand the kernel by inserting $d-1$ zeros between elements'
            ' to get the expanded kernel, then apply standard valid convolution.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'**Step 1 - Effective kernel size.**',
            '',
            f'$$\\hat{{k}} = k + (k-1)(d-1) = {k} + ({k}-1)({d}-1) = {k} + {k-1} = {c["k_hat"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['effective_kernel_size'])]
        lines += [
            '',
            f'**Step 2 - Expanded kernel.**',
            '',
            f'Expand $K$ by inserting $d-1={d-1}$ zero between each element'
            f' $\\to {c["k_hat"]} \\times {c["k_hat"]}$:',
            '',
            f'$$K_{{exp}} = {pmatrix(K_exp)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['expanded_kernel'])]
        lines += [
            '',
            f'**Step 3 - Determine output size.**',
            '',
            f'$$o = \\lfloor(7 - {c["k_hat"]})/1\\rfloor + 1 = {c["o"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output_size'])]
        lines += [
            '',
            f'**Step 4 - Apply valid convolution.**',
            '',
            f'Convolve the $7 \\times 7$ image with the ${c["k_hat"]} \\times {c["k_hat"]}$ expanded kernel:',
            '',
            f'$$O = I * K_{{exp}} = {pmatrix(O)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['computation'])]

    return '\n'.join(lines)
