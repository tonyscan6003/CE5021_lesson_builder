"""Generator for Matrix Method Convolution questions (two parts).

Part (a): Sparse matrix S (4x9) for valid convolution of 3x3 image with 2x2 kernel.
Part (b): Transpose convolution using S^T.

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts
"""
import numpy as np
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'kernel_flip': 2,  # flipped kernel / sparse matrix S
    'output': 8,       # convolution output + transpose conv
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    I = rng.integers(0, 6, size=(3, 3)).tolist()
    # kernel with at least one nonzero
    while True:
        K = rng.integers(-3, 4, size=(2, 2)).tolist()
        if any(K[i][j] != 0 for i in range(2) for j in range(2)):
            break
    X = rng.integers(1, 6, size=(2, 2)).tolist()
    return {'I': I, 'K': K, 'X': X}


def compute(params):
    I = np.array(params['I'])
    K = np.array(params['K'])
    X = np.array(params['X'])

    # Flip kernel 180 degrees for convolution
    K_flip = np.rot90(K, 2)

    # Build sparse matrix S (4x9) for valid conv of 3x3 image with 2x2 kernel
    # Output is 2x2 (4 positions). Image is unrolled row-major into length 9.
    S = np.zeros((4, 9), dtype=int)
    out_pos = 0
    for oi in range(2):
        for oj in range(2):
            row = np.zeros(9, dtype=int)
            for ki in range(2):
                for kj in range(2):
                    img_i = oi + ki
                    img_j = oj + kj
                    idx = img_i * 3 + img_j
                    row[idx] = int(K_flip[ki, kj])
            S[out_pos] = row
            out_pos += 1

    # Verify: S @ I_ur == O_ur
    I_ur = I.flatten()
    O_ur = S @ I_ur
    O = O_ur.reshape(2, 2)

    # Transpose convolution: Y = S^T @ X_ur
    X_ur = X.flatten()
    Y_ur = S.T @ X_ur
    Y = Y_ur.reshape(3, 3)

    return {
        'K_flip': K_flip.tolist(),
        'S': S.tolist(),
        'O': O.tolist(),
        'Y': Y.tolist(),
        'O_ur': O_ur.tolist(),
        'Y_ur': Y_ur.tolist(),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    I = np.array(p['I'])
    K = np.array(p['K'])
    X = np.array(p['X'])
    S = np.array(c['S'])
    O = np.array(c['O'])
    Y = np.array(c['Y'])
    K_flip = np.array(c['K_flip'])

    def _mat_str(arr, fmt_fn=None):
        arr = np.asarray(arr)
        if fmt_fn is None:
            fmt_fn = str
        if arr.ndim == 1:
            inner = ' & '.join(fmt_fn(v) for v in arr)
        else:
            rows = [' & '.join(fmt_fn(v) for v in row) for row in arr]
            inner = r' \\ '.join(rows)
        return r'\begin{pmatrix}' + inner + r'\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Matrix Method Convolution',
        '',
        f'Given image $I = {pmatrix(I)}$ and kernel $K = {pmatrix(K)}$.',
        '',
        '**Part (a):** Construct the sparse matrix $S$ ($4 \\times 9$) that implements'
        ' valid convolution $O = S \\cdot \\mathbf{I}_{ur}$ where $\\mathbf{I}_{ur}$'
        ' is the row-major unrolled image.',
        'Verify that $S \\cdot \\mathbf{I}_{ur}$ equals the unrolled $2 \\times 2$ output.',
        '',
        '**Part (b):** Transpose convolution: given input $X = {mat}$,'.format(mat=pmatrix(X)),
        'compute $\\mathbf{Y}_{ur} = S^T \\cdot \\mathbf{X}_{ur}$ and reshape to $3 \\times 3$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** $S =$ `_______________`',
            '',
            '**Part (b):** $Y =$ `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $S = {_mat_str(S)}$',
            '',
            f'Output: $O = {pmatrix(O)}$',
            '',
            f'**Part (b):** $Y = {pmatrix(Y)}$',
            '',
            '> **Hint:** Flip $K$ by 180 degrees. Each row of $S$ places the flipped kernel'
            ' elements at the correct image indices for that output position.'
            ' For transpose conv, multiply $S^T$ by the unrolled input and reshape.',
        ]

    else:
        I_ur = I.flatten().tolist()
        X_ur = X.flatten().tolist()
        Y_ur = np.array(c['Y_ur'])
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 - Flip kernel 180 degrees (for convolution).**',
            '',
            f'$$K_{{flip}} = {pmatrix(K_flip)}$$',
            '',
            '**Part (a) - Build sparse matrix S.**',
            '',
            'Each row of $S$ corresponds to one output pixel. Place $K_{{flip}}$ elements'
            ' at the correct indices in the $1 \\times 9$ row vector:',
            '',
            f'$$S = {_mat_str(S)}$$',
            '',
            f'Unrolled image: $\\mathbf{{I}}_{{ur}} = {_mat_str(np.array(I_ur))}^T$',
            '',
            f'Output: $\\mathbf{{O}}_{{ur}} = S \\cdot \\mathbf{{I}}_{{ur}} = {_mat_str(np.array(c["O_ur"]))}^T$',
            '',
            f'Reshaped: $O = {pmatrix(O)}$',
        ]
        if marks:
            lines += ['', _mark(MARKS['kernel_flip'])]
        lines += [
            '',
            '**Part (b) - Transpose convolution.**',
            '',
            f'Unrolled input: $\\mathbf{{X}}_{{ur}} = {_mat_str(np.array(X_ur))}^T$',
            '',
            f'$$\\mathbf{{Y}}_{{ur}} = S^T \\cdot \\mathbf{{X}}_{{ur}} = {_mat_str(Y_ur)}^T$$',
            '',
            f'Reshaped to $3 \\times 3$: $Y = {pmatrix(Y)}$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output'])]

    return '\n'.join(lines)
