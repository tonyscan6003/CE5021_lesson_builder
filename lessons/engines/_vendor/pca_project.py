"""PCA projection and reconstruction question generator (Eigenfaces context).

Given mean face μ, two eigenfaces e1 and e2, and a query face x (all in ℝ⁴):
  1. x_c = x − μ                          (mean-centred face)
  2. z1 = e1 · x_c,  z2 = e2 · x_c       (PCA projection coefficients)
  3. x̂ = μ + z1·e1 + z2·e2               (reconstructed face)

Three orthonormal eigenvector pairs are available (selected by seed % 3):
  Pair A: e1 = [0.5, 0.5, 0.5, 0.5],   e2 = [0.5, -0.5,  0.5, -0.5]
  Pair B: e1 = [0.5, 0.5, 0.5, 0.5],   e2 = [0.5,  0.5, -0.5, -0.5]
  Pair C: e1 = [0.5, 0.5, -0.5, -0.5], e2 = [0.5, -0.5, -0.5,  0.5]

All pairs are orthonormal (unit norm, mutually orthogonal dot product = 0).

Modes:
    'blank'      – question + blank placeholders
    'answer'     – question + final answers + hint
    'full'       – question + step-by-step derivation
    'full_marks' – question + derivation with mark annotations
"""
import numpy as np
from ._exam_boxes import answer_box, inline_box

MARKS = {
    'centering':      2,
    'projection':     4,
    'reconstruction': 4,
}

EXAM_COMPATIBLE = True

# (e1, e2) pairs – each has unit norm and e1·e2 = 0 within a pair
_EIGENPAIRS = [
    # Pair A
    ([0.5,  0.5,  0.5,  0.5], [0.5, -0.5,  0.5, -0.5]),
    # Pair B
    ([0.5,  0.5,  0.5,  0.5], [0.5,  0.5, -0.5, -0.5]),
    # Pair C
    ([0.5,  0.5, -0.5, -0.5], [0.5, -0.5, -0.5,  0.5]),
]


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def _fmt_num(v):
    """Format a number cleanly: integer if whole, else up to 4 dp (no trailing zeros)."""
    if isinstance(v, int):
        return str(v)
    f = float(v)
    if f == int(f):
        return str(int(f))
    return f'{f:.4f}'.rstrip('0').rstrip('.')


def _fmt_vec(v, label=''):
    """Format a list as a LaTeX column vector (inline bmatrix)."""
    inner = ' \\\\ '.join(_fmt_num(vi) for vi in v)
    vec = f'\\begin{{bmatrix}} {inner} \\end{{bmatrix}}'
    if label:
        return f'$${label} = {vec}$$'
    return f'$${vec}$$'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    pair_idx = int(seed) % 3 if seed is not None else int(rng.integers(0, 3))
    e1, e2 = _EIGENPAIRS[pair_idx]
    mu = [int(v) for v in rng.integers(2, 16, size=4)]
    x  = [int(v) for v in rng.integers(5, 26, size=4)]
    return {
        'mu':       mu,
        'e1':       e1,
        'e2':       e2,
        'x':        x,
        'pair_idx': pair_idx,
    }


def compute(params):
    mu = params['mu']
    e1 = params['e1']
    e2 = params['e2']
    x  = params['x']
    x_c   = [x[i] - mu[i] for i in range(4)]
    z1    = round(sum(e1[i] * x_c[i] for i in range(4)), 4)
    z2    = round(sum(e2[i] * x_c[i] for i in range(4)), 4)
    x_hat = [round(mu[i] + z1 * e1[i] + z2 * e2[i], 4) for i in range(4)]
    return {'x_c': x_c, 'z1': z1, 'z2': z2, 'x_hat': x_hat}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    mu    = p['mu']
    e1    = p['e1']
    e2    = p['e2']
    x     = p['x']
    x_c   = c['x_c']
    z1    = c['z1']
    z2    = c['z2']
    x_hat = c['x_hat']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: PCA Projection and Reconstruction (Eigenfaces)'),
        '',
        'In the Eigenfaces method, a face image $\\mathbf{x}$ is represented as a'
        ' linear combination of eigenfaces.'
        ' Given the mean face $\\boldsymbol{\\mu}$,'
        ' eigenfaces $\\mathbf{e}_1, \\mathbf{e}_2$,'
        ' and query face $\\mathbf{x}$:',
        '',
        _fmt_vec(mu, label='\\boldsymbol{\\mu}'),
        '',
        _fmt_vec(e1, label='\\mathbf{e}_1'),
        '',
        _fmt_vec(e2, label='\\mathbf{e}_2'),
        '',
        _fmt_vec(x, label='\\mathbf{x}'),
        '',
        'Compute:',
        '1. The mean-centred face'
        ' $\\mathbf{x}_c = \\mathbf{x} - \\boldsymbol{\\mu}$',
        '2. The PCA projection coefficients'
        ' $z_1 = \\mathbf{e}_1 \\cdot \\mathbf{x}_c$'
        ' and $z_2 = \\mathbf{e}_2 \\cdot \\mathbf{x}_c$',
        '3. The reconstructed face'
        ' $\\hat{\\mathbf{x}} = \\boldsymbol{\\mu}'
        ' + z_1 \\mathbf{e}_1 + z_2 \\mathbf{e}_2$',
        '',
    ]

    # ------------------------------------------------------------------ blank
    if mode == 'blank':
        lines += [
            '$\\mathbf{x}_c =$ `[___, ___, ___, ___]`',
            '',
            '$z_1 =$ `___`',
            '',
            '$z_2 =$ `___`',
            '',
            '$\\hat{\\mathbf{x}} =$ `[___, ___, ___, ___]`',
        ]

    # ------------------------------------------------------------------- exam
    elif mode == 'exam':
        lines += [
            '**Part (a):** $\\mathbf{x}_c =$',
            '',
            answer_box(),
            '',
            f'**Part (b):** $z_1 =$ {inline_box()} $z_2 =$ {inline_box()}',
            '',
            '**Part (c):** $\\hat{\\mathbf{x}} =$',
            '',
            answer_box(),
        ]

    # ------------------------------------------------------------- exam_marks
    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**Part (a) - Mean-centred face'
            ' $\\mathbf{x}_c = \\mathbf{x} - \\boldsymbol{\\mu}$:**',
            '',
        ]
        sub_inner = ' \\\\ '.join(f'{x[i]} - {mu[i]}' for i in range(4))
        res_inner = ' \\\\ '.join(_fmt_num(x_c[i]) for i in range(4))
        lines.append(
            f'$$\\mathbf{{x}}_c'
            f' = \\begin{{bmatrix}} {sub_inner} \\end{{bmatrix}}'
            f' = \\begin{{bmatrix}} {res_inner} \\end{{bmatrix}}$$'
        )
        lines += ['', _mark(MARKS['centering'])]

        lines += [
            '',
            '**Part (b) - PCA projection coefficients:**',
            '',
        ]
        z1_terms = ' + '.join(
            f'({_fmt_num(e1[i])})({_fmt_num(x_c[i])})' for i in range(4)
        )
        lines.append(
            f'$$z_1 = \\mathbf{{e}}_1 \\cdot \\mathbf{{x}}_c'
            f' = {z1_terms}'
            f' = {_fmt_num(z1)}$$'
        )
        lines.append('')
        z2_terms = ' + '.join(
            f'({_fmt_num(e2[i])})({_fmt_num(x_c[i])})' for i in range(4)
        )
        lines.append(
            f'$$z_2 = \\mathbf{{e}}_2 \\cdot \\mathbf{{x}}_c'
            f' = {z2_terms}'
            f' = {_fmt_num(z2)}$$'
        )
        lines += ['', _mark(MARKS['projection'])]

        lines += [
            '',
            '**Part (c) - Reconstructed face'
            ' $\\hat{\\mathbf{x}} = \\boldsymbol{\\mu}'
            ' + z_1 \\mathbf{e}_1 + z_2 \\mathbf{e}_2$:**',
            '',
        ]
        recon_inner = ' \\\\ '.join(
            f'{_fmt_num(mu[i])}'
            f' + ({_fmt_num(z1)})({_fmt_num(e1[i])})'
            f' + ({_fmt_num(z2)})({_fmt_num(e2[i])})'
            f' = {_fmt_num(x_hat[i])}'
            for i in range(4)
        )
        lines.append(
            f'$$\\hat{{\\mathbf{{x}}}}'
            f' = \\begin{{bmatrix}} {recon_inner} \\end{{bmatrix}}$$'
        )
        lines += ['', _fmt_vec(x_hat, label='\\hat{\\mathbf{x}}')]
        lines += ['', _mark(MARKS['reconstruction'])]

    # ----------------------------------------------------------------- answer
    elif mode == 'answer':
        lines += [
            _fmt_vec(x_c, label='\\mathbf{x}_c'),
            '',
            f'$z_1 = {_fmt_num(z1)}$',
            '',
            f'$z_2 = {_fmt_num(z2)}$',
            '',
            _fmt_vec(x_hat, label='\\hat{\\mathbf{x}}'),
            '',
            '> **Hint:** Centre the data:'
            ' $\\mathbf{x}_c = \\mathbf{x} - \\boldsymbol{\\mu}$.'
            ' Project: $z_i = \\mathbf{e}_i \\cdot \\mathbf{x}_c$ (dot product).'
            ' Reconstruct:'
            ' $\\hat{\\mathbf{x}} = \\boldsymbol{\\mu}'
            ' + z_1 \\mathbf{e}_1 + z_2 \\mathbf{e}_2$.',
        ]

    # ----------------------------------------------------- full / full_marks
    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        # --- Step 1: centering -------------------------------------------
        lines += [
            '**Step 1 – Mean-centred face'
            ' $\\mathbf{x}_c = \\mathbf{x} - \\boldsymbol{\\mu}$:**',
            '',
        ]
        sub_inner = ' \\\\ '.join(f'{x[i]} - {mu[i]}' for i in range(4))
        res_inner = ' \\\\ '.join(_fmt_num(x_c[i]) for i in range(4))
        lines.append(
            f'$$\\mathbf{{x}}_c'
            f' = \\begin{{bmatrix}} {sub_inner} \\end{{bmatrix}}'
            f' = \\begin{{bmatrix}} {res_inner} \\end{{bmatrix}}$$'
        )
        if marks:
            lines += ['', _mark(MARKS['centering'])]

        # --- Step 2: projection ------------------------------------------
        lines += [
            '',
            '**Step 2 – PCA projection coefficients:**',
            '',
        ]

        z1_terms = ' + '.join(
            f'({_fmt_num(e1[i])})({_fmt_num(x_c[i])})' for i in range(4)
        )
        lines.append(
            f'$$z_1 = \\mathbf{{e}}_1 \\cdot \\mathbf{{x}}_c'
            f' = {z1_terms}'
            f' = {_fmt_num(z1)}$$'
        )
        lines.append('')

        z2_terms = ' + '.join(
            f'({_fmt_num(e2[i])})({_fmt_num(x_c[i])})' for i in range(4)
        )
        lines.append(
            f'$$z_2 = \\mathbf{{e}}_2 \\cdot \\mathbf{{x}}_c'
            f' = {z2_terms}'
            f' = {_fmt_num(z2)}$$'
        )
        if marks:
            lines += ['', _mark(MARKS['projection'])]

        # --- Step 3: reconstruction --------------------------------------
        lines += [
            '',
            '**Step 3 – Reconstructed face'
            ' $\\hat{\\mathbf{x}} = \\boldsymbol{\\mu}'
            ' + z_1 \\mathbf{e}_1 + z_2 \\mathbf{e}_2$:**',
            '',
        ]
        recon_inner = ' \\\\ '.join(
            f'{_fmt_num(mu[i])}'
            f' + ({_fmt_num(z1)})({_fmt_num(e1[i])})'
            f' + ({_fmt_num(z2)})({_fmt_num(e2[i])})'
            f' = {_fmt_num(x_hat[i])}'
            for i in range(4)
        )
        lines.append(
            f'$$\\hat{{\\mathbf{{x}}}}'
            f' = \\begin{{bmatrix}} {recon_inner} \\end{{bmatrix}}$$'
        )
        lines += ['', _fmt_vec(x_hat, label='\\hat{\\mathbf{x}}')]
        if marks:
            lines += ['', _mark(MARKS['reconstruction'])]

    return '\n'.join(lines)
