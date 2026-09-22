"""Generator for Sobel gradient magnitude and direction questions.

Given a 3×3 image patch, compute Gx and Gy using the Sobel kernels,
then calculate magnitude M and direction theta.

Modes:
    'blank'  – question only
    'answer' – question + answers + hint
    'full'   – question + step-by-step Sobel computation
"""
import numpy as np
from ._utils import pmatrix
from ._exam_boxes import answer_box, inline_box

# Mark allocation per step — must sum to 10
MARKS = {
    'gx_gy': 4,      # Gx and Gy Sobel gradient computation
    'magnitude': 3,  # gradient magnitude
    'angle': 3,      # gradient direction angle
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


# Fixed Sobel kernels
_KX = np.array([[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]])

_KY = np.array([[1,  2,  1],
                [0,  0,  0],
                [-1, -2, -1]])

# Flipped versions used in convolution sum
_KX_FLIP = _KX[::-1, ::-1]   # [[-1,0,1],[-2,0,2],[-1,0,1]]
_KY_FLIP = _KY[::-1, ::-1]   # [[-1,-2,-1],[0,0,0],[1,2,1]]


def generate_params(seed=None, val_lo=-5, val_hi=5):
    rng = np.random.default_rng(seed)
    while True:
        I = rng.integers(val_lo, val_hi + 1, size=(3, 3)).astype(float)
        Gx = float(np.sum(_KX_FLIP * I))
        Gy = float(np.sum(_KY_FLIP * I))
        if abs(Gx) > 0.5 or abs(Gy) > 0.5:
            break
    return {'image': I.astype(int)}


def compute(params):
    I = params['image'].astype(float)
    Gx = float(np.sum(_KX_FLIP * I))
    Gy = float(np.sum(_KY_FLIP * I))
    M = float(np.sqrt(Gx ** 2 + Gy ** 2))
    theta_deg = float(np.degrees(np.arctan2(Gy, Gx)))
    return {
        'Gx': round(Gx, 2),
        'Gy': round(Gy, 2),
        'M': round(M, 2),
        'theta_deg': round(theta_deg, 2),
    }


def _sobel_terms(I, kernel_flip, var):
    """Return LaTeX string showing the non-zero kernel·image products."""
    terms = []
    for r in range(3):
        for c in range(3):
            k = int(kernel_flip[r, c])
            v = int(I[r, c])
            if k != 0:
                sign = '-' if k * v < 0 else '+'
                terms.append(f'{k:+d} \\cdot {v}')
    return ' + '.join(t.replace('+-', '-') for t in terms)


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    I = p['image']

    Kx_str = pmatrix(_KX)
    Ky_str = pmatrix(_KY)
    Kx_flip_str = pmatrix(_KX_FLIP)
    Ky_flip_str = pmatrix(_KY_FLIP)

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Gradient Magnitude and Direction'),
        '',
        'Given the $3\\times 3$ image patch',
        '',
        f'$$I = {pmatrix(I)}$$',
        '',
        'use the Sobel kernels',
        '',
        f'$$K_x = {Kx_str}, \\quad K_y = {Ky_str}$$',
        '',
        'to estimate the gradients $G_x$ and $G_y$ by convolution.'
        ' Then calculate:',
        '',
        r'1. The gradient **magnitude** $M = \sqrt{G_x^2 + G_y^2}$',
        r'2. The gradient **direction** $\theta = \tan^{-1}\\!\left(\dfrac{G_y}{G_x}\right)$ (in degrees)',
        '',
    ]

    # Build LaTeX term strings for Gx and Gy
    def _terms(kernel_flip):
        parts = []
        for r in range(3):
            for ci in range(3):
                k = int(kernel_flip[r, ci])
                v = int(I[r, ci])
                if k == 0:
                    continue
                sign = '+' if (not parts or k > 0) else '-' if k < 0 else '+'
                # Separate the leading sign from the abs value
                if parts:
                    parts.append(f'{k:+d} \\cdot {v}')
                else:
                    parts.append(f'{k} \\cdot {v}')
        return ' '.join(parts).replace('+ -', '- ').replace('- -', '+ ')

    gx_expr = _terms(_KX_FLIP)
    gy_expr = _terms(_KY_FLIP)

    if mode == 'blank':
        lines += [
            '**M =** `_______________`',
            '',
            r'**$\theta$ =** `_______________`',
        ]

    elif mode == 'exam':
        lines += [
            f'**Part (a):** $G_x =$ {inline_box()} $G_y =$ {inline_box()}',
            '',
            '**Part (b): M =**',
            '',
            answer_box(),
            '',
            r'**Part (c): $\theta$ =**',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Sobel gradients.**',
            '',
            r'Convolution flips the kernel 180° before computing the dot product with the patch.',
            '',
            f'$$G_x = {gx_expr} = {c["Gx"]}$$',
            '',
            f'$$G_y = {gy_expr} = {c["Gy"]}$$',
            '',
            _mark(MARKS['gx_gy']),
        ]
        lines += [
            '',
            '**Part (b) - Magnitude.**',
            '',
            f'$$M = \\sqrt{{G_x^2 + G_y^2}}'
            f'= \\sqrt{{{c["Gx"]}^2 + {c["Gy"]}^2}} = {c["M"]}$$',
            '',
            _mark(MARKS['magnitude']),
        ]
        lines += [
            '',
            '**Part (c) - Direction.**',
            '',
            f'$$\\theta = \\tan^{{-1}}\\\\!\\left(\\frac{{G_y}}{{G_x}}\\right)'
            f'= \\tan^{{-1}}\\\\!\\left(\\frac{{{c["Gy"]}}}{{{c["Gx"]}}}\\right)'
            f'= {c["theta_deg"]}°$$',
            '',
            _mark(MARKS['angle']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answers:** $M = {c["M"]}$, $\\theta = {c["theta_deg"]}°$',
            '',
            '> **Hint:** Convolve $I$ with each Sobel kernel (remember to flip the kernel'
            r' 180° before sliding). Since $I$ and $K$ are both $3\times 3$, the'
            r' convolution is a single dot product. Then $M = \sqrt{G_x^2+G_y^2}$'
            r' and $\theta = \arctan(G_y/G_x)$ converted to degrees.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Apply the Sobel kernels by convolution.**',
            '',
            r'Convolution flips the kernel 180° before computing the dot product with the patch.',
            '',
            f'Flipped $K_x$: $\\;{Kx_flip_str}$',
            '',
            f'Flipped $K_y$: $\\;{Ky_flip_str}$',
            '',
            r'Since $I$ and each kernel are both $3\times 3$, the convolution result is a single scalar.',
            '',
            f'$$G_x = {gx_expr} = {c["Gx"]}$$',
            '',
            f'$$G_y = {gy_expr} = {c["Gy"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['gx_gy'])]
        lines += [
            '',
            '**Step 2 – Compute magnitude and direction.**',
            '',
            f'$$M = \\sqrt{{G_x^2 + G_y^2}}'
            f'= \\sqrt{{{c["Gx"]}^2 + {c["Gy"]}^2}} = {c["M"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['magnitude'])]
        lines += [
            '',
            f'$$\\theta = \\tan^{{-1}}\\\\!\\left(\\frac{{G_y}}{{G_x}}\\right)'
            f'= \\tan^{{-1}}\\\\!\\left(\\frac{{{c["Gy"]}}}{{{c["Gx"]}}}\\right)'
            f'= {c["theta_deg"]}°$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['angle'])]

    return '\n'.join(lines)
