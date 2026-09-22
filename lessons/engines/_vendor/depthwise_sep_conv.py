"""Generator for Depthwise Separable Convolution questions (MobileNet style).

Compares computation cost of a standard convolution against a depthwise
separable convolution (depthwise + pointwise) on the same input/output spec.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'N_std': 3,     # standard convolution multiplications
    'N_dw': 2,      # depthwise stage multiplications
    'N_pw': 2,      # pointwise stage multiplications
    'reduction': 3, # reduction and ratio
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    h = int(rng.choice([8, 12, 14, 16, 28]))
    c_in = int(rng.choice([16, 32, 64]))
    c_out = int(rng.choice([32, 64, 128, 256]))
    k = int(rng.choice([3, 5]))
    return {'H': h, 'W': h, 'C_in': c_in, 'C_out': c_out, 'k': k}


def compute(params):
    H, W = params['H'], params['W']
    C_in, C_out, k = params['C_in'], params['C_out'], params['k']
    # Same padding → output spatial size = input spatial size
    N_std = k * k * C_in * C_out * H * W
    N_dw  = k * k * C_in * H * W          # one k×k filter per input channel
    N_pw  = C_in * C_out * H * W           # 1×1 pointwise
    N_dsc = N_dw + N_pw
    reduction = N_std - N_dsc
    ratio = round(N_dsc / N_std, 4)
    return {
        'N_std': N_std, 'N_dw': N_dw, 'N_pw': N_pw,
        'N_dsc': N_dsc, 'reduction': reduction, 'ratio': ratio,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    H, W, C_in, C_out, k = p['H'], p['W'], p['C_in'], p['C_out'], p['k']

    lines = [
        f'## Question{num_str}: Depthwise Separable Convolution',
        '',
        f'A convolutional layer processes a ${H} \\times {W} \\times {C_in}$ feature map'
        f' and produces ${H} \\times {W} \\times {C_out}$ output (same padding, stride 1).',
        '',
        f'**Standard convolution** uses ${C_out}$ filters of size ${k} \\times {k} \\times {C_in}$.',
        '',
        '**Depthwise separable convolution** replaces this with two stages:',
        f'- **Depthwise:** one ${k} \\times {k}$ filter per input channel ($C_{{in}} = {C_in}$ filters total).',
        f'- **Pointwise:** $1 \\times 1$ convolutions to mix channels to $C_{{out}} = {C_out}$.',
        '',
        'For each stage, count only **multiplications** (one per weight per output location).',
        '',
        '**Part (a):** Multiplications for the standard convolution $N_\\text{std}$.',
        '',
        '**Part (b):** Multiplications for the depthwise stage $N_\\text{dw}$ and'
        ' pointwise stage $N_\\text{pw}$; total $N_\\text{dsc} = N_\\text{dw} + N_\\text{pw}$.',
        '',
        '**Part (c):** Reduction in multiplications $N_\\text{std} - N_\\text{dsc}$'
        ' and the ratio $N_\\text{dsc} / N_\\text{std}$.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** $N_\\text{std} =$ `___`',
            '',
            '**Part (b):** $N_\\text{dw} =$ `___`, $N_\\text{pw} =$ `___`,'
            ' $N_\\text{dsc} =$ `___`',
            '',
            '**Part (c):** Reduction $=$ `___`, Ratio $=$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $N_\\text{{std}} = {c["N_std"]:,}$',
            '',
            f'**Part (b):** $N_\\text{{dw}} = {c["N_dw"]:,}$,'
            f' $N_\\text{{pw}} = {c["N_pw"]:,}$,'
            f' $N_\\text{{dsc}} = {c["N_dsc"]:,}$',
            '',
            f'**Part (c):** Reduction $= {c["reduction"]:,}$,'
            f' Ratio $= {c["ratio"]}$',
            '',
            f'> **Hint:** $N_\\text{{std}} = k^2 \\cdot C_{{in}} \\cdot C_{{out}} \\cdot H \\cdot W$; '
            f'$N_\\text{{dw}} = k^2 \\cdot C_{{in}} \\cdot H \\cdot W$; '
            f'$N_\\text{{pw}} = C_{{in}} \\cdot C_{{out}} \\cdot H \\cdot W$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) — Standard convolution.**',
            '',
            f'$$N_\\text{{std}} = k^2 \\cdot C_{{in}} \\cdot C_{{out}} \\cdot H \\cdot W'
            f' = {k}^2 \\times {C_in} \\times {C_out} \\times {H} \\times {W}'
            f' = {c["N_std"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_std'])]
        lines += [
            '',
            '**Part (b) — Depthwise separable convolution.**',
            '',
            f'Depthwise ($k \\times k$ per channel, $C_{{in}}$ filters):',
            f'$$N_\\text{{dw}} = k^2 \\cdot C_{{in}} \\cdot H \\cdot W'
            f' = {k}^2 \\times {C_in} \\times {H} \\times {W} = {c["N_dw"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_dw'])]
        lines += [
            '',
            f'Pointwise ($1 \\times 1$, mixes {C_in} channels to {C_out}):',
            f'$$N_\\text{{pw}} = C_{{in}} \\cdot C_{{out}} \\cdot H \\cdot W'
            f' = {C_in} \\times {C_out} \\times {H} \\times {W} = {c["N_pw"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_pw'])]
        lines += [
            '',
            f'$$N_\\text{{dsc}} = {c["N_dw"]:,} + {c["N_pw"]:,} = {c["N_dsc"]:,}$$',
            '',
            '**Part (c) — Reduction and ratio.**',
            '',
            f'$$N_\\text{{std}} - N_\\text{{dsc}} = {c["N_std"]:,} - {c["N_dsc"]:,}'
            f' = {c["reduction"]:,}$$',
            '',
            f'$$\\frac{{N_\\text{{dsc}}}}{{N_\\text{{std}}}} = '
            f'\\frac{{{c["N_dsc"]:,}}}{{{c["N_std"]:,}}} = {c["ratio"]}$$',
            '',
            f'The depthwise separable convolution uses only {100*c["ratio"]:.1f}% of the'
            f' multiplications of the standard convolution.',
        ]
        if marks:
            lines += ['', _mark(MARKS['reduction'])]

    return '\n'.join(lines)
