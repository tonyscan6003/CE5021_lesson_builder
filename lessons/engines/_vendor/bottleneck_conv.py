"""Generator for Bottleneck Convolution questions.

Compares multiplications for a standard conv vs a bottleneck conv.
Shows reduction in computation.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution

Pass marks=True to render() to show mark allocation per step (full mode only).
Total marks sum to 10.
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'N_M0': 3,
    'N_M1': 2,
    'N_M2': 2,
    'reduction': 3,
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    m_choices = [24, 28, 32, 36, 40]
    C_choices = [64, 96, 128, 136, 192]
    N_choices = [32, 54, 64, 96]
    P_choices = [8, 16, 18, 24]
    k_choices = [3, 5, 7]

    m = int(rng.choice(m_choices))
    C = int(rng.choice(C_choices))
    N = int(rng.choice(N_choices))
    P = int(rng.choice(P_choices))
    k = int(rng.choice(k_choices))
    p = k // 2

    return {'m': m, 'C': C, 'N': N, 'P': P, 'k': k, 'p': p}


def compute(params):
    m, C, N, P, k = params['m'], params['C'], params['N'], params['P'], params['k']
    N_M0 = k * k * C * N * m * m
    N_M1 = 1 * 1 * C * P * m * m
    N_M2 = k * k * P * N * m * m
    reduction = N_M0 - N_M1 - N_M2
    return {'N_M0': N_M0, 'N_M1': N_M1, 'N_M2': N_M2, 'reduction': reduction}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    m, C, N, P, k, pad = p['m'], p['C'], p['N'], p['P'], p['k'], p['p']

    lines = [
        f'## Question{num_str}: Bottleneck Convolution',
        '',
        'A bottleneck block is used to reduce computation. The input has spatial size'
        f' ${m} \\times {m}$, $C = {C}$ channels. The block produces $N = {N}$ output channels'
        f' using a bottleneck of $P = {P}$ channels, with kernel size $k = {k}$'
        f' (same padding $p = \\lfloor k/2 \\rfloor = {pad}$, stride $s=1$).',
        '',
        'Compute the following:',
        '',
        f'- $N_{{M0}}$: multiplications in the **original** $k \\times k$ convolution ($C \\to N$)',
        f'- $N_{{M1}}$: multiplications in the **bottleneck 1st stage** ($1 \\times 1$ conv, $C \\to P$)',
        f'- $N_{{M2}}$: multiplications in the **bottleneck 2nd stage** ($k \\times k$ conv, $P \\to N$)',
        f'- **Reduction** $= N_{{M0}} - N_{{M1}} - N_{{M2}}$',
        '',
    ]

    if mode == 'blank':
        lines += [
            '$N_{M0} =$ `___`, $N_{M1} =$ `___`, $N_{M2} =$ `___`, Reduction $=$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'$N_{{M0}} = {c["N_M0"]}$',
            '',
            f'$N_{{M1}} = {c["N_M1"]}$',
            '',
            f'$N_{{M2}} = {c["N_M2"]}$',
            '',
            f'**Reduction** $= {c["reduction"]}$',
            '',
            '> **Hint:** $N_{{M0}} = k^2 \\cdot C \\cdot N \\cdot m^2$;'
            ' $N_{{M1}} = 1 \\cdot C \\cdot P \\cdot m^2$;'
            ' $N_{{M2}} = k^2 \\cdot P \\cdot N \\cdot m^2$.'
            ' Reduction $= N_{{M0}} - N_{{M1}} - N_{{M2}}$.',
        ]

    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Original convolution:**',
            '',
            f'$$N_{{M0}} = k^2 \\cdot C \\cdot N \\cdot m^2'
            f' = {k}^2 \\cdot {C} \\cdot {N} \\cdot {m}^2'
            f' = {k*k} \\cdot {C} \\cdot {N} \\cdot {m*m} = {c["N_M0"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_M0'])]
        lines += [
            '',
            '**Bottleneck stage 1 ($1 \\times 1$ conv, $C \\to P$):**',
            '',
            f'$$N_{{M1}} = 1 \\cdot C \\cdot P \\cdot m^2'
            f' = 1 \\cdot {C} \\cdot {P} \\cdot {m}^2 = {c["N_M1"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_M1'])]
        lines += [
            '',
            '**Bottleneck stage 2 ($k \\times k$ conv, $P \\to N$):**',
            '',
            f'$$N_{{M2}} = k^2 \\cdot P \\cdot N \\cdot m^2'
            f' = {k}^2 \\cdot {P} \\cdot {N} \\cdot {m}^2'
            f' = {k*k} \\cdot {P} \\cdot {N} \\cdot {m*m} = {c["N_M2"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['N_M2'])]
        lines += [
            '',
            '**Reduction in multiplications:**',
            '',
            f'$$N_{{M0}} - N_{{M1}} - N_{{M2}} = {c["N_M0"]} - {c["N_M1"]} - {c["N_M2"]} = {c["reduction"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['reduction'])]

    return '\n'.join(lines)
