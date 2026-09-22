"""Generator for Harris corner response questions.

Given a second moment matrix M = [[A, C], [C, B]], compute the corner
response R = Det(M) - alpha * Trace(M)^2 and classify the point.

Modes:
    'blank'  – question only
    'answer' – question + R value + classification + hint
    'full'   – question + step-by-step derivation + classification
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'm_matrix': 4,      # determinant and trace of M
    'r_values': 4,      # corner response R
    'classification': 2, # point classification
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, alpha=0.04, val_lo=0.5, val_hi=9.9):
    rng = np.random.default_rng(seed)
    while True:
        A = round(float(rng.uniform(val_lo, val_hi)), 1)
        B = round(float(rng.uniform(val_lo, val_hi)), 1)
        C = round(float(rng.uniform(-4.0, 4.0)), 1)
        # Ensure det > 0 and that the classification is unambiguous
        det = A * B - C ** 2
        trace = A + B
        R = det - alpha * trace ** 2
        if abs(R) > 0.5:  # avoid near-zero (ambiguous flat/edge boundary)
            break
    return {'A': A, 'B': B, 'C': C, 'alpha': alpha}


def compute(params):
    A, B, C, alpha = params['A'], params['B'], params['C'], params['alpha']
    det = round(A * B - C ** 2, 4)
    trace = round(A + B, 4)
    R = round(det - alpha * trace ** 2, 4)
    if R > 0.5:
        classification = 'corner'
    elif R < -0.5:
        classification = 'edge'
    else:
        classification = 'flat region'
    return {'det': det, 'trace': trace, 'R': R, 'classification': classification}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    A, B, C, alpha = p['A'], p['B'], p['C'], p['alpha']

    M_str = (r'\begin{pmatrix}'
             f'{A} & {C} \\\\ {C} & {B}'
             r'\end{pmatrix}')

    lines = [
        f'## Question{num_str}: Harris Corner Response',
        '',
        f'In the Harris corner detection algorithm the second moment matrix is',
        '',
        f'$$M = {M_str}$$',
        '',
        f'Calculate the corner response $R = \\det(M) - \\alpha\\,\\bigl(\\mathrm{{tr}}(M)\\bigr)^2$'
        f' using $\\alpha = {alpha}$, and classify the point as a **corner**, **edge**, or **flat region**.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**R =** `_______________`',
            '',
            '**Classification:** `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $R = {c["R"]}$ → **{c["classification"]}**',
            '',
            '> **Hint:** Compute $\\det(M) = AB - C^2$ and $\\mathrm{tr}(M) = A + B$,'
            ' then evaluate $R = \\det(M) - \\alpha\\,(\\mathrm{tr}(M))^2$.'
            ' $R \\gg 0$ → corner; $R \\ll 0$ → edge; $R \\approx 0$ → flat.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Compute determinant and trace.**',
            '',
            f'$$\\det(M) = AB - C^2 = {A} \\cdot {B} - ({C})^2 = {c["det"]}$$',
            '',
            f'$$\\mathrm{{tr}}(M) = A + B = {A} + {B} = {c["trace"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['m_matrix'])]
        lines += [
            '',
            '**Step 2 – Evaluate the corner response.**',
            '',
            f'$$R = \\det(M) - \\alpha\\,(\\mathrm{{tr}}(M))^2'
            f'= {c["det"]} - {alpha} \\cdot ({c["trace"]})^2 = {c["R"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['r_values'])]
        lines += [
            '',
            '**Step 3 – Classify.**',
            '',
            '| $R$ | Interpretation |',
            '|-----|----------------|',
            '| $R \\gg 0$ | Corner |',
            '| $R \\ll 0$ | Edge |',
            '| $R \\approx 0$ | Flat region |',
            '',
            f'Since $R = {c["R"]}$, this point is a **{c["classification"]}**.',
        ]
        if marks:
            lines += ['', _mark(MARKS['classification'])]

    return '\n'.join(lines)
