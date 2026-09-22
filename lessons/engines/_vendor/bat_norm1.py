"""Generator for Batch Normalisation questions (apply to batch).

Random batch x of 5 floats, scale gamma, shift beta.
Step 1: mu_B, sigma2_B (biased variance).
Step 2: x_hat_i = (x_i - mu_B) / sqrt(sigma2_B).
Step 3: y_i = gamma * x_hat_i + beta.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution

Pass marks=True to render() to show mark allocation per step (full mode only).
Total marks sum to 10.
"""
import math
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'mean': 2,        # batch mean
    'variance': 3,    # batch variance
    'normalized': 3,  # normalised x_hat
    'output': 2,      # scaled/shifted output y
}


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    x = [round(float(rng.uniform(0, 5)), 1) for _ in range(5)]
    gamma = round(float(rng.uniform(0.1, 1.0)), 1)
    beta = round(float(rng.uniform(0.1, 0.5)), 1)
    return {'x': x, 'gamma': gamma, 'beta': beta}


def compute(params):
    x = params['x']
    gamma, beta = params['gamma'], params['beta']
    m = len(x)
    mu = round(sum(x) / m, 2)
    sigma2 = round(sum((xi - mu) ** 2 for xi in x) / m, 2)
    x_hat = [round((xi - mu) / math.sqrt(sigma2) if sigma2 > 0 else 0.0, 2) for xi in x]
    y = [round(gamma * xh + beta, 1) for xh in x_hat]
    return {'mu': mu, 'sigma2': sigma2, 'x_hat': x_hat, 'y': y}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    x = p['x']

    x_vec = '\\begin{pmatrix}' + ' & '.join(str(xi) for xi in x) + '\\end{pmatrix}'
    y_vec = '\\begin{pmatrix}' + ' & '.join(str(yi) for yi in c['y']) + '\\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Batch Normalisation',
        '',
        'Apply batch normalisation to the following batch of 5 values:',
        '',
        f'$$\\mathbf{{x}} = {x_vec}$$',
        '',
        f'Using scale parameter $\\gamma = {p["gamma"]}$ and shift parameter $\\beta = {p["beta"]}$.',
        '',
        'The batch norm formula is:',
        r'$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B}}, \qquad y_i = \gamma \hat{x}_i + \beta$$',
        '',
        'where $\\mu_B$ and $\\sigma^2_B$ are the **biased** batch mean and variance ($m=5$).',
        '',
        'Give the output $\\mathbf{y}$ as a $1 \\times 5$ row vector (rounded to 1 d.p.).',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** $\\mathbf{y} =$ `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $\\mathbf{{y}} = {y_vec}$',
            '',
            '> **Hint:** Compute mean and biased variance of the batch.'
            ' Normalise each element, then scale by $\\gamma$ and shift by $\\beta$.',
        ]

    else:
        x_hat_vec = '\\begin{pmatrix}' + ' & '.join(str(v) for v in c['x_hat']) + '\\end{pmatrix}'
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 - Compute batch mean and variance.**',
            '',
            f'$$\\mu_B = \\frac{{1}}{{5}}({" + ".join(str(xi) for xi in x)}) = {c["mu"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['mean'])]
        lines += [
            '',
            f'$$\\sigma^2_B = \\frac{{1}}{{5}} \\sum_i (x_i - \\mu_B)^2 = {c["sigma2"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['variance'])]
        lines += [
            '',
            '**Step 2 - Normalise.**',
            '',
            f'$$\\hat{{\\mathbf{{x}}}} = {x_hat_vec}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['normalized'])]
        lines += [
            '',
            '**Step 3 - Scale and shift.**',
            '',
            f'$$y_i = {p["gamma"]} \\cdot \\hat{{x}}_i + {p["beta"]}$$',
            '',
            f'$$\\mathbf{{y}} = {y_vec}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['output'])]

    return '\n'.join(lines)
