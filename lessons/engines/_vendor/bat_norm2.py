"""Generator for Batch Norm Global Stats + Inference questions (two parts).

Part (a): Update global mean/variance using exponential moving average.
Part (b): Apply batch norm at inference time using updated global stats.

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts

Pass marks=True to render() to show mark allocation per step (full mode only).
Total marks sum to 10.
"""
import math
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'batch_stats': 3,  # batch mean and variance
    'ema_mean': 2,     # EMA-updated global mean
    'ema_var': 2,      # EMA-updated global variance
    'inference': 3,    # inference normalisation output
}


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    x = [round(float(rng.uniform(1, 5)), 1) for _ in range(5)]
    mu_prev = round(float(rng.uniform(2, 5)), 2)
    sigma2_prev = round(float(rng.uniform(0.5, 2.0)), 2)
    decay = 0.95
    x0 = round(float(rng.uniform(0.1, 5.0)), 1)
    gamma = round(float(rng.uniform(0.5, 1.0)), 1)
    beta = round(float(rng.uniform(0.1, 0.5)), 1)
    return {
        'x': x, 'mu_prev': mu_prev, 'sigma2_prev': sigma2_prev,
        'decay': decay, 'x0': x0, 'gamma': gamma, 'beta': beta,
    }


def compute(params):
    x = params['x']
    mu_prev = params['mu_prev']
    sigma2_prev = params['sigma2_prev']
    d = params['decay']
    x0 = params['x0']
    gamma = params['gamma']
    beta = params['beta']

    m = len(x)
    mu_B = round(sum(x) / m, 2)
    sigma2_B = round(sum((xi - mu_B) ** 2 for xi in x) / m, 2)

    mu_new = round(mu_prev * d + mu_B * (1 - d), 2)
    sigma2_new = round(sigma2_prev * d + sigma2_B * (1 - d), 2)

    x_hat0 = round((x0 - mu_new) / math.sqrt(sigma2_new) if sigma2_new > 0 else 0.0, 2)
    y0 = round(gamma * x_hat0 + beta, 1)

    return {
        'mu_B': mu_B, 'sigma2_B': sigma2_B,
        'mu_new': mu_new, 'sigma2_new': sigma2_new,
        'x_hat0': x_hat0, 'y0': y0,
    }


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    x = p['x']
    d = p['decay']

    x_vec = '\\begin{pmatrix}' + ' & '.join(str(xi) for xi in x) + '\\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Batch Norm Global Stats + Inference',
        '',
        '**Part (a):** During training the current batch statistics are updated into the'
        ' running global statistics using exponential moving average with decay $d = {:.2f}$:'.format(d),
        '',
        f'$$\\mu \\leftarrow \\mu_- \\cdot d + \\mu_B \\cdot (1-d), '
        r'\qquad \sigma^2 \leftarrow \sigma^2_- \cdot d + \sigma^2_B \cdot (1-d)$$',
        '',
        f'Current global statistics: $\\mu_- = {p["mu_prev"]},\\; \\sigma^2_- = {p["sigma2_prev"]}$.',
        '',
        f'Current batch: $\\mathbf{{x}} = {x_vec}$',
        '',
        'Compute updated global $\\mu$ and $\\sigma^2$ (batch variance is **biased**, $m=5$). Round to 2 d.p.',
        '',
        '**Part (b):** At inference, a single input $x_0 = {}$ is normalised using the updated'.format(p["x0"]),
        f'global statistics from Part (a), with $\\gamma = {p["gamma"]},\\; \\beta = {p["beta"]}$:',
        '',
        r'$$\hat{x}_0 = \frac{x_0 - \mu}{\sqrt{\sigma^2}}, \qquad y_0 = \gamma \hat{x}_0 + \beta$$',
        '',
        'Compute $y_0$ (round to 1 d.p.).',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** $\\mu =$ `___`, $\\sigma^2 =$ `___`',
            '',
            '**Part (b):** $y_0 =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $\\mu = {c["mu_new"]},\\; \\sigma^2 = {c["sigma2_new"]}$',
            '',
            f'**Part (b):** $y_0 = {c["y0"]}$',
            '',
            '> **Hint:** For (a) compute batch mean and biased variance, then apply the EMA update.'
            ' For (b) normalise $x_0$ with updated $\\mu, \\sigma^2$, then scale/shift.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Update global statistics.**',
            '',
            f'Batch mean: $\\mu_B = ({" + ".join(str(xi) for xi in x)}) / 5 = {c["mu_B"]}$',
            '',
            f'Batch variance (biased): $\\sigma^2_B = {c["sigma2_B"]}$',
        ]
        if marks:
            lines += ['', _mark(MARKS['batch_stats'])]
        lines += [
            '',
            f'Updated global mean:',
            f'$$\\mu = {p["mu_prev"]} \\times {d} + {c["mu_B"]} \\times {round(1-d, 2)} = {c["mu_new"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['ema_mean'])]
        lines += [
            '',
            f'Updated global variance:',
            f'$$\\sigma^2 = {p["sigma2_prev"]} \\times {d} + {c["sigma2_B"]} \\times {round(1-d, 2)} = {c["sigma2_new"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['ema_var'])]
        lines += [
            '',
            '**Part (b) - Inference normalisation.**',
            '',
            f'$$\\hat{{x}}_0 = \\frac{{{p["x0"]} - {c["mu_new"]}}}{{\\sqrt{{{c["sigma2_new"]}}}}} = {c["x_hat0"]}$$',
            '',
            f'$$y_0 = {p["gamma"]} \\times {c["x_hat0"]} + {p["beta"]} = {c["y0"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['inference'])]

    return '\n'.join(lines)
