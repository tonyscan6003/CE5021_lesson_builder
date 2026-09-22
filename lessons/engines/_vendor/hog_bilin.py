"""Generator for HOG bilinear magnitude interpolation questions.

Given a pixel magnitude mu and direction theta (degrees), distribute mu
between the two nearest orientation histogram bins using bilinear interpolation.

With 9 bins covering 0°–180° (bin centres at 0°, 20°, …, 160°, width 20°):
    mu1 = mu * |phi2 - theta| / |phi2 - phi1|   (contribution to nearer bin)
    mu2 = mu * |theta - phi1| / |phi2 - phi1|   (contribution to farther bin)

Modes:
    'blank'  – question only
    'answer' – question + answer vector + hint
    'full'   – question + step-by-step derivation
"""
import numpy as np
from ._utils import pmatrix

# Mark allocation per step — must sum to 10
MARKS = {
    'gradient': 3,          # identifying nearest bin centres
    'bin_contributions': 5, # mu1 computation
    'weighted': 2,          # mu2 computation and verification
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


_N_BINS = 9
_BIN_WIDTH = 180.0 / _N_BINS   # 20°
_BIN_CENTRES = [k * _BIN_WIDTH for k in range(_N_BINS)]  # 0,20,...,160


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    mu = round(float(rng.uniform(10, 150)), 0)
    # Choose theta not exactly on a bin centre
    while True:
        theta = round(float(rng.uniform(1, 179)), 1)
        # Must not land exactly on a bin boundary/centre for a meaningful question
        remainder = theta % _BIN_WIDTH
        if 2.0 < remainder < _BIN_WIDTH - 2.0:
            break
    return {'mu': int(mu), 'theta': theta, 'n_bins': _N_BINS,
            'bin_width': _BIN_WIDTH}


def compute(params):
    theta = params['theta']
    mu = params['mu']
    bw = params['bin_width']

    # Find which bin interval theta falls in
    bin_idx = int(theta // bw)
    phi1 = round(bin_idx * bw, 2)
    phi2 = round((bin_idx + 1) * bw, 2)
    # Handle edge case where theta == 180
    if phi2 > 180:
        phi2 = 180.0

    mu1 = round(mu * abs(phi2 - theta) / abs(phi2 - phi1), 2)
    mu2 = round(mu * abs(theta - phi1) / abs(phi2 - phi1), 2)

    return {
        'phi1': phi1, 'phi2': phi2,
        'bin1': bin_idx, 'bin2': bin_idx + 1,
        'mu1': mu1, 'mu2': mu2,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    mu, theta = p['mu'], p['theta']
    phi1, phi2 = c['phi1'], c['phi2']
    bw = p['bin_width']

    centres_str = ', '.join(f'{int(b)}°' for b in _BIN_CENTRES)

    lines = [
        f'## Question{num_str}: HOG Bilinear Magnitude Interpolation',
        '',
        f'In the HOG algorithm a histogram is built in each cell using magnitude'
        f' and direction information. The histogram uses ${p["n_bins"]}$ orientation'
        f' bins with centres at ${centres_str}$ (bin width ${int(bw)}°$).',
        '',
        f'For a pixel with magnitude $\\mu = {mu}$ and direction $\\theta = {theta}°$,'
        f' calculate the magnitude contribution to the **two nearest bins** using bilinear'
        f' interpolation:',
        '',
        r'$$\mu_1 = \mu \frac{|\phi_2 - \theta|}{|\phi_2 - \phi_1|}, \qquad'
        r'\mu_2 = \mu \frac{|\theta - \phi_1|}{|\phi_2 - \phi_1|}$$',
        '',
        r'Give your answer as a $1\times 2$ matrix $[\mu_1,\;\mu_2]$.',
        '',
    ]

    result_vec = np.array([c['mu1'], c['mu2']])

    if mode == 'blank':
        lines.append('**Answer:** `[ ___ , ___ ]`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $[\\mu_1,\\;\\mu_2] = [{c["mu1"]},\\;{c["mu2"]}]$',
            '',
            f'> **Hint:** $\\theta = {theta}°$ falls between bin {c["bin1"]}'
            f' ($\\phi_1 = {phi1}°$) and bin {c["bin2"]} ($\\phi_2 = {phi2}°$).'
            r' Contribution to bin 1 is inversely proportional to distance from'
            r' $\phi_2$: $\mu_1 = \mu\,|\phi_2 - \theta| / |\phi_2 - \phi_1|$.'
            r' Contribution to bin 2 is inversely proportional to distance from'
            r' $\phi_1$: $\mu_2 = \mu\,|\theta - \phi_1| / |\phi_2 - \phi_1|$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'**Step 1 – Identify the two nearest bin centres.**',
            '',
            f'$\\theta = {theta}°$ lies between bin {c["bin1"]} at $\\phi_1 = {phi1}°$'
            f' and bin {c["bin2"]} at $\\phi_2 = {phi2}°$.',
        ]
        if marks:
            lines += ['', _mark(MARKS['gradient'])]
        lines += [
            '',
            '**Step 2 – Apply bilinear interpolation.**',
            '',
            r'The contribution to each bin is **inversely** proportional to the'
            r' distance from the *other* bin centre.',
            '',
            f'$$\\mu_1 = \\mu \\frac{{|\\phi_2 - \\theta|}}{{|\\phi_2 - \\phi_1|}}'
            f'= {mu} \\cdot \\frac{{|{phi2} - {theta}|}}{{|{phi2} - {phi1}|}}'
            f'= {c["mu1"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['bin_contributions'])]
        lines += [
            '',
            f'$$\\mu_2 = \\mu \\frac{{|\\theta - \\phi_1|}}{{|\\phi_2 - \\phi_1|}}'
            f'= {mu} \\cdot \\frac{{|{theta} - {phi1}|}}{{|{phi2} - {phi1}|}}'
            f'= {c["mu2"]}$$',
            '',
            '**Verification:** $\\mu_1 + \\mu_2 = '
            f'{c["mu1"]} + {c["mu2"]} = {round(c["mu1"]+c["mu2"],2)}$'
            f' (should equal $\\mu = {mu}$)',
        ]
        if marks:
            lines += ['', _mark(MARKS['weighted'])]

    return '\n'.join(lines)
