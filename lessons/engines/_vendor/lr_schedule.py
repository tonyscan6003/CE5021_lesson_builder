"""Generator for Learning Rate Schedule questions (step decay).

Step decay: η(t) = η₀ × γ^floor(t / S)
Students compute the LR at given epochs and find when it drops below a threshold.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import math
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'lr_t1': 3,       # LR at t1
    'lr_t2': 3,       # LR at t2
    'epoch_below': 4, # first epoch below threshold
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    eta0 = float(rng.choice([0.1, 0.05, 0.01]))
    gamma = float(rng.choice([0.1, 0.2, 0.5]))
    S = int(rng.choice([10, 20, 30]))
    # Pick two query epochs that span at least 2 decay steps
    t1 = int(rng.integers(S + 1, 2 * S))      # one decay applied
    t2 = int(rng.integers(2 * S + 1, 4 * S))  # two or three decays
    # Threshold: somewhere below eta0 but reachable in a few steps
    threshold = round(eta0 * gamma ** 2, 6)
    return {
        'eta0': eta0, 'gamma': gamma, 'S': S,
        't1': t1, 't2': t2, 'threshold': threshold,
    }


def compute(params):
    eta0 = params['eta0']
    gamma = params['gamma']
    S = params['S']
    t1, t2 = params['t1'], params['t2']
    threshold = params['threshold']

    def lr_at(t):
        return round(eta0 * (gamma ** (t // S)), 6)

    eta_t1 = lr_at(t1)
    eta_t2 = lr_at(t2)

    # First epoch where LR drops strictly below threshold
    # LR drops below threshold when eta0 * gamma^n < threshold
    # gamma^n < threshold/eta0  →  n > log(threshold/eta0) / log(gamma)
    if gamma < 1 and threshold < eta0:
        n_needed = math.ceil(math.log(threshold / eta0) / math.log(gamma))
        epoch_below = n_needed * S
    else:
        epoch_below = None

    return {'eta_t1': eta_t1, 'eta_t2': eta_t2, 'epoch_below': epoch_below}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    eta0, gamma, S = p['eta0'], p['gamma'], p['S']
    t1, t2 = p['t1'], p['t2']
    threshold = p['threshold']

    n1 = t1 // S
    n2 = t2 // S

    lines = [
        f'## Question{num_str}: Learning Rate Step Decay',
        '',
        'A neural network is trained using **step decay** learning rate scheduling:',
        '',
        f'$$\\eta(t) = \\eta_0 \\cdot \\gamma^{{\\lfloor t / S \\rfloor}}$$',
        '',
        f'with initial learning rate $\\eta_0 = {eta0}$, decay factor $\\gamma = {gamma}$,'
        f' and step size $S = {S}$ epochs.',
        '',
        f'**Part (a):** Compute the learning rate at epoch $t = {t1}$.',
        '',
        f'**Part (b):** Compute the learning rate at epoch $t = {t2}$.',
        '',
        f'**Part (c):** At which epoch does the learning rate first drop'
        f' below $\\eta_\\text{{thresh}} = {threshold}$?',
        '',
    ]

    if mode == 'blank':
        lines += [
            f'**Part (a):** $\\eta({t1}) =$ `___`',
            '',
            f'**Part (b):** $\\eta({t2}) =$ `___`',
            '',
            '**Part (c):** First epoch below threshold $=$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $\\eta({t1}) = {c["eta_t1"]}$',
            '',
            f'**Part (b):** $\\eta({t2}) = {c["eta_t2"]}$',
            '',
            f'**Part (c):** Epoch $= {c["epoch_below"]}$',
            '',
            f'> **Hint:** Count how many full steps of {S} epochs fit before $t$:'
            f' $n = \\lfloor t / {S} \\rfloor$. Then $\\eta = {eta0} \\times {gamma}^n$.'
            f' For part (c) solve $\\gamma^n < \\eta_\\text{{thresh}} / \\eta_0$ for $n$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a).**',
            '',
            f'Number of completed decay steps at $t = {t1}$:'
            f' $n = \\lfloor {t1} / {S} \\rfloor = {n1}$',
            '',
            f'$$\\eta({t1}) = {eta0} \\times {gamma}^{{{n1}}} = {c["eta_t1"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['lr_t1'])]
        lines += [
            '',
            '**Part (b).**',
            '',
            f'Number of completed decay steps at $t = {t2}$:'
            f' $n = \\lfloor {t2} / {S} \\rfloor = {n2}$',
            '',
            f'$$\\eta({t2}) = {eta0} \\times {gamma}^{{{n2}}} = {c["eta_t2"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['lr_t2'])]
        lines += [
            '',
            '**Part (c).**',
            '',
            f'Need $\\eta_0 \\cdot \\gamma^n < {threshold}$, i.e.'
            f' $\\gamma^n < {threshold} / {eta0} = {round(threshold / eta0, 6)}$.',
            '',
            f'Taking logarithms: $n > \\dfrac{{\\ln({round(threshold/eta0, 6)})}}{{\\ln({gamma})}}$',
            '',
        ]
        if c['epoch_below'] is not None:
            n_needed = c['epoch_below'] // S
            lines += [
                f'The smallest integer satisfying this is $n = {n_needed}$,'
                f' reached at epoch $t = {n_needed} \\times {S} = {c["epoch_below"]}$.',
                '',
                f'$$\\boxed{{\\text{{First epoch below threshold}} = {c["epoch_below"]}}}$$',
            ]
        if marks:
            lines += ['', _mark(MARKS['epoch_below'])]

    return '\n'.join(lines)
