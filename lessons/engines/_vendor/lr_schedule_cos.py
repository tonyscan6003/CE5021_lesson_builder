"""Generator for Cosine Annealing learning rate schedule questions.

η(t) = η_min + ½(η_max − η_min)(1 + cos(π·t/T))

Students compute the LR at given epochs across the annealing cycle.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import math
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'boundary': 3,    # boundary values eta(0) and eta(T)
    'mid_values': 4,  # mid-cycle values
    'midpoint': 3,    # midpoint epoch
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    eta_max = float(rng.choice([0.1, 0.05, 0.01]))
    eta_min = round(eta_max / float(rng.choice([10, 100])), 4)
    T = int(rng.choice([50, 100, 200]))
    # Query at t = 0, T/4, T/2, 3T/4, T — pick three non-trivial ones
    candidates = [T // 4, T // 2, (3 * T) // 4]
    t1, t2, t3 = candidates
    return {'eta_max': eta_max, 'eta_min': eta_min, 'T': T,
            't1': t1, 't2': t2, 't3': t3}


def compute(params):
    eta_max = params['eta_max']
    eta_min = params['eta_min']
    T = params['T']

    def cosine_lr(t):
        val = eta_min + 0.5 * (eta_max - eta_min) * (1 + math.cos(math.pi * t / T))
        return round(val, 6)

    return {
        'eta_0':  cosine_lr(0),
        'eta_t1': cosine_lr(params['t1']),
        'eta_t2': cosine_lr(params['t2']),
        'eta_t3': cosine_lr(params['t3']),
        'eta_T':  cosine_lr(T),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    eta_max, eta_min, T = p['eta_max'], p['eta_min'], p['T']
    t1, t2, t3 = p['t1'], p['t2'], p['t3']

    lines = [
        f'## Question{num_str}: Cosine Annealing Learning Rate Schedule',
        '',
        'A network is trained with **cosine annealing** over $T$ epochs:',
        '',
        r'$$\eta(t) = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})'
        r'\left(1 + \cos\!\left(\frac{\pi\, t}{T}\right)\right)$$',
        '',
        f'with $\\eta_{{\\max}} = {eta_max}$, $\\eta_{{\\min}} = {eta_min}$,'
        f' $T = {T}$ epochs.',
        '',
        f'**Part (a):** Compute $\\eta(0)$ and $\\eta(T)$.'
        f' Verify these equal $\\eta_{{\\max}}$ and $\\eta_{{\\min}}$ respectively.',
        '',
        f'**Part (b):** Compute $\\eta({t1})$, $\\eta({t2})$, and $\\eta({t3})$'
        f' (round to 4 d.p.).',
        '',
        f'**Part (c):** At what epoch does the learning rate equal'
        f' the midpoint $\\frac{{\\eta_{{\\max}} + \\eta_{{\\min}}}}{{2}}$?',
        '',
    ]

    midpoint = round((eta_max + eta_min) / 2, 6)

    if mode == 'blank':
        lines += [
            f'**Part (a):** $\\eta(0) =$ `___` , $\\eta(T) =$ `___`',
            '',
            f'**Part (b):** $\\eta({t1}) =$ `___` , $\\eta({t2}) =$ `___`'
            f' , $\\eta({t3}) =$ `___`',
            '',
            '**Part (c):** Midpoint epoch $=$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $\\eta(0) = {c["eta_0"]}$ , $\\eta(T) = {c["eta_T"]}$',
            '',
            f'**Part (b):** $\\eta({t1}) = {c["eta_t1"]}$ ,'
            f' $\\eta({t2}) = {c["eta_t2"]}$ ,'
            f' $\\eta({t3}) = {c["eta_t3"]}$',
            '',
            f'**Part (c):** $t = {T // 2}$ (half-period, where $\\cos(\\pi/2)=0$)',
            '',
            r'> **Hint:** At $t=0$: $\cos(0)=1$ so $\eta=\eta_{\max}$.'
            r' At $t=T$: $\cos(\pi)=-1$ so $\eta=\eta_{\min}$.'
            r' The midpoint occurs when $\cos(\pi t/T)=0$, i.e. $t=T/2$.',
        ]

    else:
        def cos_str(t):
            frac = f'\\pi \\cdot {t} / {T}'
            val = math.cos(math.pi * t / T)
            return frac, round(val, 4)

        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) — Boundary values.**',
            '',
            r'At $t = 0$: $\cos(0) = 1$',
            f'$$\\eta(0) = {eta_min} + \\tfrac{{1}}{{2}}({eta_max} - {eta_min})(1 + 1)'
            f' = {eta_min} + {eta_max - eta_min} = {c["eta_0"]}$$',
            '',
            r'At $t = T$: $\cos(\pi) = -1$',
            f'$$\\eta(T) = {eta_min} + \\tfrac{{1}}{{2}}({eta_max} - {eta_min})(1 - 1)'
            f' = {eta_min} = {c["eta_T"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['boundary'])]
        lines += [
            '',
            '**Part (b) — Mid-cycle values.**',
            '',
        ]
        for t, key, label in [(t1, 'eta_t1', t1), (t2, 'eta_t2', t2), (t3, 'eta_t3', t3)]:
            frac, cos_val = cos_str(t)
            lines += [
                f'$t = {t}$: $\\cos({frac}) = {cos_val}$',
                f'$$\\eta({t}) = {eta_min} + \\tfrac{{1}}{{2}}({eta_max} - {eta_min})'
                f'(1 + {cos_val}) = {c[key]}$$',
                '',
            ]
        if marks:
            lines += [_mark(MARKS['mid_values']), '']
        lines += [
            '**Part (c) — Midpoint epoch.**',
            '',
            f'The midpoint LR is $({eta_max} + {eta_min})/2 = {midpoint}$.',
            '',
            r'This requires $\cos(\pi t / T) = 0$, i.e. $\pi t / T = \pi/2$,'
            r' giving $t = T/2$:',
            '',
            f'$$t = {T} / 2 = {T // 2}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['midpoint'])]

    return '\n'.join(lines)
