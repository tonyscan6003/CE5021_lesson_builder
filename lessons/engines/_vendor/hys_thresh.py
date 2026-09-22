"""Hysteresis thresholding question generator (Canny edge detection).

A Canny edge detector uses upper threshold T_max and lower threshold T_min.
Pixel P (always < T_max) is connected to pixel A (always > T_max).
Determine whether P is accepted or rejected as an edge pixel.

Two equally-likely scenarios:
  Scenario 0 — P < T_min           → False / rejected
               (regardless of A — T_min is a hard lower bound)
  Scenario 1 — T_min ≤ P < T_max   → True / accepted
               (weak edge connected to the strong edge A)

Answer is 'T' (accepted) or 'F' (rejected).
"""
import numpy as np

MARKS = {
    'rule':   4,   # correctly identify and state the applicable rule
    'answer': 6,   # correct T/F conclusion with justification
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)

    # --- thresholds --------------------------------------------------------
    Tmax = int(rng.integers(120, 231))
    # Tmin: 35–70 % of Tmax, and at least 20 units below Tmax
    Tmin_lo = max(25, int(Tmax * 0.35))
    Tmin_hi = min(Tmax - 20, int(Tmax * 0.70))
    if Tmin_hi < Tmin_lo:
        Tmin_hi = Tmin_lo + 10
    Tmin = int(rng.integers(Tmin_lo, Tmin_hi + 1))

    # --- scenario ----------------------------------------------------------
    scenario = int(rng.integers(0, 2))   # 0 = reject, 1 = accept

    if scenario == 0:                    # P < Tmin  → reject
        # Keep P comfortably below Tmin so the case is unambiguous
        gap = max(5, (Tmin - 1) // 4)
        P_hi = Tmin - gap
        P_lo = max(1, P_hi - max(10, Tmin // 3))
        if P_lo > P_hi:
            P_lo, P_hi = 1, max(1, Tmin - 2)
        P = int(rng.integers(P_lo, P_hi + 1))
    else:                                # Tmin ≤ P < Tmax  → accept
        # Keep P comfortably inside the weak-edge band
        band = Tmax - Tmin
        margin = max(3, band // 8)
        P_lo = Tmin + margin
        P_hi = Tmax - margin
        if P_lo > P_hi:
            P_lo, P_hi = Tmin, Tmax - 1
        P = int(rng.integers(P_lo, P_hi + 1))

    # A is always a strong-edge pixel: A > Tmax
    A = int(rng.integers(Tmax + 5, Tmax + 91))

    return {
        'Tmax':     Tmax,
        'Tmin':     Tmin,
        'P':        P,
        'A':        A,
        'scenario': scenario,   # 0 = reject, 1 = accept (stored for transparency)
    }


def compute(params):
    accepted = params['P'] >= params['Tmin']   # and P < Tmax (always), with A > Tmax (always)
    return {'accepted': accepted}


# ---------------------------------------------------------------------------
# Shared rule text (used in both answer hint and solution)
# ---------------------------------------------------------------------------
_RULE_LINES = [
    'Hysteresis thresholding in the Canny algorithm uses two thresholds'
    ' $T_{\\rm max}$ (upper) and $T_{\\rm min}$ (lower) to classify each pixel $P$:',
    '',
    '- **Accepted (strong edge):** $P \\geq T_{\\rm max}$',
    '- **Accepted (weak edge):** $T_{\\rm min} \\leq P < T_{\\rm max}$'
    ' **and** $P$ is connected to a pixel $A > T_{\\rm max}$',
    '- **Rejected:** $P < T_{\\rm min}$ — regardless of any connected pixels',
]


def render(params, computed, mode='full', question_num=None, marks=False):
    prms, c = params, computed
    num_str  = f' {question_num}' if question_num is not None else ''
    Tmax     = prms['Tmax']
    Tmin     = prms['Tmin']
    P        = prms['P']
    A        = prms['A']
    accepted = c['accepted']
    ans_str  = 'T (True / accepted)' if accepted else 'F (False / rejected)'

    lines = [
        f'## Question{num_str}: Hysteresis Thresholding (Canny Edge Detection)',
        '',
        f'A Canny edge detector is configured with hysteresis thresholds'
        f' $T_{{\\rm max}} = {Tmax}$ and $T_{{\\rm min}} = {Tmin}$.',
        '',
        f'Pixel $P = {P}$ is connected to pixel $A = {A}$.',
        '',
        'Will pixel $P$ be **accepted** or **rejected** as part of an edge?',
        'Give your answer as **T** (True / accepted) or **F** (False / rejected).',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `___`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** {ans_str}',
            '',
            '> **Hint:** ' + _RULE_LINES[0],
            '>',
        ]
        for rl in _RULE_LINES[2:]:     # skip the blank separator line
            lines.append(f'> {rl}')

    else:   # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += ['**Hysteresis thresholding rule:**', ''] + _RULE_LINES
        if marks:
            lines += ['', _mark(MARKS['rule'])]

        lines += ['', '**Applying to the given values:**', '']

        if not accepted:    # scenario 0: P < Tmin → reject
            lines += [
                f'$P = {P}$ and $T_{{\\rm min}} = {Tmin}$:',
                '',
                f'$$P = {P} < T_{{\\rm min}} = {Tmin}$$',
                '',
                f'$P$ is below the lower threshold, so it is **always rejected**'
                f' regardless of the connected pixel $A = {A}$.',
                '',
                '**Answer: F (rejected)**',
            ]
        else:               # scenario 1: Tmin ≤ P < Tmax → accept
            lines += [
                f'Check whether $P$ falls in the weak-edge band'
                f' $[T_{{\\rm min}},\\, T_{{\\rm max}})$:',
                '',
                f'$$T_{{\\rm min}} = {Tmin} \\leq P = {P} < T_{{\\rm max}} = {Tmax} \\quad \\checkmark$$',
                '',
                f'Check whether $A$ is a strong-edge pixel:',
                '',
                f'$$A = {A} > T_{{\\rm max}} = {Tmax} \\quad \\checkmark$$',
                '',
                f'$P$ is a **weak-edge pixel** connected to the strong-edge pixel $A$,'
                f' so $P$ is accepted.',
                '',
                '**Answer: T (accepted)**',
            ]

        if marks:
            lines += ['', _mark(MARKS['answer'])]

    return '\n'.join(lines)
