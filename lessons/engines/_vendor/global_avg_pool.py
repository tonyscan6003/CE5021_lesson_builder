"""Generator for Global Average Pooling questions.

Compares GAP + small FC classifier against Flatten + large FC classifier.
Students compute GAP output size, parameter counts, and savings.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'gap_output': 2,   # GAP output size
    'gap_params': 3,   # GAP + FC parameter count
    'flat_params': 3,  # Flatten + FC parameter count
    'savings': 2,      # parameter savings
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    h = int(rng.choice([4, 6, 7, 8, 14]))
    c = int(rng.choice([128, 256, 512]))
    n_classes = int(rng.choice([10, 100, 200, 1000]))
    return {'H': h, 'W': h, 'C': c, 'N': n_classes}


def compute(params):
    H, W, C, N = params['H'], params['W'], params['C'], params['N']
    # GAP: averages each channel → output is C-dim vector
    gap_out = C
    gap_fc_params = C * N          # FC from C → N (no bias counted for simplicity)
    # Flatten + FC
    flat_out = H * W * C
    flat_fc_params = flat_out * N
    # Savings
    param_savings = flat_fc_params - gap_fc_params
    ratio = round(gap_fc_params / flat_fc_params, 4)
    return {
        'gap_out': gap_out,
        'gap_fc_params': gap_fc_params,
        'flat_out': flat_out,
        'flat_fc_params': flat_fc_params,
        'param_savings': param_savings,
        'ratio': ratio,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    H, W, C, N = p['H'], p['W'], p['C'], p['N']

    lines = [
        f'## Question{num_str}: Global Average Pooling',
        '',
        f'A CNN produces a feature map of size ${H} \\times {W} \\times {C}$'
        f' before the final classifier, which must output $N = {N}$ class scores.',
        '',
        'Two classifier heads are compared (count weights only, ignore biases):',
        '',
        '- **Option A — GAP + FC:** Global average pooling collapses each channel'
        ' to a single value, then one fully-connected layer maps to $N$ classes.',
        '- **Option B — Flatten + FC:** The feature map is flattened to a vector,'
        ' then one fully-connected layer maps directly to $N$ classes.',
        '',
        '**Part (a):** What is the output size of the GAP operation?',
        '',
        '**Part (b):** How many weights does each classifier head require?',
        '',
        '**Part (c):** How many weights are saved by using GAP instead of Flatten?',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** GAP output size $=$ `___`',
            '',
            '**Part (b):** Option A (GAP + FC) $=$ `___` weights ;'
            '  Option B (Flatten + FC) $=$ `___` weights',
            '',
            '**Part (c):** Savings $=$ `___` weights',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** GAP output size $= {c["gap_out"]}$',
            '',
            f'**Part (b):** Option A $= {c["gap_fc_params"]:,}$ weights ;'
            f'  Option B $= {c["flat_fc_params"]:,}$ weights',
            '',
            f'**Part (c):** Savings $= {c["param_savings"]:,}$ weights'
            f' (Option A uses only {100*c["ratio"]:.2f}% of Option B parameters)',
            '',
            f'> **Hint:** GAP outputs one value per channel ($C = {C}$).'
            f' Flatten outputs $H \\times W \\times C = {c["flat_out"]:,}$.'
            f' Weights for FC = input\_size $\\times$ {N}.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) — GAP output size.**',
            '',
            f'GAP averages each of the $C = {C}$ channels over its ${H} \\times {W}$'
            f' spatial locations, yielding one scalar per channel:',
            '',
            f'$$\\text{{GAP output}} = C = {c["gap_out"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['gap_output'])]
        lines += [
            '',
            '**Part (b) — Classifier parameter counts.**',
            '',
            f'Option A (GAP + FC): FC maps ${C}$ inputs to ${N}$ outputs:',
            f'$$W_A = C \\times N = {C} \\times {N} = {c["gap_fc_params"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['gap_params'])]
        lines += [
            '',
            f'Option B (Flatten + FC): flatten gives ${H} \\times {W} \\times {C}'
            f' = {c["flat_out"]:,}$ inputs; FC maps to ${N}$ outputs:',
            f'$$W_B = H \\cdot W \\cdot C \\times N'
            f' = {c["flat_out"]:,} \\times {N} = {c["flat_fc_params"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['flat_params'])]
        lines += [
            '',
            '**Part (c) — Parameter savings.**',
            '',
            f'$$W_B - W_A = {c["flat_fc_params"]:,} - {c["gap_fc_params"]:,}'
            f' = {c["param_savings"]:,}$$',
            '',
            f'Option A (GAP) uses only $\\frac{{{c["gap_fc_params"]:,}}}{{{c["flat_fc_params"]:,}}}'
            f' = {c["ratio"]}$ of the parameters, a reduction factor of'
            f' $H \\times W = {H} \\times {W} = {H*W}\\times$.',
        ]
        if marks:
            lines += ['', _mark(MARKS['savings'])]

    return '\n'.join(lines)
