"""Generator for Inception Block parameter count questions (Inception v1 style).

An inception module has four parallel branches:
  Branch 1: 1×1 conv (C_in → n1)
  Branch 2: 1×1 bottleneck → 3×3 conv (C_in → r3 → n3)
  Branch 3: 1×1 bottleneck → 5×5 conv (C_in → r5 → n5)
  Branch 4: 3×3 max pool (no params) → 1×1 proj conv (C_in → np)

Students count weights per branch and total, then compute total output channels.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'branch1': 1,   # branch 1 weights
    'branch2': 2,   # branch 2 weights (2a + 2b)
    'branch3': 2,   # branch 3 weights (3a + 3b)
    'branch4': 1,   # branch 4 weights
    'total': 2,     # total weight count
    'c_out': 2,     # output channels
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    c_in = int(rng.choice([64, 96, 128, 192]))
    n1   = int(rng.choice([16, 32, 48, 64]))       # branch 1 output
    r3   = int(rng.choice([16, 24, 32]))             # branch 2 bottleneck
    n3   = int(rng.choice([32, 48, 64, 96]))         # branch 2 output
    r5   = int(rng.choice([4, 8, 12, 16]))           # branch 3 bottleneck
    n5   = int(rng.choice([8, 16, 24, 32]))          # branch 3 output
    np_  = int(rng.choice([16, 24, 32]))             # branch 4 pool proj output
    return {'C_in': c_in, 'n1': n1, 'r3': r3, 'n3': n3,
            'r5': r5, 'n5': n5, 'np': np_}


def compute(params):
    C, n1, r3, n3, r5, n5, np_ = (
        params['C_in'], params['n1'], params['r3'], params['n3'],
        params['r5'], params['n5'], params['np'],
    )
    w1  = C * n1                     # branch 1: 1×1
    w2a = C * r3                     # branch 2: 1×1 bottleneck
    w2b = r3 * 3 * 3 * n3           # branch 2: 3×3
    w3a = C * r5                     # branch 3: 1×1 bottleneck
    w3b = r5 * 5 * 5 * n5           # branch 3: 5×5
    w4  = C * np_                    # branch 4: 1×1 proj after max pool
    total = w1 + w2a + w2b + w3a + w3b + w4
    c_out = n1 + n3 + n5 + np_
    return {
        'w1': w1, 'w2a': w2a, 'w2b': w2b,
        'w3a': w3a, 'w3b': w3b, 'w4': w4,
        'total': total, 'c_out': c_out,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    C, n1, r3, n3, r5, n5, np_ = (
        p['C_in'], p['n1'], p['r3'], p['n3'],
        p['r5'], p['n5'], p['np'],
    )

    lines = [
        f'## Question{num_str}: Inception Block Parameter Count',
        '',
        f'An inception module receives a feature map with $C_{{in}} = {C}$ channels.',
        'It has four parallel branches (count **weights only**, ignore biases):',
        '',
        f'| Branch | Operation | Filters |',
        f'|--------|-----------|---------|',
        f'| 1 | $1 \\times 1$ conv | $C_{{in}} \\to {n1}$ |',
        f'| 2a | $1 \\times 1$ bottleneck | $C_{{in}} \\to {r3}$ |',
        f'| 2b | $3 \\times 3$ conv | ${r3} \\to {n3}$ |',
        f'| 3a | $1 \\times 1$ bottleneck | $C_{{in}} \\to {r5}$ |',
        f'| 3b | $5 \\times 5$ conv | ${r5} \\to {n5}$ |',
        f'| 4a | $3 \\times 3$ max pool | (no parameters) |',
        f'| 4b | $1 \\times 1$ proj conv | $C_{{in}} \\to {np_}$ |',
        '',
        '**Part (a):** How many weights does each branch contribute?',
        '',
        '**Part (b):** What is the total weight count for the inception module?',
        '',
        '**Part (c):** What is the total number of output channels $C_\\text{out}$?',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):**',
            f'- Branch 1: `___`',
            f'- Branch 2 (2a + 2b): `___` + `___`',
            f'- Branch 3 (3a + 3b): `___` + `___`',
            f'- Branch 4: `___`',
            '',
            '**Part (b):** Total weights $=$ `___`',
            '',
            '**Part (c):** $C_\\text{out} =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            '**Part (a):**',
            f'- Branch 1: ${c["w1"]:,}$',
            f'- Branch 2: ${c["w2a"]:,} + {c["w2b"]:,} = {c["w2a"]+c["w2b"]:,}$',
            f'- Branch 3: ${c["w3a"]:,} + {c["w3b"]:,} = {c["w3a"]+c["w3b"]:,}$',
            f'- Branch 4: ${c["w4"]:,}$',
            '',
            f'**Part (b):** Total $= {c["total"]:,}$',
            '',
            f'**Part (c):** $C_\\text{{out}} = {c["c_out"]}$',
            '',
            f'> **Hint:** Weights for a conv = (kernel_h × kernel_w × C_in × C_out).'
            f' Branches run in parallel and their outputs are **concatenated**'
            f' along the channel dimension.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) — Weights per branch.**',
            '',
            f'Branch 1 ($1 \\times 1$, $C_{{in}} \\to {n1}$):',
            f'$$W_1 = 1 \\times 1 \\times {C} \\times {n1} = {c["w1"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['branch1'])]
        lines += [
            '',
            f'Branch 2a ($1 \\times 1$ bottleneck, $C_{{in}} \\to {r3}$):',
            f'$$W_{{2a}} = 1 \\times 1 \\times {C} \\times {r3} = {c["w2a"]:,}$$',
            '',
            f'Branch 2b ($3 \\times 3$, ${r3} \\to {n3}$):',
            f'$$W_{{2b}} = 3 \\times 3 \\times {r3} \\times {n3} = {c["w2b"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['branch2'])]
        lines += [
            '',
            f'Branch 3a ($1 \\times 1$ bottleneck, $C_{{in}} \\to {r5}$):',
            f'$$W_{{3a}} = 1 \\times 1 \\times {C} \\times {r5} = {c["w3a"]:,}$$',
            '',
            f'Branch 3b ($5 \\times 5$, ${r5} \\to {n5}$):',
            f'$$W_{{3b}} = 5 \\times 5 \\times {r5} \\times {n5} = {c["w3b"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['branch3'])]
        lines += [
            '',
            f'Branch 4 ($1 \\times 1$ proj after max pool, $C_{{in}} \\to {np_}$):',
            f'$$W_4 = 1 \\times 1 \\times {C} \\times {np_} = {c["w4"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['branch4'])]
        lines += [
            '',
            '**Part (b) — Total weights.**',
            '',
            f'$$W_\\text{{total}} = {c["w1"]:,} + {c["w2a"]:,} + {c["w2b"]:,}'
            f' + {c["w3a"]:,} + {c["w3b"]:,} + {c["w4"]:,} = {c["total"]:,}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['total'])]
        lines += [
            '',
            '**Part (c) — Output channels.**',
            '',
            'Outputs of all branches are concatenated along the channel axis:',
            f'$$C_\\text{{out}} = {n1} + {n3} + {n5} + {np_} = {c["c_out"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['c_out'])]

    return '\n'.join(lines)
