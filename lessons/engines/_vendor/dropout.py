"""Generator for Dropout (inverted dropout) questions.

Given a neuron activation vector, a dropout mask, and a keep probability,
compute the scaled training output and confirm the inference output.

Inverted dropout scales by 1/p during training so inference needs no adjustment.

Modes:
    'blank'  - question only
    'answer' - answers + hint
    'full'   - step-by-step solution
"""
import numpy as np

# Mark allocation per step — must sum to 10
MARKS = {
    'train_output': 7,  # training output with inverted dropout
    'inference': 3,     # inference output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    n = 6
    x = [round(float(rng.uniform(0.1, 2.5)), 1) for _ in range(n)]
    # Keep probabilities: 0.5, 0.6, 0.7, 0.8
    p = float(rng.choice([0.5, 0.6, 0.7, 0.8]))
    # Generate mask: exactly round(p*n) ones
    n_keep = round(p * n)
    mask = [0] * n
    keep_idx = rng.choice(n, size=n_keep, replace=False)
    for i in keep_idx:
        mask[int(i)] = 1
    return {'x': x, 'p': p, 'mask': mask}


def compute(params):
    x, p, mask = params['x'], params['p'], params['mask']
    # Inverted dropout: scale active units by 1/p during training
    y_train = [round(xi * mi / p, 2) for xi, mi in zip(x, mask)]
    # At inference: output equals input unchanged
    y_inf = [xi for xi in x]
    return {'y_train': y_train, 'y_inf': y_inf}


def render(params, computed, mode='full', question_num=None, marks=False):
    p_d, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    x, p, mask = p_d['x'], p_d['p'], p_d['mask']
    drop_rate = round(1 - p, 2)

    def vec(vals):
        return '\\begin{pmatrix}' + ' & '.join(str(v) for v in vals) + '\\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Dropout (Inverted Dropout)',
        '',
        f'A fully-connected layer outputs the following activation vector during training:',
        '',
        f'$$\\mathbf{{x}} = {vec(x)}$$',
        '',
        f'Dropout is applied with **drop rate** $r = {drop_rate}$ (keep probability $p = {p}$).',
        'Using **inverted dropout**, active units are scaled by $1/p$ during training',
        'so that inference requires no rescaling.',
        '',
        f'The dropout mask for this forward pass is:',
        '',
        f'$$\\mathbf{{m}} = {vec(mask)}$$',
        '',
        '**Part (a):** Compute the training output $y_i$ (round to 2 d.p.).',
        '',
        '**Part (b):** What is the output at **inference** time for the same $\\mathbf{x}$?',
        '',
    ]

    if mode == 'blank':
        lines += [
            f'**Part (a):** $\\mathbf{{y}}_\\text{{train}} =$ `_______________`',
            '',
            f'**Part (b):** $\\mathbf{{y}}_\\text{{inf}} =$ `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $\\mathbf{{y}}_\\text{{train}} = {vec(c["y_train"])}$',
            '',
            f'**Part (b):** $\\mathbf{{y}}_\\text{{inf}} = {vec(c["y_inf"])}$',
            '',
            f'> **Hint:** Multiply each activation by its mask entry, then divide by $p = {p}$.'
            f' At inference the full network runs with no masking or scaling.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) — Training output with inverted dropout.**',
            '',
            'Apply mask and scale: $y_i = x_i \\cdot m_i / p$',
            '',
        ]
        for i, (xi, mi, yi) in enumerate(zip(x, mask, c['y_train'])):
            if mi == 1:
                lines.append(f'$y_{i+1} = {xi} \\times 1 \\;/\\; {p} = {yi}$  ')
            else:
                lines.append(f'$y_{i+1} = {xi} \\times 0 \\;/\\; {p} = 0.0$  ')
        lines += [
            '',
            f'$$\\mathbf{{y}}_\\text{{train}} = {vec(c["y_train"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['train_output'])]
        lines += [
            '',
            '**Part (b) — Inference output.**',
            '',
            'With inverted dropout no rescaling is needed at inference — the network',
            'runs at full capacity with all units active:',
            '',
            f'$$\\mathbf{{y}}_\\text{{inf}} = \\mathbf{{x}} = {vec(c["y_inf"])}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['inference'])]

    return '\n'.join(lines)
