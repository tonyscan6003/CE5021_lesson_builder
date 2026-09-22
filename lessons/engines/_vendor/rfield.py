"""Generator for Receptive Field Calculation questions.

3 layers with random kernel sizes and strides.
Formula: r_{l-1} = s_l * r_l + (k_l - s_l)

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution
"""
import numpy as np
from ._exam_boxes import answer_box


# Mark allocation per step — must sum to 10
MARKS = {
    'layer_n': 3,          # r2 computation
    'layer_n_minus_1': 3,  # r1 computation
    'layer_0': 4,          # r0 final receptive field
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    k = [int(rng.choice([3, 4, 5, 6, 7])) for _ in range(3)]
    s = [int(rng.choice([1, 2, 3])), int(rng.choice([1, 2, 3])), 1]
    return {'k': k, 's': s}


def compute(params):
    k = params['k']
    s = params['s']
    r3 = 1
    r2 = s[2] * r3 + (k[2] - s[2])
    r1 = s[1] * r2 + (k[1] - s[1])
    r0 = s[0] * r1 + (k[0] - s[0])
    return {'r': [r0, r1, r2, r3]}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    k = p['k']
    s = p['s']
    r = c['r']

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Receptive Field Calculation'),
        '',
        'A CNN has **3 convolutional layers** with the following parameters:',
        '',
        '| Layer | Kernel size $k$ | Stride $s$ |',
        '|-------|----------------|-----------|',
        f'| 1     | ${k[0]}$        | ${s[0]}$  |',
        f'| 2     | ${k[1]}$        | ${s[1]}$  |',
        f'| 3     | ${k[2]}$        | ${s[2]}$  |',
        '',
        'Compute the **receptive field** $r_0$ of a single output unit in terms of input pixels.',
        '',
    ]

    if mode == 'blank':
        lines.append('**$r_0 =$ ** `_______________`')

    elif mode == 'exam':
        lines += [
            'Work backwards from the output ($r_3 = 1$) one layer at a time.',
            '',
            '**Part (a):** $r_2 =$',
            '',
            answer_box(),
            '',
            '**Part (b):** $r_1 =$',
            '',
            answer_box(),
            '',
            '**Part (c):** $r_0 =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            'The recurrence relation $r_{l-1} = s_l \\cdot r_l + (k_l - s_l)$'
            ' is used to calculate receptive field values, starting from $r_3 = 1$.',
            '',
            '**Part (a) - $r_2$.**',
            '',
            'Start from the output: $r_3 = 1$.',
            '',
            f'$$r_2 = s_3 \\cdot r_3 + (k_3 - s_3) = {s[2]} \\cdot {r[3]} + ({k[2]} - {s[2]}) = {r[2]}$$',
            '',
            _mark(MARKS['layer_n']),
        ]
        lines += [
            '',
            '**Part (b) - $r_1$.**',
            '',
            f'$$r_1 = s_2 \\cdot r_2 + (k_2 - s_2) = {s[1]} \\cdot {r[2]} + ({k[1]} - {s[1]}) = {r[1]}$$',
            '',
            _mark(MARKS['layer_n_minus_1']),
        ]
        lines += [
            '',
            '**Part (c) - $r_0$.**',
            '',
            f'$$r_0 = s_1 \\cdot r_1 + (k_1 - s_1) = {s[0]} \\cdot {r[1]} + ({k[0]} - {s[0]}) = {r[0]}$$',
            '',
            f'The receptive field is $r_0 = \\boxed{{{r[0]}}}$ pixels.',
            '',
            _mark(MARKS['layer_0']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $r_0 = {r[0]}$',
            '',
            '> **Hint:** Apply $r_{{l-1}} = s_l \\cdot r_l + (k_l - s_l)$ backwards from'
            f' $r_3 = 1$: $r_2 = {r[2]}$, $r_1 = {r[1]}$, $r_0 = {r[0]}$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            'The recurrence relation $r_{l-1} = s_l \\cdot r_l + (k_l - s_l)$'
            ' is used to calculate receptive field values, starting from $r_3 = 1$.',
            '',
            'Start from the output: $r_3 = 1$.',
            '',
            f'$$r_2 = s_3 \\cdot r_3 + (k_3 - s_3) = {s[2]} \\cdot {r[3]} + ({k[2]} - {s[2]}) = {r[2]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['layer_n'])]
        lines += [
            '',
            f'$$r_1 = s_2 \\cdot r_2 + (k_2 - s_2) = {s[1]} \\cdot {r[2]} + ({k[1]} - {s[1]}) = {r[1]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['layer_n_minus_1'])]
        lines += [
            '',
            f'$$r_0 = s_1 \\cdot r_1 + (k_1 - s_1) = {s[0]} \\cdot {r[1]} + ({k[0]} - {s[0]}) = {r[0]}$$',
            '',
            f'The receptive field is $r_0 = \\boxed{{{r[0]}}}$ pixels.',
        ]
        if marks:
            lines += ['', _mark(MARKS['layer_0'])]

    return '\n'.join(lines)
