"""Generator for Multi-channel Convolution questions.

3 input channels (3x3 images), 3 kernel channels (2x2 kernels).
Valid conv each pair (2x2 result), sum to single 2x2 feature map.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution showing each channel
"""
import numpy as np
from scipy.signal import convolve2d
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'channel1': 3,  # channel 1 convolution
    'channel2': 3,  # channel 2 convolution
    'channel3': 2,  # channel 3 convolution
    'sum': 2,       # final sum
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    channels = [rng.integers(0, 6, size=(3, 3)).tolist() for _ in range(3)]
    kernels = [rng.integers(-2, 3, size=(2, 2)).tolist() for _ in range(3)]
    return {'channels': channels, 'kernels': kernels}


def compute(params):
    channels = [np.array(c) for c in params['channels']]
    kernels = [np.array(k) for k in params['kernels']]
    ch_outputs = []
    for ch, k in zip(channels, kernels):
        out = convolve2d(ch, k, mode='valid')
        ch_outputs.append(out.tolist())
    total = sum(np.array(o) for o in ch_outputs)
    return {'ch_outputs': ch_outputs, 'output': total.tolist()}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    channels = [np.array(ch) for ch in p['channels']]
    kernels = [np.array(k) for k in p['kernels']]
    ch_outputs = [np.array(o) for o in c['ch_outputs']]
    output = np.array(c['output'])

    lines = [
        f'## Question{num_str}: Multi-channel Convolution',
        '',
        'A convolutional layer has **3 input channels**, each $3 \\times 3$:',
        '',
    ]
    for i, ch in enumerate(channels):
        lines.append(f'$$C_{{{i+1}}} = {pmatrix(ch)}$$')
        lines.append('')

    lines += [
        'The corresponding $2 \\times 2$ kernels are:',
        '',
    ]
    for i, k in enumerate(kernels):
        lines.append(f'$$K_{{{i+1}}} = {pmatrix(k)}$$')
        lines.append('')

    lines += [
        'Convolve each channel with its kernel (valid, no padding) to get a $2 \\times 2$ result,'
        ' then **sum** all channel results to produce the output feature map.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Output feature map:** `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** $O = {pmatrix(output)}$',
            '',
            '> **Hint:** Convolve each $3 \\times 3$ channel with its $2 \\times 2$ kernel (valid conv).'
            ' Sum the three $2 \\times 2$ results to get the output feature map.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        channel_mark_keys = ['channel1', 'channel2', 'channel3']
        for i, (k, o) in enumerate(zip(kernels, ch_outputs)):
            ch = channels[i]
            lines += [
                f'**Channel {i+1}:** $C_{{{i+1}}} * K_{{{i+1}}}$',
                '',
                f'$$C_{{{i+1}}} * K_{{{i+1}}} = {pmatrix(o)}$$',
            ]
            if marks:
                lines += ['', _mark(MARKS[channel_mark_keys[i]])]
            lines += ['']
        lines += [
            '**Sum all channels:**',
            '',
            '$$O = ' + ' + '.join(f'(C_{i+1}*K_{i+1})' for i in range(3)) + f' = {pmatrix(output)}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['sum'])]

    return '\n'.join(lines)
