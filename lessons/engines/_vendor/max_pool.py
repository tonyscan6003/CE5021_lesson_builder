"""Generator for Max and Average Pooling questions (two parts).

Random 4x4 integer matrix. Pool size 2x2, stride 2. Output 2x2.
Part (a): Max pooling.
Part (b): Average pooling (rounded to 2 d.p.).

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts
"""
import numpy as np
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'max_pool': 5,  # max pooling output
    'avg_pool': 5,  # average pooling output
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 10, size=(4, 4)).tolist()
    return {'img': img}


def compute(params):
    img = np.array(params['img'])
    max_out = np.zeros((2, 2), dtype=int)
    avg_out = np.zeros((2, 2))
    patches = {}
    for i in range(2):
        for j in range(2):
            patch = img[i*2:(i+1)*2, j*2:(j+1)*2]
            patches[(i, j)] = patch.tolist()
            max_out[i, j] = int(patch.max())
            avg_out[i, j] = round(float(patch.mean()), 2)
    return {
        'max_out': max_out.tolist(),
        'avg_out': avg_out.tolist(),
        'patches': patches,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img = np.array(p['img'])
    max_out = np.array(c['max_out'])
    avg_out = np.array(c['avg_out'])

    img_str = pmatrix(img)
    max_str = pmatrix(max_out)

    # avg matrix with floats
    def _avg_mat(mat):
        rows = [' & '.join(f'{v:.2f}' for v in row) for row in mat]
        return r'\begin{pmatrix}' + r' \\ '.join(rows) + r'\end{pmatrix}'

    avg_str = _avg_mat(c['avg_out'])

    lines = [
        f'## Question{num_str}: Max and Average Pooling',
        '',
        'Apply pooling (pool size $2 \\times 2$, stride $2$, no padding) to the following $4 \\times 4$ image:',
        '',
        f'$$I = {img_str}$$',
        '',
        '**Part (a):** Compute the **max pooling** output. Give the $2 \\times 2$ result.',
        '',
        '**Part (b):** Compute the **average pooling** output (values rounded to 2 d.p.). Give the $2 \\times 2$ result.',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** Max pool output = `_______________`',
            '',
            '**Part (b):** Avg pool output = `_______________`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** Max pool $= {max_str}$',
            '',
            f'**Part (b):** Avg pool $= {avg_str}$',
            '',
            '> **Hint:** Divide the image into four non-overlapping $2 \\times 2$ patches.'
            ' For max pooling take the maximum of each patch; for average pooling take the mean.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            'The $4 \\times 4$ image is divided into four $2 \\times 2$ patches:',
            '',
        ]
        patch_labels = [('top-left', 0, 0), ('top-right', 0, 1),
                        ('bottom-left', 1, 0), ('bottom-right', 1, 1)]
        for label, i, j in patch_labels:
            patch = np.array(c['patches'][(i, j)])
            mx = c['max_out'][i][j]
            av = c['avg_out'][i][j]
            lines += [
                f'**Patch ({label}):**',
                '',
                f'$$P_{{{i+1}{j+1}}} = {pmatrix(patch)}$$',
                '',
                f'- Max $= {mx}$',
                f'- Avg $= ({" + ".join(str(v) for row in patch.tolist() for v in row)}) / 4 = {av}$',
                '',
            ]
        lines += [
            '**Part (a) - Max pooling output:**',
            '',
            f'$$M = {max_str}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['max_pool'])]
        lines += [
            '',
            '**Part (b) - Average pooling output:**',
            '',
            f'$$A = {avg_str}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['avg_pool'])]

    return '\n'.join(lines)
