"""Non-maximum suppression (NMS) step of Canny edge detection.

Given a 5×5 gradient magnitude grid |G| and a quantised direction grid D
(values: 0, 45, 90, 135 degrees), apply NMS to produce a thinned edge map.

Each interior pixel (rows 1-3, cols 1-3) is kept (1) if its magnitude is ≥
both neighbours along its gradient direction; boundary pixels are always 0.

Direction → neighbour offsets (dr, dc):
  0°   → (0, -1) and (0, +1)   left / right
  45°  → (-1, +1) and (+1, -1) top-right / bottom-left
  90°  → (-1, 0) and (+1, 0)   top / bottom
  135° → (-1, -1) and (+1, +1) top-left / bottom-right

Modes:
    'blank'      – question only, empty output grid
    'answer'     – question + correct result + blockquote hint
    'full'       – question + full per-pixel working (no mark annotations)
    'full_marks' – same as full, with mark annotations
"""
import numpy as np

MARKS = {
    'rule':   3,   # state the NMS rule and direction→neighbour mapping
    'result': 7,   # correct NMS output matrix
}

_SIZE = 5

# Direction → (neighbour1_offset, neighbour2_offset)  as (dr, dc)
_DIR_NEIGHBOURS = {
    0:   ((0, -1),  (0, +1)),
    45:  ((-1, +1), (+1, -1)),
    90:  ((-1, 0),  (+1, 0)),
    135: ((-1, -1), (+1, +1)),
}

_DIR_LABELS = {
    0:   'left / right',
    45:  'top-right / bottom-left',
    90:  'top / bottom',
    135: 'top-left / bottom-right',
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def _fmt_matrix(arr, label=''):
    """Format a 2D integer array as a LaTeX bmatrix display equation."""
    rows = [' & '.join(str(int(v)) for v in row) for row in arr]
    inner = r' \\ '.join(rows)
    mat = r'\begin{bmatrix}' + inner + r'\end{bmatrix}'
    if label:
        return f'$${label} = {mat}$$'
    return f'$${mat}$$'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    directions = [0, 45, 90, 135]

    for _ in range(200):
        mags = rng.integers(10, 200, size=(_SIZE, _SIZE))
        dirs = rng.choice(directions, size=(_SIZE, _SIZE))

        # Count kept interior pixels
        kept = 0
        for r in range(1, _SIZE - 1):
            for c in range(1, _SIZE - 1):
                d = int(dirs[r, c])
                (dr1, dc1), (dr2, dc2) = _DIR_NEIGHBOURS[d]
                n1 = int(mags[r + dr1, c + dc1])
                n2 = int(mags[r + dr2, c + dc2])
                m = int(mags[r, c])
                if m >= n1 and m >= n2:
                    kept += 1

        if 2 <= kept <= 7:
            return {'mags': mags.tolist(), 'dirs': dirs.tolist()}

    # Fallback: return last generated (very unlikely to be needed)
    return {'mags': mags.tolist(), 'dirs': dirs.tolist()}


def compute(params):
    mags = np.array(params['mags'])
    dirs = np.array(params['dirs'])
    result = np.zeros((_SIZE, _SIZE), dtype=int)

    for r in range(1, _SIZE - 1):
        for c in range(1, _SIZE - 1):
            d = int(dirs[r, c])
            (dr1, dc1), (dr2, dc2) = _DIR_NEIGHBOURS[d]
            n1 = int(mags[r + dr1, c + dc1])
            n2 = int(mags[r + dr2, c + dc2])
            m = int(mags[r, c])
            if m >= n1 and m >= n2:
                result[r, c] = 1

    return {'result': result.tolist()}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    mags = np.array(p['mags'])
    dirs = np.array(p['dirs'])
    result = np.array(c['result'])

    # Direction mapping sentence
    dir_map_lines = [
        'The quantised gradient direction determines which two neighbours to compare:',
        '',
        '- **0°** → left / right: $(r,\\, c-1)$ and $(r,\\, c+1)$',
        '- **45°** → top-right / bottom-left: $(r-1,\\, c+1)$ and $(r+1,\\, c-1)$',
        '- **90°** → top / bottom: $(r-1,\\, c)$ and $(r+1,\\, c)$',
        '- **135°** → top-left / bottom-right: $(r-1,\\, c-1)$ and $(r+1,\\, c+1)$',
    ]

    lines = [
        f'## Question{num_str}: Non-Maximum Suppression (Canny Edge Detection)',
        '',
        'Apply non-maximum suppression (NMS) to the following $5\\times 5$ gradient'
        ' magnitude grid $|G|$ and quantised direction grid $D$.',
        '',
        _fmt_matrix(mags, r'|G|'),
        '',
        _fmt_matrix(dirs, 'D'),
        '',
    ]
    lines += dir_map_lines
    lines += [
        '',
        'For each **interior pixel** (rows 1–3, cols 1–3, using 0-based indexing):'
        ' keep the pixel ($E=1$) if $|G|(r,c) \\geq$ both neighbours along direction $D$;'
        ' otherwise suppress it ($E=0$). All boundary pixels are $E=0$.',
        '',
        'Produce the $5\\times 5$ NMS output matrix $E$.',
        '',
    ]

    if mode == 'blank':
        blank = np.zeros((_SIZE, _SIZE), dtype=int)
        lines.append(_fmt_matrix(blank, 'E'))

    elif mode == 'answer':
        lines += [
            _fmt_matrix(result, 'E'),
            '',
            '> **Hint:** For each interior pixel, look up its direction to find the two'
            ' neighbour positions. Keep the pixel if $|G|(r,c) \\geq$ both neighbours;'
            ' set to 0 otherwise. All boundary pixels are 0.',
        ]

    else:  # full / full_marks
        use_marks = marks  # mode is 'full_marks' when marks=True
        lines += ['---', '', '### Solution', '']
        if use_marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**NMS rule:** A pixel at $(r,c)$ is kept ($E=1$) if and only if'
            ' $|G|(r,c) \\geq$ the magnitude of both neighbours in the direction'
            ' of $D(r,c)$; otherwise it is suppressed ($E=0$).'
            ' All boundary pixels (row or column 0 or 4) are set to 0.',
            '',
        ]
        lines += dir_map_lines
        if use_marks:
            lines += ['', _mark(MARKS['rule'])]

        lines += ['', '**Per-pixel NMS (interior pixels only):**', '']

        for r in range(1, _SIZE - 1):
            for c in range(1, _SIZE - 1):
                d = int(dirs[r, c])
                (dr1, dc1), (dr2, dc2) = _DIR_NEIGHBOURS[d]
                m = int(mags[r, c])
                n1 = int(mags[r + dr1, c + dc1])
                n2 = int(mags[r + dr2, c + dc2])
                kept = result[r, c] == 1
                label = _DIR_LABELS[d]
                verdict = '**kept**' if kept else '**suppressed**'
                r1_label = f'({r + dr1},{c + dc1})'
                r2_label = f'({r + dr2},{c + dc2})'
                lines.append(
                    f'- $({r},{c})$: $|G|={m}$, $D={d}°$ ({label})'
                    f' → compare {r1_label}$={n1}$ / {r2_label}$={n2}$'
                    f' → {verdict}'
                )

        lines += ['', _fmt_matrix(result, 'E')]
        if use_marks:
            lines += ['', _mark(MARKS['result'])]

    return '\n'.join(lines)
