"""SIFT orientation histogram question generator.

Given gradient magnitudes and directions for a 4×4 image patch (16 pixels),
build an 8-bin orientation histogram and identify the dominant orientation bin.

Bins are 45° wide covering 0°–360°:
  Bin 0: [0°, 45°), Bin 1: [45°, 90°), ..., Bin 7: [315°, 360°)

Bin index = floor(direction / 45), with direction in [0, 360).

Each pixel contributes its magnitude (unweighted) to the bin corresponding to
its direction.  The dominant orientation is the bin with the highest total.

Modes:
    'blank'      – question + blank answer table
    'answer'     – question + filled histogram table + hint
    'full'       – question + step-by-step derivation
    'full_marks' – question + derivation with mark annotations
"""
import numpy as np

MARKS = {
    'histogram': 7,
    'dominant':  3,
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def _fmt_matrix(arr, label=''):
    """Format a 2-D integer array as a LaTeX bmatrix block."""
    rows = [' & '.join(str(v) for v in row) for row in arr]
    inner = ' \\\\\n    '.join(rows)
    mat = f'\\begin{{bmatrix}}\n    {inner}\n\\end{{bmatrix}}'
    if label:
        return f'$${label} = {mat}$$'
    return f'$${mat}$$'


def _bin_range(b):
    """Return the angle-range string for bin index b."""
    lo = b * 45
    hi = (b + 1) * 45
    return f'[{lo}°, {hi}°)'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    while True:
        mags = rng.integers(1, 21, size=(4, 4))
        dirs = rng.integers(0, 360, size=(4, 4))
        # Build histogram
        hist = np.zeros(8, dtype=int)
        for i in range(4):
            for j in range(4):
                hist[dirs[i, j] // 45] += mags[i, j]
        sorted_vals = np.sort(hist)[::-1]
        # Require the dominant bin to be at least 1.4× the second-highest
        if sorted_vals[0] >= 1.4 * sorted_vals[1]:
            break
    return {
        'magnitudes': mags.tolist(),
        'directions': dirs.tolist(),
    }


def compute(params):
    mags = params['magnitudes']
    dirs = params['directions']
    hist = [0] * 8
    for i in range(4):
        for j in range(4):
            b = dirs[i][j] // 45
            hist[b] += mags[i][j]
    dominant_bin = int(np.argmax(hist))
    return {'histogram': hist, 'dominant_bin': dominant_bin}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    mags = p['magnitudes']
    dirs = p['directions']
    hist = c['histogram']
    dom  = c['dominant_bin']

    lines = [
        f'## Question{num_str}: SIFT Orientation Histogram',
        '',
        'The following $4 \\times 4$ image patch has gradient magnitudes'
        ' $\\mathbf{M}$ and gradient directions $\\mathbf{D}$ (in degrees):',
        '',
        _fmt_matrix(mags, label='\\mathbf{M}'),
        '',
        _fmt_matrix(dirs, label='\\mathbf{D}'),
        '',
        'Bins are 45° wide:'
        ' Bin 0 = [0°, 45°),'
        ' Bin 1 = [45°, 90°),'
        ' Bin 2 = [90°, 135°),'
        ' Bin 3 = [135°, 180°),'
        ' Bin 4 = [180°, 225°),'
        ' Bin 5 = [225°, 270°),'
        ' Bin 6 = [270°, 315°),'
        ' Bin 7 = [315°, 360°).',
        '',
        'For each pixel, compute'
        ' $\\text{bin} = \\lfloor \\text{direction} / 45 \\rfloor$'
        ' and accumulate its magnitude into that bin.',
        '',
        '**(a)** Build the 8-bin orientation histogram (total magnitude per bin).',
        '',
        '**(b)** Identify the dominant orientation bin.',
        '',
    ]

    # ------------------------------------------------------------------ blank
    if mode == 'blank':
        lines += [
            '| Bin | Range | Total magnitude |',
            '|-----|-------|----------------|',
        ]
        for b in range(8):
            lines.append(f'| {b} | {_bin_range(b)} |  |')
        lines += ['', 'Dominant orientation: `___`']

    # ----------------------------------------------------------------- answer
    elif mode == 'answer':
        lines += [
            '| Bin | Range | Total magnitude |',
            '|-----|-------|----------------|',
        ]
        for b in range(8):
            lines.append(f'| {b} | {_bin_range(b)} | {hist[b]} |')
        lines += [
            '',
            f'**Dominant orientation: Bin {dom} ({_bin_range(dom)})'
            f' with total = {hist[dom]}**',
            '',
            '> **Hint:** For each pixel, compute'
            ' $\\text{bin} = \\lfloor \\text{direction} / 45 \\rfloor$'
            ' and add its magnitude to that bin.'
            ' The dominant orientation is the bin with the highest total.',
        ]

    # ----------------------------------------------------- full / full_marks
    else:
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**Step 1 – Bin assignment for each pixel:**',
            '',
            '| Row | Col | Magnitude | Direction | Bin |',
            '|-----|-----|-----------|-----------|-----|',
        ]
        for i in range(4):
            for j in range(4):
                b = dirs[i][j] // 45
                lines.append(
                    f'| {i} | {j} | {mags[i][j]} | {dirs[i][j]}° | {b} |'
                )
        if marks:
            lines += ['', _mark(MARKS['histogram'])]

        lines += [
            '',
            '**Step 2 – Histogram summary:**',
            '',
            '| Bin | Range | Total magnitude |',
            '|-----|-------|----------------|',
        ]
        for b in range(8):
            lines.append(f'| {b} | {_bin_range(b)} | {hist[b]} |')

        lines += [
            '',
            '**Step 3 – Dominant orientation:**',
            '',
            f'Bin {dom} ({_bin_range(dom)}) has the highest total:'
            f' **{hist[dom]}**',
        ]
        if marks:
            lines += ['', _mark(MARKS['dominant'])]

    return '\n'.join(lines)
