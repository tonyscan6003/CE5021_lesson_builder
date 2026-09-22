"""Haar-like feature computation using an integral image.

Given a 6x6 image and its pre-computed integral image, compute the value of
a two-rectangle Haar-like feature.

Feature types:
  'h' -- horizontal edge: two vertically-stacked rectangles.
         Top rect is "black" (negative), bottom rect is "white" (positive).
         Feature = sum(bottom) - sum(top).
  'v' -- vertical edge: two horizontally-adjacent rectangles.
         Left rect is "black" (negative), right rect is "white" (positive).
         Feature = sum(right) - sum(left).

Integral image convention (0-indexed, inclusive bounds):
  II[i][j] = sum of all pixels in rows 0..i, cols 0..j.

Four-corner formula:
  rect_sum(r1,c1,r2,c2) = II[r2][c2]
                         - (II[r1-1][c2]   if r1 > 0 else 0)
                         - (II[r2][c1-1]   if c1 > 0 else 0)
                         + (II[r1-1][c1-1] if r1 > 0 and c1 > 0 else 0)
"""
import numpy as np
from ._exam_boxes import answer_box, inline_box

MARKS = {
    'rect_sums':     6,   # correct lookup and computation of both rectangle sums
    'feature_value': 4,   # correct feature value (sign matters)
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _rect_sum_val(II, r1, c1, r2, c2):
    """Compute rectangle sum from integral image using the four-corner formula."""
    s = int(II[r2][c2])
    if r1 > 0:
        s -= int(II[r1 - 1][c2])
    if c1 > 0:
        s -= int(II[r2][c1 - 1])
    if r1 > 0 and c1 > 0:
        s += int(II[r1 - 1][c1 - 1])
    return s


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------

def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 16, size=(6, 6)).astype(int)

    ftype = str(rng.choice(['h', 'v']))

    if ftype == 'h':
        h = int(rng.choice([2, 3]))
        w = int(rng.choice([2, 3, 4]))
        # r0 + 2h - 1 <= 5  =>  r0 <= 6 - 2h
        r0_max = 6 - 2 * h
        # c0 + w - 1  <= 5  =>  c0 <= 6 - w
        c0_max = 6 - w
    else:   # 'v'
        h = int(rng.choice([2, 3, 4]))
        w = int(rng.choice([2, 3]))
        # r0 + h - 1  <= 5  =>  r0 <= 6 - h
        r0_max = 6 - h
        # c0 + 2w - 1 <= 5  =>  c0 <= 6 - 2w
        c0_max = 6 - 2 * w

    r0 = int(rng.integers(0, r0_max + 1))
    c0 = int(rng.integers(0, c0_max + 1))

    return {
        'image':        img.tolist(),
        'feature_type': ftype,
        'r0': r0,
        'c0': c0,
        'h':  h,
        'w':  w,
    }


def compute(params):
    img   = np.array(params['image'], dtype=int)
    ftype = params['feature_type']
    r0, c0, h, w = params['r0'], params['c0'], params['h'], params['w']

    # Integral image: II[i][j] = sum of pixels in rows 0..i, cols 0..j
    II = np.cumsum(np.cumsum(img, axis=0), axis=1).astype(int)

    if ftype == 'h':
        # rect1 = top  (black), rect2 = bottom (white)
        s1 = _rect_sum_val(II, r0,      c0, r0 + h - 1,     c0 + w - 1)
        s2 = _rect_sum_val(II, r0 + h,  c0, r0 + 2 * h - 1, c0 + w - 1)
    else:   # 'v'
        # rect1 = left (black), rect2 = right  (white)
        s1 = _rect_sum_val(II, r0, c0,      r0 + h - 1, c0 + w - 1)
        s2 = _rect_sum_val(II, r0, c0 + w,  r0 + h - 1, c0 + 2 * w - 1)

    fval = s2 - s1   # white rectangle - black rectangle

    return {
        'integral_image': II.tolist(),
        'sum_rect1':      s1,    # black rect (top or left)
        'sum_rect2':      s2,    # white rect (bottom or right)
        'feature_value':  fval,
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c    = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    img     = p['image']
    ftype   = p['feature_type']
    r0, c0, h, w = p['r0'], p['c0'], p['h'], p['w']
    II      = c['integral_image']
    s1, s2  = c['sum_rect1'], c['sum_rect2']
    fval    = c['feature_value']

    # -----------------------------------------------------------------------
    # Local helper: 2-D array -> LaTeX bmatrix string
    # -----------------------------------------------------------------------
    def _fmt_matrix(arr, label=''):
        arr = np.asarray(arr)
        rows_tex = [' & '.join(str(int(v)) for v in row) for row in arr]
        inner = r' \\ '.join(rows_tex)
        mat = r'\begin{bmatrix}' + inner + r'\end{bmatrix}'
        return f'{label} = {mat}' if label else mat

    # -----------------------------------------------------------------------
    # Derive rectangle coordinates and labels from feature type
    # -----------------------------------------------------------------------
    if ftype == 'h':
        r1a, c1a, r2a, c2a = r0,      c0, r0 + h - 1,     c0 + w - 1
        r1b, c1b, r2b, c2b = r0 + h,  c0, r0 + 2 * h - 1, c0 + w - 1
        rect1_name   = 'top ("black")'
        rect2_name   = 'bottom ("white")'
        feat_name    = 'horizontal-edge'
        feat_formula = 'Feature $=$ sum(bottom rectangle) $-$ sum(top rectangle)'
    else:
        r1a, c1a, r2a, c2a = r0, c0,     r0 + h - 1, c0 + w - 1
        r1b, c1b, r2b, c2b = r0, c0 + w, r0 + h - 1, c0 + 2 * w - 1
        rect1_name   = 'left ("black")'
        rect2_name   = 'right ("white")'
        feat_name    = 'vertical-edge'
        feat_formula = 'Feature $=$ sum(right rectangle) $-$ sum(left rectangle)'

    # -----------------------------------------------------------------------
    # Header shared across all modes
    # -----------------------------------------------------------------------
    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Haar-Like Feature Computation'),
        '',
        'The image $I$ and its pre-computed integral image $II$ are given below'
        ' (0-indexed rows and columns).',
        '',
        f'$$I = {_fmt_matrix(img)}$$',
        '',
        f'$$II = {_fmt_matrix(II)}$$',
        '',
        f'A **{feat_name}** Haar feature is defined with two rectangles,'
        f' each of height $h = {h}$, width $w = {w}$,'
        f' with top-left corner $(r_0, c_0) = ({r0}, {c0})$ (0-indexed).',
        '',
        f'{feat_formula}.',
        '',
        '**Four-corner formula** for a rectangle with inclusive bounds'
        ' $(r_1, c_1)$ to $(r_2, c_2)$:',
        '',
        r'$$\text{rect\_sum}(r_1,c_1,r_2,c_2) = II_{r_2,c_2}'
        r' - II_{r_1-1,c_2} - II_{r_2,c_1-1} + II_{r_1-1,c_1-1}$$',
        '',
        r'(Any term whose index falls below 0 is taken as $0$.)',
        '',
        'Compute the Haar feature value.',
        '',
    ]

    # ---- Helper: four-corner lookup lines for one rectangle ----
    def _lookup_lines(r1, c1, r2, c2, II_arr, rect_label):
        A = int(II_arr[r2][c2])
        B = int(II_arr[r1 - 1][c2]) if r1 > 0 else 0
        C = int(II_arr[r2][c1 - 1]) if c1 > 0 else 0
        D = int(II_arr[r1 - 1][c1 - 1]) if (r1 > 0 and c1 > 0) else 0
        total = A - B - C + D

        out = [
            f'**{rect_label}:** rows $[{r1},\\,{r2}]$, cols $[{c1},\\,{c2}]$',
            '',
            f'- $II_{{{r2},{c2}}} = {A}$',
        ]
        if r1 > 0:
            out.append(f'- $II_{{{r1 - 1},{c2}}} = {B}$')
        else:
            out.append(f'- $r_1 = 0$: top boundary term $= 0$')
        if c1 > 0:
            out.append(f'- $II_{{{r2},{c1 - 1}}} = {C}$')
        else:
            out.append(f'- $c_1 = 0$: left boundary term $= 0$')
        if r1 > 0 and c1 > 0:
            out.append(f'- $II_{{{r1 - 1},{c1 - 1}}} = {D}$')
        else:
            out.append(f'- Corner boundary term $= 0$')
        out += [
            '',
            f'$$\\text{{sum}} = {A} - {B} - {C} + {D} = {total}$$',
        ]
        return out

    # -----------------------------------------------------------------------
    # Mode-specific content
    # -----------------------------------------------------------------------
    if mode == 'blank':
        lines.append('**Feature value =** `___`')

    elif mode == 'exam':
        lines += [
            '**Part (a):** Rectangle sums — '
            f'{rect1_name.split(" (")[0]} $=$ {inline_box()} '
            f'{rect2_name.split(" (")[0]} $=$ {inline_box()}',
            '',
            '**Part (b):** Feature value $=$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Rectangle sums.**',
            '',
        ]
        lines += _lookup_lines(r1a, c1a, r2a, c2a, II, f'Rectangle 1 — {rect1_name}')
        lines += ['']
        lines += _lookup_lines(r1b, c1b, r2b, c2b, II, f'Rectangle 2 — {rect2_name}')
        lines += ['', _mark(MARKS['rect_sums'])]
        lines += [
            '',
            '**Part (b) - Feature value.**',
            '',
            f'{feat_formula}:',
            '',
            f'$$\\text{{Feature}} = {s2} - {s1} = {fval}$$',
            '',
            _mark(MARKS['feature_value']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Feature value =** ${fval}$',
            '',
            '> **Hint:** Apply the four-corner formula to each rectangle using the'
            ' pre-computed integral image. Look up the four corner values at'
            ' (bottom-right, top-right, bottom-left, top-left) of each rectangle,'
            ' substituting 0 for any index that falls below 0. Compute A - B - C + D'
            ' for each rectangle, then subtract: white minus black.',
        ]

    else:   # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**Step 1 - Compute rectangle sums.**',
            '',
        ]
        lines += _lookup_lines(r1a, c1a, r2a, c2a, II, f'Rectangle 1 — {rect1_name}')
        lines += ['']
        lines += _lookup_lines(r1b, c1b, r2b, c2b, II, f'Rectangle 2 — {rect2_name}')

        if marks:
            lines += ['', _mark(MARKS['rect_sums'])]

        lines += [
            '',
            '**Step 2 - Haar feature value.**',
            '',
            f'{feat_formula}:',
            '',
            f'$$\\text{{Feature}} = {s2} - {s1} = {fval}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['feature_value'])]

    return '\n'.join(lines)
