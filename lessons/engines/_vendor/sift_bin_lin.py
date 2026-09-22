"""Generator for SIFT bilinear distribution questions.

Inverse of bilinear interpolation: a pixel at interior position (xi, yj) has
known gradient magnitude M.  Distribute M to the four corner bins of the
2×2 cell using bilinear (area-weighted) allocation — exactly as SIFT does
when accumulating gradient magnitudes into its spatial histogram bins.

Each corner c receives:

    M_c = M * A_c / 4

where A_c is the area of the sub-rectangle diagonally opposite corner c and
the total cell area is always 4 (= 2×2).  The four contributions sum to M.

Uses the IMAGE coordinate system: origin top-left, y increases downward.
  (0,0) = top-left     (2,0) = top-right
  (0,2) = bottom-left  (2,2) = bottom-right

Modes:
    'blank'  – question + grid plot (corner contributions unknown)
    'answer' – question + shaded grid + answers + hint
    'full'   – question + shaded grid + step-by-step solution
"""
import io
import os
import base64
import numpy as np
from ._exam_boxes import cell_box

MARKS = {
    'areas':         4,   # compute the four sub-rectangle areas (1 mark each)
    'contributions': 6,   # compute the four magnitude contributions
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    xi = round(float(rng.uniform(0.2, 1.8)), 1)
    yj = round(float(rng.uniform(0.2, 1.8)), 1)
    M = int(rng.integers(8, 51))
    return {'xi': xi, 'yj': yj, 'M': M}


def compute(params):
    xi, yj = params['xi'], params['yj']
    M = params['M']

    A00 = round((2 - xi) * (2 - yj), 4)   # (0,0) top-left    → opposite bottom-right
    A20 = round(xi * (2 - yj), 4)          # (2,0) top-right   → opposite bottom-left
    A02 = round((2 - xi) * yj, 4)          # (0,2) bottom-left → opposite top-right
    A22 = round(xi * yj, 4)                # (2,2) bottom-right → opposite top-left
    total = A00 + A20 + A02 + A22          # == 4.0

    return {
        'areas': {'(0,0)': A00, '(2,0)': A20, '(0,2)': A02, '(2,2)': A22},
        'total_area': total,
        'contributions': {
            '(0,0)': round(M * A00 / total, 4),
            '(2,0)': round(M * A20 / total, 4),
            '(0,2)': round(M * A02 / total, 4),
            '(2,2)': round(M * A22 / total, 4),
        },
    }


_CORNER_POS = {
    '(0,0)': 'top-left',
    '(2,0)': 'top-right',
    '(0,2)': 'bottom-left',
    '(2,2)': 'bottom-right',
}

_CORNER_OPP_DESC = {
    '(0,0)': 'bottom-right',
    '(2,0)': 'bottom-left',
    '(0,2)': 'top-right',
    '(2,2)': 'top-left',
}


def _make_plot(params, computed, shade, show_contribs=False, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    xi, yj = params['xi'], params['yj']
    M = params['M']
    corners = [(0, 0), (2, 0), (0, 2), (2, 2)]
    colors = ['#e41a1c', '#ff7f00', '#377eb8', '#4daf4a']

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()
    ax.set_aspect('equal')

    if shade:
        # Shaded area = sub-rectangle diagonally opposite each corner
        # (same geometry as bi_lin, same color assignment)
        regions = [
            (xi, yj, 2 - xi, 2 - yj),   # (0,0) → shaded region: bottom-right
            (0,  yj, xi,     2 - yj),    # (2,0) → shaded region: bottom-left
            (xi, 0,  2 - xi, yj),        # (0,2) → shaded region: top-right
            (0,  0,  xi,     yj),        # (2,2) → shaded region: top-left
        ]
        for (rx, ry, rw, rh), col in zip(regions, colors):
            ax.add_patch(patches.Rectangle((rx, ry), rw, rh,
                                           linewidth=0, facecolor=col, alpha=0.30))

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(patches.Rectangle((0, 0), 2, 2,
                                   linewidth=1.5, edgecolor='k', facecolor='none'))
    ax.axvline(xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(yj, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    offsets_pts = {
        (0, 0): (8,   20),
        (2, 0): (-8,  20),
        (0, 2): (8,  -20),
        (2, 2): (-8, -20),
    }
    ha_map = {(0, 0): 'left',  (2, 0): 'right',
              (0, 2): 'left',  (2, 2): 'right'}
    va_map = {(0, 0): 'bottom', (2, 0): 'bottom',
              (0, 2): 'top',    (2, 2): 'top'}

    contribs = computed['contributions'] if show_contribs else {}

    for (cx, cy), col in zip(corners, colors):
        ax.plot(cx, cy, 'o', color=col, markersize=11, zorder=5)
        dx_pts, dy_pts = offsets_pts[(cx, cy)]
        key = f'({cx},{cy})'
        if show_contribs and contribs:
            lbl = f'$({cx},{cy})$\n$M_c={contribs[key]}$'
        else:
            lbl = f'$({cx},{cy})$\n$M_c=?$'
        ax.annotate(lbl, (cx, cy),
                    xytext=(dx_pts, dy_pts), textcoords='offset points',
                    color=col, fontsize=9, linespacing=1.4,
                    ha=ha_map[(cx, cy)], va=va_map[(cx, cy)])

    # Interior pixel: known magnitude M (this is what we're distributing)
    ax.plot(xi, yj, 'k^', markersize=9, zorder=6)
    ax.annotate(f'$({xi},\\,{yj})$\n$M={M}$', (xi, yj),
                xytext=(6, -6), textcoords='offset points',
                fontsize=9, va='top', linespacing=1.4)

    ax.set_xticks([0, xi, 2])
    ax.set_yticks([0, yj, 2])
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y\\;(\\downarrow)$', fontsize=11)
    ax.set_title('SIFT bilinear distribution grid\n'
                 '(image coords: origin top-left, $y$ increases downward)',
                 fontsize=9)
    ax.grid(False)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format='png', dpi=90)
        plt.close(fig)
        return None

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def render(params, computed, mode='full', question_num=None, marks=False,
           out_dir=None, img_stem=None):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    xi, yj, M = p['xi'], p['yj'], p['M']

    shade = mode in ('answer', 'full')
    show_contribs = mode == 'full'
    if out_dir and img_stem:
        img_filename = f'{img_stem}.png'
        _make_plot(p, c, shade=shade, show_contribs=show_contribs,
                   save_path=os.path.join(out_dir, img_filename))
        img_md = f'![SIFT bilinear distribution grid]({img_filename})'
    else:
        b64 = _make_plot(p, c, shade=shade, show_contribs=show_contribs)
        img_md = f'![SIFT bilinear distribution grid](data:image/png;base64,{b64})'

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: SIFT Bilinear Distribution'),
        '',
        '> **Note:** Image coordinate system is used — the origin is at the'
        ' **top-left**, and $y$ increases **downward**.',
        '',
        r'In the SIFT descriptor, gradient magnitudes at pixel positions are'
        r' distributed to nearby spatial histogram bins using bilinear weighting.',
        r' A pixel at interior position $(x_i, y_j)$'
        f' $= ({xi},\\, {yj})$ has gradient magnitude $M = {M}$.',
        '',
        r'Determine the magnitude contribution $M_c$ allocated to each corner'
        r' of the $2\times 2$ bin grid (see plot).',
        '',
        img_md,
        '',
        r'Each corner $c$ receives a share of $M$ proportional to the area'
        r' $A_c$ of the sub-rectangle **diagonally opposite** corner $c$:',
        '',
        r'$$M_c = M \cdot \frac{A_c}{\displaystyle\sum_c A_c}, \qquad \sum_c A_c = 4$$',
        '',
    ]

    if mode == 'blank':
        lines += [
            '| Corner | Position | $M_c$ |',
            '|--------|----------|-------|',
            '| $(0,0)$ | top-left     | `___` |',
            '| $(2,0)$ | top-right    | `___` |',
            '| $(0,2)$ | bottom-left  | `___` |',
            '| $(2,2)$ | bottom-right | `___` |',
            '',
        ]

    elif mode == 'exam':
        cbox = cell_box()
        lines += [
            '**Part (a):** Sub-rectangle areas.',
            '',
            '| Corner | Position | $A_c$ |',
            '|--------|----------|-------|',
            f'| $(0,0)$ | top-left     | {cbox} |',
            f'| $(2,0)$ | top-right    | {cbox} |',
            f'| $(0,2)$ | bottom-left  | {cbox} |',
            f'| $(2,2)$ | bottom-right | {cbox} |',
            '',
            '**Part (b):** Magnitude contributions.',
            '',
            '| Corner | Position | $M_c$ |',
            '|--------|----------|-------|',
            f'| $(0,0)$ | top-left     | {cbox} |',
            f'| $(2,0)$ | top-right    | {cbox} |',
            f'| $(0,2)$ | bottom-left  | {cbox} |',
            f'| $(2,2)$ | bottom-right | {cbox} |',
            '',
        ]

    elif mode == 'exam_marks':
        areas = c['areas']
        contribs = c['contributions']
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**Part (a) - Sub-rectangle areas.**',
            '',
            r'The pixel at $(x_i, y_j)$ divides the $2\times 2$ cell into four'
            r' rectangles. The area diagonally opposite each corner is:',
            '',
        ]
        for corner, area in areas.items():
            pos = _CORNER_POS[corner]
            opp = _CORNER_OPP_DESC[corner]
            cx_str, cy_str = corner[1:-1].split(',')
            if corner == '(0,0)':
                a_str = f'(2 - {xi})(2 - {yj})'
            elif corner == '(2,0)':
                a_str = f'{xi}(2 - {yj})'
            elif corner == '(0,2)':
                a_str = f'(2 - {xi}) \\cdot {yj}'
            else:
                a_str = f'{xi} \\cdot {yj}'
            lines.append(
                f'$A_{{({cx_str},{cy_str})}}$ ({pos}, opposite = {opp}): '
                f'$A = {a_str} = {area}$'
            )
        lines += [
            '',
            f'Check: ${" + ".join(str(a) for a in areas.values())}'
            f' = {c["total_area"]}$ ✓',
            '',
            _mark(MARKS['areas']),
        ]

        lines += [
            '',
            '**Part (b) - Distribute magnitude $M$ to each corner.**',
            '',
            f'$M = {M}$, total area $= 4$:',
            '',
        ]
        for corner, contrib in contribs.items():
            area = areas[corner]
            cx_str, cy_str = corner[1:-1].split(',')
            lines.append(
                f'$M_{{({cx_str},{cy_str})}} = {M} \\times {area} / 4 = {contrib}$'
            )
        check_sum = round(sum(contribs.values()), 4)
        lines += [
            '',
            f'Check: ${" + ".join(str(v) for v in contribs.values())}'
            f' = {check_sum} = M$ ✓',
            '',
            _mark(MARKS['contributions']),
        ]

    elif mode == 'answer':
        contribs = c['contributions']
        lines += [
            '| Corner | Position | $M_c$ |',
            '|--------|----------|-------|',
            f'| $(0,0)$ | top-left     | ${contribs["(0,0)"]}$ |',
            f'| $(2,0)$ | top-right    | ${contribs["(2,0)"]}$ |',
            f'| $(0,2)$ | bottom-left  | ${contribs["(0,2)"]}$ |',
            f'| $(2,2)$ | bottom-right | ${contribs["(2,2)"]}$ |',
            '',
            r'> **Hint:** For each corner $c$, compute the area $A_c$ of the'
            r' sub-rectangle diagonally opposite: $A_c = \Delta x \cdot \Delta y$'
            r' where $\Delta x, \Delta y$ are the distances from the pixel to the'
            r' opposite corner. Then $M_c = M \cdot A_c / 4$.'
            r' Check: contributions must sum to $M$.',
        ]

    else:  # full / full_marks
        areas = c['areas']
        contribs = c['contributions']
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        lines += [
            '**Step 1 – Compute the four sub-rectangle areas.**',
            '',
            r'The pixel at $(x_i, y_j)$ divides the $2\times 2$ cell into four'
            r' rectangles. The area diagonally opposite each corner is:',
            '',
        ]
        for corner, area in areas.items():
            pos = _CORNER_POS[corner]
            opp = _CORNER_OPP_DESC[corner]
            cx_str, cy_str = corner[1:-1].split(',')
            if corner == '(0,0)':
                a_str = f'(2 - {xi})(2 - {yj})'
            elif corner == '(2,0)':
                a_str = f'{xi}(2 - {yj})'
            elif corner == '(0,2)':
                a_str = f'(2 - {xi}) \\cdot {yj}'
            else:
                a_str = f'{xi} \\cdot {yj}'
            lines.append(
                f'$A_{{({cx_str},{cy_str})}}$ ({pos}, opposite = {opp}): '
                f'$A = {a_str} = {area}$'
            )
        lines += [
            '',
            f'Check: ${" + ".join(str(a) for a in areas.values())}'
            f' = {c["total_area"]}$ ✓',
        ]
        if marks:
            lines += ['', _mark(MARKS['areas'])]

        lines += [
            '',
            '**Step 2 – Distribute magnitude $M$ to each corner.**',
            '',
            f'$M = {M}$, total area $= 4$:',
            '',
        ]
        for corner, contrib in contribs.items():
            area = areas[corner]
            cx_str, cy_str = corner[1:-1].split(',')
            lines.append(
                f'$M_{{({cx_str},{cy_str})}} = {M} \\times {area} / 4 = {contrib}$'
            )
        check_sum = round(sum(contribs.values()), 4)
        lines += [
            '',
            f'Check: ${" + ".join(str(v) for v in contribs.values())}'
            f' = {check_sum} = M$ ✓',
        ]
        if marks:
            lines += ['', _mark(MARKS['contributions'])]

    return '\n'.join(lines)
