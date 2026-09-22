"""Generator for bilinear interpolation questions.

Four corner pixels of a 2×2 cell each have a known magnitude.  The student
must compute the bilinearly-interpolated magnitude at an interior point (xi, yj)
using the area-weighting rule:

    M_interp = sum_c( I_c * A_c ) / 4

where A_c is the area of the sub-rectangle diagonally opposite corner c, and
the total area is always 4 (= 2×2).

Uses the IMAGE coordinate system: origin top-left, y increases downward.
  (0,0) = top-left     (2,0) = top-right
  (0,2) = bottom-left  (2,2) = bottom-right

Modes:
    'blank'  – question + grid plot (pixel marked, no shading)
    'answer' – question + shaded grid plot + interpolated value + hint
    'full'   – question + shaded grid plot + step-by-step solution
"""
import io
import os
import base64
import numpy as np
from ._exam_boxes import answer_box, inline_box

# Mark allocation per step — must sum to 10
MARKS = {
    'coordinates': 2,   # sub-rectangle areas
    'weights': 4,       # weighted contributions
    'interpolated': 4,  # final interpolated value
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    xi = round(float(rng.uniform(0.2, 1.8)), 1)
    yj = round(float(rng.uniform(0.2, 1.8)), 1)
    # Integer corner magnitudes keep the arithmetic clean
    I00 = int(rng.integers(1, 21))
    I20 = int(rng.integers(1, 21))
    I02 = int(rng.integers(1, 21))
    I22 = int(rng.integers(1, 21))
    return {'xi': xi, 'yj': yj, 'I00': I00, 'I20': I20, 'I02': I02, 'I22': I22}


def compute(params):
    xi, yj = params['xi'], params['yj']
    I00, I20 = params['I00'], params['I20']
    I02, I22 = params['I02'], params['I22']

    # In image coords (y down): area diagonally opposite each corner
    # (0,0) top-left     → opposite = bottom-right = (2-xi)*(2-yj)
    # (2,0) top-right    → opposite = bottom-left  = xi*(2-yj)
    # (0,2) bottom-left  → opposite = top-right    = (2-xi)*yj
    # (2,2) bottom-right → opposite = top-left     = xi*yj
    A00 = round((2 - xi) * (2 - yj), 4)
    A20 = round(xi * (2 - yj), 4)
    A02 = round((2 - xi) * yj, 4)
    A22 = round(xi * yj, 4)
    total = A00 + A20 + A02 + A22  # == 4.0

    weighted = {
        '(0,0)': round(I00 * A00, 4),
        '(2,0)': round(I20 * A20, 4),
        '(0,2)': round(I02 * A02, 4),
        '(2,2)': round(I22 * A22, 4),
    }
    M_interp = round(sum(weighted.values()) / total, 4)

    return {
        'areas': {'(0,0)': A00, '(2,0)': A20, '(0,2)': A02, '(2,2)': A22},
        'corner_mags': {'(0,0)': I00, '(2,0)': I20, '(0,2)': I02, '(2,2)': I22},
        'weighted': weighted,
        'total_area': total,
        'M_interp': M_interp,
    }


def _make_plot(params, shade, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    xi, yj = params['xi'], params['yj']
    mags = {
        (0, 0): params['I00'],
        (2, 0): params['I20'],
        (0, 2): params['I02'],
        (2, 2): params['I22'],
    }
    colors = ['#e41a1c', '#ff7f00', '#377eb8', '#4daf4a']
    corners = [(0, 0), (2, 0), (0, 2), (2, 2)]

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.5, 2.5)
    ax.invert_yaxis()   # image coords: y=0 at top, y=2 at bottom
    ax.set_aspect('equal')

    if shade:
        # Each colored sub-rectangle is the region DIAGONALLY OPPOSITE its corner
        # corner (0,0) [top-left]     → bottom-right: x∈[xi,2], y∈[yj,2]
        # corner (2,0) [top-right]    → bottom-left:  x∈[0,xi], y∈[yj,2]
        # corner (0,2) [bottom-left]  → top-right:    x∈[xi,2], y∈[0,yj]
        # corner (2,2) [bottom-right] → top-left:     x∈[0,xi], y∈[0,yj]
        regions = [
            (xi, yj, 2 - xi, 2 - yj),
            (0,  yj, xi,     2 - yj),
            (xi, 0,  2 - xi, yj),
            (0,  0,  xi,     yj),
        ]
        for (rx, ry, rw, rh), col in zip(regions, colors):
            ax.add_patch(patches.Rectangle((rx, ry), rw, rh,
                                           linewidth=0, facecolor=col, alpha=0.30))

    # Grid border
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(patches.Rectangle((0, 0), 2, 2,
                                   linewidth=1.5, edgecolor='k', facecolor='none'))
    # Dashed cross-hair at pixel position
    ax.axvline(xi, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.axhline(yj, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    # Corner dots + two-line labels (coordinate + magnitude value).
    # Offsets are in screen-space points so they're independent of the inverted y axis.
    # Top corners (y=0): label goes above (+y in screen = upward)
    # Bottom corners (y=2): label goes below (-y in screen = downward)
    offsets_pts = {
        (0, 0): (8,  20),   # above-right
        (2, 0): (-8, 20),   # above-left
        (0, 2): (8, -20),   # below-right
        (2, 2): (-8,-20),   # below-left
    }
    ha_map = {(0, 0): 'left',  (2, 0): 'right',
              (0, 2): 'left',  (2, 2): 'right'}
    va_map = {(0, 0): 'bottom', (2, 0): 'bottom',
              (0, 2): 'top',    (2, 2): 'top'}

    for (cx, cy), col in zip(corners, colors):
        ax.plot(cx, cy, 'o', color=col, markersize=11, zorder=5)
        dx_pts, dy_pts = offsets_pts[(cx, cy)]
        lbl = f'$({cx},{cy})$\n$I={mags[(cx, cy)]}$'
        ax.annotate(lbl, (cx, cy),
                    xytext=(dx_pts, dy_pts), textcoords='offset points',
                    color=col, fontsize=9, linespacing=1.4,
                    ha=ha_map[(cx, cy)], va=va_map[(cx, cy)])

    # Interpolation point marker
    ax.plot(xi, yj, 'k^', markersize=9, zorder=6)
    ax.annotate(f'$({xi},\\,{yj})$\n$I=\\,?$', (xi, yj),
                xytext=(6, -6), textcoords='offset points',
                fontsize=9, va='top', linespacing=1.4)

    ax.set_xticks([0, xi, 2])
    ax.set_yticks([0, yj, 2])
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y\\;(\\downarrow)$', fontsize=11)
    ax.set_title('Bilinear interpolation grid\n'
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


def render(params, computed, mode='full', question_num=None, marks=False,
           out_dir=None, img_stem=None):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    xi, yj = p['xi'], p['yj']
    cmags = c['corner_mags']

    shade = mode in ('answer', 'full')
    if out_dir and img_stem:
        img_filename = f'{img_stem}.png'
        _make_plot(p, shade=shade, save_path=os.path.join(out_dir, img_filename))
        img_md = f'![Bilinear interpolation grid]({img_filename})'
    else:
        b64 = _make_plot(p, shade=shade)
        img_md = f'![Bilinear interpolation grid](data:image/png;base64,{b64})'

    # Corner magnitude table for the question text
    mag_rows = ' | '.join(
        f'$I_{{({cx},{cy})}} = {cmags[f"({cx},{cy})"]}$'
        for cx, cy in [(0,0),(2,0),(0,2),(2,2)]
    )

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Bilinear Interpolation'),
        '',
        '> **Note:** Image coordinate system is used — the origin is at the'
        ' **top-left**, and $y$ increases **downward**.',
        '',
        r'A $2\times 2$ grid cell has known intensity values at its four corners'
        r' (see plot). An interior pixel at $(x_i, y_j)$'
        f' $= ({xi}, {yj})$ must be interpolated.',
        '',
        '| Corner | Position | Intensity |',
        '|--------|----------|-----------|',
        f'| $(0,0)$ | top-left     | $I_{{(0,0)}} = {cmags["(0,0)"]}$ |',
        f'| $(2,0)$ | top-right    | $I_{{(2,0)}} = {cmags["(2,0)"]}$ |',
        f'| $(0,2)$ | bottom-left  | $I_{{(0,2)}} = {cmags["(0,2)"]}$ |',
        f'| $(2,2)$ | bottom-right | $I_{{(2,2)}} = {cmags["(2,2)"]}$ |',
        '',
        img_md,
        '',
        r'Using bilinear interpolation, compute the interpolated intensity $I(x_i, y_j)$:',
        '',
        r'$$I(x_i,y_j) = \frac{\displaystyle\sum_{c}\, I_c \cdot A_c}{\displaystyle\sum_{c} A_c}$$',
        '',
        r'where $A_c$ is the area of the sub-rectangle **diagonally opposite** corner $c$'
        r' and $\sum_c A_c = 4$.',
        '',
    ]

    if mode == 'blank':
        lines += ['$I(x_i, y_j)$ = `___`', '']

    elif mode == 'exam':
        qbox = inline_box(width_pt=90)
        lines += [
            '**Part (a):** Sub-rectangle areas — '
            f'$A_{{(0,0)}}=${qbox} $A_{{(2,0)}}=${qbox} $A_{{(0,2)}}=${qbox} $A_{{(2,2)}}=${qbox}',
            '',
            '**Part (b):** Weighted contributions — '
            f'$I_{{(0,0)}}A_{{(0,0)}}=${qbox} $I_{{(2,0)}}A_{{(2,0)}}=${qbox}'
            f' $I_{{(0,2)}}A_{{(0,2)}}=${qbox} $I_{{(2,2)}}A_{{(2,2)}}=${qbox}',
            '',
            '**Part (c):** $I(x_i, y_j) =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Sub-rectangle areas.**',
            '',
            r'In image coordinates (y increases downward) the pixel $(x_i, y_j)$'
            r' splits the $2\times 2$ cell into four rectangles.'
            r' The area diagonally opposite each corner is:',
            '',
        ]
        for corner, area in c['areas'].items():
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
            f'Check: ${" + ".join(str(a) for a in c["areas"].values())}'
            f' = {c["total_area"]}$ ✓',
            '',
            _mark(MARKS['coordinates']),
        ]
        lines += [
            '',
            '**Part (b) - Weighted contributions.**',
            '',
        ]
        for corner, w in c['weighted'].items():
            I_c = cmags[corner]
            A_c = c['areas'][corner]
            cx_str, cy_str = corner[1:-1].split(',')
            lines.append(
                f'$I_{{({cx_str},{cy_str})}} \\cdot A_{{({cx_str},{cy_str})}}$'
                f' = ${I_c} \\times {A_c} = {w}$'
            )
        lines += ['', _mark(MARKS['weights'])]
        weighted_vals = list(c['weighted'].values())
        total_w = round(sum(weighted_vals), 4)
        sum_str = ' + '.join(str(v) for v in weighted_vals)
        lines += [
            '',
            '**Part (c) - Sum and divide by the total area.**',
            '',
            f'$$I({xi}, {yj})'
            f' = \\frac{{{sum_str}}}{{4}}'
            f' = \\frac{{{total_w}}}{{4}}'
            f' = {c["M_interp"]}$$',
            '',
            _mark(MARKS['interpolated']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $I({xi}, {yj}) = {c["M_interp"]}$',
            '',
            '> **Hint:** For each corner $c$, find the area $A_c$ of the'
            r' sub-rectangle diagonally opposite to it: $A_c = \Delta x \cdot \Delta y$'
            r' where $\Delta x$ and $\Delta y$ are the distances from the pixel to'
            ' the opposite corner along each axis.'
            r' Then $I = (I_{(0,0)} A_{(0,0)} + I_{(2,0)} A_{(2,0)} +'
            r' I_{(0,2)} A_{(0,2)} + I_{(2,2)} A_{(2,2)}) / 4$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Compute the four sub-rectangle areas.**',
            '',
            r'In image coordinates (y increases downward) the pixel $(x_i, y_j)$'
            r' splits the $2\times 2$ cell into four rectangles.'
            r' The area diagonally opposite each corner is:',
            '',
        ]
        for corner, area in c['areas'].items():
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
            f'Check: ${" + ".join(str(a) for a in c["areas"].values())}'
            f' = {c["total_area"]}$ ✓',
        ]
        if marks:
            lines += ['', _mark(MARKS['coordinates'])]
        lines += [
            '',
            '**Step 2 – Compute each corner\'s weighted contribution.**',
            '',
        ]
        for corner, w in c['weighted'].items():
            I_c = cmags[corner]
            A_c = c['areas'][corner]
            pos = _CORNER_POS[corner]
            cx_str, cy_str = corner[1:-1].split(',')
            lines.append(
                f'$I_{{({cx_str},{cy_str})}} \\cdot A_{{({cx_str},{cy_str})}}$'
                f' = ${I_c} \\times {A_c} = {w}$'
            )
        if marks:
            lines += ['', _mark(MARKS['weights'])]
        lines += [
            '',
            '**Step 3 – Sum and divide by the total area.**',
            '',
        ]
        weighted_vals = list(c['weighted'].values())
        total_w = round(sum(weighted_vals), 4)
        sum_str = ' + '.join(str(v) for v in weighted_vals)
        lines += [
            f'$$I({xi}, {yj})'
            f' = \\frac{{{sum_str}}}{{4}}'
            f' = \\frac{{{total_w}}}{{4}}'
            f' = {c["M_interp"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['interpolated'])]

    return '\n'.join(lines)
