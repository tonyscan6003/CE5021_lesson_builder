"""Generator for Intersection over Union (IoU) questions.

Two bounding boxes (predicted Bp and ground truth Bgt) are always generated
to overlap.  The student computes intersection area, union area, and IoU.

Modes:
    'blank'  – question + box diagram
    'answer' – question + diagram + answer + hint
    'full'   – question + diagram + step-by-step solution
"""
import io
import base64
import numpy as np
from ._exam_boxes import answer_box

# Mark allocation per step — must sum to 10
MARKS = {
    'intersection': 3,  # intersection rectangle area
    'union': 4,         # individual box areas and union
    'ratio': 3,         # IoU ratio
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, img_size=100):
    rng = np.random.default_rng(seed)
    while True:
        # Box 1 (predicted)
        x1 = int(rng.integers(10, 60))
        y1 = int(rng.integers(10, 60))
        x2 = x1 + int(rng.integers(15, 40))
        y2 = y1 + int(rng.integers(15, 40))

        # Box 2 (ground truth) — overlap enforced
        ox = int(rng.integers(5, 15))  # overlap width
        oy = int(rng.integers(5, 15))  # overlap height
        x3 = x2 - ox
        y3 = y2 - oy
        x4 = x3 + int(rng.integers(15, 40))
        y4 = y3 + int(rng.integers(15, 40))

        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        if xi2 > xi1 and yi2 > yi1:
            break

    inter = (xi2 - xi1) * (yi2 - yi1)
    area_p = (x2 - x1) * (y2 - y1)
    area_gt = (x4 - x3) * (y4 - y3)
    union = area_p + area_gt - inter
    iou = round(inter / union, 2)

    return {
        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
        'x3': x3, 'y3': y3, 'x4': x4, 'y4': y4,
    }


def compute(params):
    p = params
    xi1 = max(p['x1'], p['x3'])
    yi1 = max(p['y1'], p['y3'])
    xi2 = min(p['x2'], p['x4'])
    yi2 = min(p['y2'], p['y4'])
    inter = (xi2 - xi1) * (yi2 - yi1)
    area_p = (p['x2'] - p['x1']) * (p['y2'] - p['y1'])
    area_gt = (p['x4'] - p['x3']) * (p['y4'] - p['y3'])
    union = area_p + area_gt - inter
    iou = round(inter / union, 2)
    return {
        'xi1': xi1, 'yi1': yi1, 'xi2': xi2, 'yi2': yi2,
        'inter': inter, 'area_p': area_p, 'area_gt': area_gt,
        'union': union, 'iou': iou,
    }


def _make_plot(params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    p = params
    fig, ax = plt.subplots(figsize=(5, 5))

    # Predicted box — blue; ground truth — green
    bp = mpatches.FancyBboxPatch(
        (p['x1'], p['y1']), p['x2'] - p['x1'], p['y2'] - p['y1'],
        boxstyle='square,pad=0', linewidth=2,
        edgecolor='#377eb8', facecolor='#377eb8', alpha=0.25,
        label='$B_p$ (predicted)')
    bgt = mpatches.FancyBboxPatch(
        (p['x3'], p['y3']), p['x4'] - p['x3'], p['y4'] - p['y3'],
        boxstyle='square,pad=0', linewidth=2,
        edgecolor='#2ca02c', facecolor='#2ca02c', alpha=0.25,
        label='$B_{gt}$ (ground truth)')
    ax.add_patch(bp)
    ax.add_patch(bgt)

    # Corner annotations with screen-space offsets so they sit outside their box.
    # In image coords (y inverted): (x1,y1) and (x3,y3) are top-left corners
    # (upper on screen), (x2,y2) and (x4,y4) are bottom-right (lower on screen).
    # Positive y offset in points = upward on screen regardless of axis direction.
    corner_info = [
        ((p['x1'], p['y1']), f'({p["x1"]},{p["y1"]})', 'right', (-4,  8)),
        ((p['x2'], p['y2']), f'({p["x2"]},{p["y2"]})', 'left',  ( 4, -10)),
        ((p['x3'], p['y3']), f'({p["x3"]},{p["y3"]})', 'left',  ( 4,  8)),
        ((p['x4'], p['y4']), f'({p["x4"]},{p["y4"]})', 'right', (-4, -10)),
    ]
    for (cx, cy), label, ha, (dx, dy) in corner_info:
        ax.annotate(label, (cx, cy), fontsize=7.5, ha=ha,
                    xytext=(dx, dy), textcoords='offset points')

    margin = 10
    all_x = [p['x1'], p['x2'], p['x3'], p['x4']]
    all_y = [p['y1'], p['y2'], p['y3'], p['y4']]
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.invert_yaxis()   # image coordinates: y=0 at top, y increases downward
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y\\;(\\downarrow)$')
    ax.set_title('Bounding Boxes\n(image coords: origin top-left, $y$ increases downward)',
                 fontsize=9)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''

    b64 = _make_plot(p)
    img_md = f'![Bounding boxes](data:image/png;base64,{b64})'

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Intersection over Union'),
        '',
        '> **Note:** Image coordinate system is used — the origin is at the'
        ' **top-left**, and $y$ increases **downward**.',
        '',
        'Calculate the IoU of the two bounding boxes shown.',
        '',
        f'**Predicted box** $B_p$ (blue): top-left $({p["x1"]}, {p["y1"]})$,'
        f' bottom-right $({p["x2"]}, {p["y2"]})$.',
        '',
        f'**Ground-truth box** $B_{{gt}}$ (green): top-left $({p["x3"]}, {p["y3"]})$,'
        f' bottom-right $({p["x4"]}, {p["y4"]})$.',
        '',
        img_md,
        '',
        r'$$\mathrm{IoU} = \frac{\mathrm{Area}(B_p \cap B_{gt})}{\mathrm{Area}(B_p \cup B_{gt})}$$',
        '',
        'Give your answer to 2 decimal places.',
        '',
    ]

    if mode == 'blank':
        lines.append('**IoU =** `_______________`')

    elif mode == 'exam':
        lines += [
            '**Part (a):** Area of intersection $=$',
            '',
            answer_box(),
            '',
            '**Part (b):** Area of union $=$',
            '',
            answer_box(),
            '',
            '**Part (c):** IoU $=$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Intersection rectangle.**',
            '',
            r'In image coordinates (y downward), the intersection top-left corner'
            r' uses $\max$ on both axes (picking the lower/rightmost of the two top-left corners),'
            r' and the bottom-right uses $\min$ (the higher/leftmost of the two bottom-right corners).',
            '',
            f'$$x_{{I1}} = \\max({p["x1"]}, {p["x3"]}) = {c["xi1"]}$$',
            f'$$y_{{I1}} = \\max({p["y1"]}, {p["y3"]}) = {c["yi1"]}$$',
            f'$$x_{{I2}} = \\min({p["x2"]}, {p["x4"]}) = {c["xi2"]}$$',
            f'$$y_{{I2}} = \\min({p["y2"]}, {p["y4"]}) = {c["yi2"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_p \\cap B_{{gt}})'
            f'= ({c["xi2"]} - {c["xi1"]}) \\cdot ({c["yi2"]} - {c["yi1"]}) = {c["inter"]}$$',
            '',
            _mark(MARKS['intersection']),
        ]
        lines += [
            '',
            '**Part (b) - Individual box areas and union.**',
            '',
            f'$$\\mathrm{{Area}}(B_p)'
            f'= ({p["x2"]} - {p["x1"]}) \\cdot ({p["y2"]} - {p["y1"]}) = {c["area_p"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_{{gt}})'
            f'= ({p["x4"]} - {p["x3"]}) \\cdot ({p["y4"]} - {p["y3"]}) = {c["area_gt"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_p \\cup B_{{gt}})'
            f'= {c["area_p"]} + {c["area_gt"]} - {c["inter"]} = {c["union"]}$$',
            '',
            _mark(MARKS['union']),
        ]
        lines += [
            '',
            '**Part (c) - IoU ratio.**',
            '',
            f'$$\\mathrm{{IoU}} = \\frac{{{c["inter"]}}}{{{c["union"]}}} = {c["iou"]}$$',
            '',
            _mark(MARKS['ratio']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $\\mathrm{{IoU}} = {c["iou"]}$',
            '',
            r'> **Hint:** Find the intersection rectangle corners using'
            r' $x_{I1} = \max(x_{p1}, x_{gt1})$, $y_{I1} = \max(y_{p1}, y_{gt1})$,'
            r' $x_{I2} = \min(x_{p2}, x_{gt2})$, $y_{I2} = \min(y_{p2}, y_{gt2})$'
            r' (in image coordinates, $\max$ on $y$ picks the lower of the two top edges).'
            r' Then $\text{Intersection} = (x_{I2}-x_{I1})(y_{I2}-y_{I1})$,'
            r' $\text{Union} = \text{Area}(B_p) + \text{Area}(B_{gt}) - \text{Intersection}$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Intersection rectangle.**',
            '',
            r'In image coordinates (y downward), the intersection top-left corner'
            r' uses $\max$ on both axes (picking the lower/rightmost of the two top-left corners),'
            r' and the bottom-right uses $\min$ (the higher/leftmost of the two bottom-right corners).',
            '',
            f'$$x_{{I1}} = \\max({p["x1"]}, {p["x3"]}) = {c["xi1"]}$$',
            f'$$y_{{I1}} = \\max({p["y1"]}, {p["y3"]}) = {c["yi1"]}$$',
            f'$$x_{{I2}} = \\min({p["x2"]}, {p["x4"]}) = {c["xi2"]}$$',
            f'$$y_{{I2}} = \\min({p["y2"]}, {p["y4"]}) = {c["yi2"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_p \\cap B_{{gt}})'
            f'= ({c["xi2"]} - {c["xi1"]}) \\cdot ({c["yi2"]} - {c["yi1"]}) = {c["inter"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['intersection'])]
        lines += [
            '',
            '**Step 2 – Individual box areas.**',
            '',
            f'$$\\mathrm{{Area}}(B_p)'
            f'= ({p["x2"]} - {p["x1"]}) \\cdot ({p["y2"]} - {p["y1"]}) = {c["area_p"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_{{gt}})'
            f'= ({p["x4"]} - {p["x3"]}) \\cdot ({p["y4"]} - {p["y3"]}) = {c["area_gt"]}$$',
            '',
            f'$$\\mathrm{{Area}}(B_p \\cup B_{{gt}})'
            f'= {c["area_p"]} + {c["area_gt"]} - {c["inter"]} = {c["union"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['union'])]
        lines += [
            '',
            '**Step 3 – Union and IoU.**',
            '',
            f'$$\\mathrm{{IoU}} = \\frac{{{c["inter"]}}}{{{c["union"]}}} = {c["iou"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['ratio'])]

    return '\n'.join(lines)
