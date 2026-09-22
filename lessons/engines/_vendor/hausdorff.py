"""Generator for Hausdorff distance questions (three-part).

Two nearby pentagons A and B are generated. The student works through:
  (a) directed distance h(A→B) = max_{a∈A} min_{b∈B} ||a-b||
  (b) directed distance h(B→A) = max_{b∈B} min_{a∈A} ||b-a||
  (c) Hausdorff distance H(A,B) = max(h(A→B), h(B→A))

A plot of both point sets is rendered in all modes.

Modes:
    'blank'  – question + plot, three blanks
    'answer' – question + plot + all three answers + hint
    'full'   – question + plot + full step-by-step for all three parts
"""
import io
import base64
import numpy as np
from ._exam_boxes import answer_box

# Mark allocation per step — must sum to 10
MARKS = {
    'directed_ab': 4,  # directed Hausdorff h(A->B)
    'directed_ba': 4,  # directed Hausdorff h(B->A)
    'hausdorff': 2,    # symmetric Hausdorff H(A,B)
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def _regular_pentagon(cx, cy, radius, angle_offset=0.0):
    """Return 5 (x, y) vertices of a regular pentagon."""
    angles = [angle_offset + 2 * np.pi * k / 5 for k in range(5)]
    return [(round(cx + radius * np.cos(a), 2),
             round(cy + radius * np.sin(a), 2)) for a in angles]


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    radius = round(float(rng.uniform(1.0, 2.0)), 1)
    cx_a = round(float(rng.uniform(3.0, 5.0)), 1)
    cy_a = round(float(rng.uniform(3.0, 5.0)), 1)

    # Pentagon B is shifted by a small random vector so they overlap / are close
    shift_x = round(float(rng.uniform(-1.2, 1.2)), 1)
    shift_y = round(float(rng.uniform(-1.2, 1.2)), 1)
    cx_b = round(cx_a + shift_x, 1)
    cy_b = round(cy_a + shift_y, 1)

    angle_a = round(float(rng.uniform(0, 2 * np.pi)), 2)
    angle_b = round(float(rng.uniform(0, 2 * np.pi)), 2)

    A = _regular_pentagon(cx_a, cy_a, radius, angle_a)
    B = _regular_pentagon(cx_b, cy_b, radius, angle_b)
    return {'A': A, 'B': B}


def compute(params):
    A, B = params['A'], params['B']

    def dist(p, q):
        return round(float(np.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)), 2)

    # Part (a): h(A→B) — for each a find min dist to B
    table_ab = []
    for a in A:
        dists = [dist(a, b) for b in B]
        min_d = round(min(dists), 2)
        table_ab.append({'point': a, 'dists': dists, 'min': min_d})
    h_ab = round(max(r['min'] for r in table_ab), 2)

    # Part (b): h(B→A) — for each b find min dist to A
    table_ba = []
    for b in B:
        dists = [dist(b, a) for a in A]
        min_d = round(min(dists), 2)
        table_ba.append({'point': b, 'dists': dists, 'min': min_d})
    h_ba = round(max(r['min'] for r in table_ba), 2)

    # Part (c): H(A,B) = max of the two directed distances
    H = round(max(h_ab, h_ba), 2)

    return {
        'table_ab': table_ab,
        'table_ba': table_ba,
        'h_ab': h_ab,
        'h_ba': h_ba,
        'H': H,
    }


def _make_plot(params):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPoly
    from matplotlib.collections import PatchCollection

    A, B = params['A'], params['B']

    fig, ax = plt.subplots(figsize=(5, 5))

    poly_a = MplPoly(A, closed=True)
    poly_b = MplPoly(B, closed=True)

    for pts, col, lbl in [(A, '#377eb8', 'A'), (B, '#e41a1c', 'B')]:
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, '-o', color=col, linewidth=1.5, markersize=6, label=f'Set {lbl}')
        for i, (x, y) in enumerate(pts):
            ax.annotate(f'${"ab"[lbl == "B"]}_{i}=({x},{y})$',
                        (x, y), fontsize=7.5, textcoords='offset points',
                        xytext=(5, 3), color=col)

    all_x = [p[0] for p in A + B]
    all_y = [p[1] for p in A + B]
    pad = 0.8
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect('equal')
    ax.legend(fontsize=9)
    ax.set_title('Point sets A and B', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=90)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _dist_table_lines(pts_from, pts_to, label_from, label_to, table):
    """Return markdown lines for one directed distance table."""
    lines = []
    for i, row in enumerate(table):
        pt = row['point']
        lines += [f'**${label_from}_{i} = ({pt[0]}, {pt[1]})$**', '']
        for j, (q, d) in enumerate(zip(pts_to, row['dists'])):
            lines.append(
                f'- ${label_to}_{j} = ({q[0]},{q[1]})$:'
                f' $\\|{label_from}_{i} - {label_to}_{j}\\| = {d}$'
            )
        lines += ['', f'$\\min = {row["min"]}$', '']
    return lines


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    A, B = p['A'], p['B']

    b64 = _make_plot(p)
    img_md = f'![Hausdorff distance point sets](data:image/png;base64,{b64})'

    a_str = ', '.join(f'({x},{y})' for x, y in A)
    b_str = ', '.join(f'({x},{y})' for x, y in B)

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Hausdorff Distance'),
        '',
        f'Given two point sets $A = \\{{\\,{a_str}\\,\\}}$'
        f' and $B = \\{{\\,{b_str}\\,\\}}$:',
        '',
        img_md,
        '',
        r'Distances are Euclidean: $\|p - q\| = \sqrt{(p_x-q_x)^2+(p_y-q_y)^2}$.',
        '',
        r'**(a)** Compute the **directed** Hausdorff distance'
        r' $h(A \to B) = \max_{a \in A}\;\min_{b \in B}\;\|a - b\|$',
        '',
        r'**(b)** Compute the **directed** Hausdorff distance'
        r' $h(B \to A) = \max_{b \in B}\;\min_{a \in A}\;\|b - a\|$',
        '',
        r'**(c)** Hence determine the **Hausdorff distance**'
        r' $H(A,B) = \max\bigl(h(A \to B),\;h(B \to A)\bigr)$',
        '',
    ]

    if mode == 'blank':
        lines += [
            r'**(a)** $h(A \to B)$ = `___`',
            '',
            r'**(b)** $h(B \to A)$ = `___`',
            '',
            r'**(c)** $H(A,B)$ = `___`',
        ]

    elif mode == 'exam':
        lines += [
            r'**Part (a)** $h(A \to B)$ =',
            '',
            answer_box(),
            '',
            r'**Part (b)** $h(B \to A)$ =',
            '',
            answer_box(),
            '',
            r'**Part (c)** $H(A,B)$ =',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a)** — $h(A \\to B)$', '',
            r'For each $a_i \in A$, compute $\|a_i - b_j\|$ for every $b_j \in B$'
            r' and keep the minimum.',
            '',
        ]
        lines += _dist_table_lines(A, B, 'a', 'b', c['table_ab'])
        lines += [
            '$$h(A \\to B)'
            f' = \\max({", ".join(str(r["min"]) for r in c["table_ab"])})'
            f' = {c["h_ab"]}$$',
            '',
            _mark(MARKS['directed_ab']),
        ]
        lines += [
            '',
            '**Part (b)** — $h(B \\to A)$', '',
            r'For each $b_j \in B$, compute $\|b_j - a_i\|$ for every $a_i \in A$'
            r' and keep the minimum.',
            '',
        ]
        lines += _dist_table_lines(B, A, 'b', 'a', c['table_ba'])
        lines += [
            '$$h(B \\to A)'
            f' = \\max({", ".join(str(r["min"]) for r in c["table_ba"])})'
            f' = {c["h_ba"]}$$',
            '',
            _mark(MARKS['directed_ba']),
        ]
        lines += [
            '',
            '**Part (c)** — $H(A,B)$', '',
            f'$$H(A,B) = \\max\\bigl(h(A \\to B),\\;h(B \\to A)\\bigr)'
            f' = \\max({c["h_ab"]},\\;{c["h_ba"]}) = {c["H"]}$$',
            '',
            _mark(MARKS['hausdorff']),
        ]

    elif mode == 'answer':
        lines += [
            f'**(a)** $h(A \\to B) = {c["h_ab"]}$',
            '',
            f'**(b)** $h(B \\to A) = {c["h_ba"]}$',
            '',
            f'**(c)** $H(A,B) = \\max({c["h_ab"]},\\;{c["h_ba"]}) = {c["H"]}$',
            '',
            r'> **Hint:** **(a)/(b)** For each point in the source set, find its'
            r' nearest neighbour in the target set (minimum distance). The directed'
            r' distance is the **largest** of those minimums.'
            r' **(c)** The (symmetric) Hausdorff distance is the maximum of the two'
            r' directed distances.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '#### Part (a) — $h(A \\to B)$', '',
            r'For each $a_i \in A$, compute $\|a_i - b_j\|$ for every $b_j \in B$'
            r' and keep the minimum.',
            '',
        ]
        lines += _dist_table_lines(A, B, 'a', 'b', c['table_ab'])
        lines += [
            '$$h(A \\to B)'
            f' = \\max({", ".join(str(r["min"]) for r in c["table_ab"])})'
            f' = {c["h_ab"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['directed_ab'])]
        lines += [
            '',
            '#### Part (b) — $h(B \\to A)$', '',
            r'For each $b_j \in B$, compute $\|b_j - a_i\|$ for every $a_i \in A$'
            r' and keep the minimum.',
            '',
        ]
        lines += _dist_table_lines(B, A, 'b', 'a', c['table_ba'])
        lines += [
            '$$h(B \\to A)'
            f' = \\max({", ".join(str(r["min"]) for r in c["table_ba"])})'
            f' = {c["h_ba"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['directed_ba'])]
        lines += [
            '',
            '#### Part (c) — $H(A,B)$', '',
            f'$$H(A,B) = \\max\\bigl(h(A \\to B),\\;h(B \\to A)\\bigr)'
            f' = \\max({c["h_ab"]},\\;{c["h_ba"]}) = {c["H"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['hausdorff'])]

    return '\n'.join(lines)
