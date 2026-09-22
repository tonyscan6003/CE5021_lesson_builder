"""Generator for Hough transform (slope from parameter space) questions.

A set of image points is shown as sinusoids in (rho, theta) parameter space.
The student identifies the dominant intersection and computes the line slope.

Modes:
    'blank'  – question + sinusoid plot (intersection unmarked)
    'answer' – question + plot (intersection marked) + answer + hint
    'full'   – question + plot + step-by-step solution
"""
import io
import os
import base64
import numpy as np
from ._exam_boxes import answer_box, inline_box

# Mark allocation per step — must sum to 10
MARKS = {
    'accumulator': 6,  # reading dominant intersection
    'peaks': 4,        # converting to slope
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None, n_line_pts=4, n_noise_pts=1):
    rng = np.random.default_rng(seed)
    theta = round(float(rng.uniform(0.55, 1.15)), 2)
    rho = round(float(rng.uniform(-3.0, 3.0)), 2)

    m = -np.cos(theta) / np.sin(theta)
    c = rho / np.sin(theta)

    x_pool = list(range(-3, 4))
    rng.shuffle(x_pool)
    x_line = sorted(x_pool[:n_line_pts])
    line_pts = [(float(x), round(float(m * x + c), 2)) for x in x_line]

    noise_pts = []
    for _ in range(n_noise_pts):
        xn = float(rng.integers(-2, 3))
        yn_target = m * xn + c
        sign = float(rng.choice([-1, 1]))
        yn = round(float(yn_target + sign * rng.uniform(2.5, 5.0)), 2)
        noise_pts.append((xn, yn))

    return {
        'theta': theta,
        'rho': rho,
        'm': round(float(m), 3),
        'c': round(float(c), 3),
        'line_pts': line_pts,
        'noise_pts': noise_pts,
        'all_pts': line_pts + noise_pts,
    }


def _make_plot(params, mark_intersection, save_path=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    theta_arr = np.linspace(0, np.pi, 500)
    palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628']

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (xi, yi) in enumerate(params['all_pts']):
        rho_arr = xi * np.cos(theta_arr) + yi * np.sin(theta_arr)
        ax.plot(theta_arr, rho_arr, color=palette[i % len(palette)],
                linewidth=1.6, label=f'$({xi:.1f},\\ {yi:.1f})$')

    if mark_intersection:
        t0, r0 = params['theta'], params['rho']
        ax.plot(t0, r0, 'k*', markersize=11, zorder=6,
                label=fr'$(\rho,\theta)=({r0},\ {t0})$')

    ax.set_xlabel(r'$\theta$ (radians)', fontsize=11)
    ax.set_ylabel(r'$\rho$', fontsize=11)
    ax.set_title(r'Hough Parameter Space  $\rho = x\cos\theta + y\sin\theta$', fontsize=11)
    ax.set_xlim(0, np.pi)
    ax.set_ylim(-8, 8)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
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


def compute(params):
    return {}


def render(params, computed, mode='full', question_num=None, marks=False,
           out_dir=None, img_stem=None):
    num_str = f' {question_num}' if question_num is not None else ''
    theta, rho, m = params['theta'], params['rho'], params['m']

    mark = mode in ('answer', 'full')
    if out_dir and img_stem:
        img_filename = f'{img_stem}.png'
        _make_plot(params, mark_intersection=mark, save_path=os.path.join(out_dir, img_filename))
        img_md = f'![Hough parameter space]({img_filename})'
    else:
        b64 = _make_plot(params, mark_intersection=mark)
        img_md = f'![Hough parameter space](data:image/png;base64,{b64})'

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Hough Transform'),
        '',
        'The Hough transform has been applied to a series of image points.',
        r'The sinusoids in parameter $(\rho, \theta)$ space are shown below.',
        '',
        img_md,
        '',
        r'Identify the dominant intersection $(\rho, \theta)$ where the most sinusoids'
        r' meet, and determine the **slope** $m$ of the primary line in the image.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `_______________`')

    elif mode == 'exam':
        lines += [
            r'**Part (a):** Dominant intersection — '
            f'$\\rho =$ {inline_box()} $\\theta =$ {inline_box()}',
            '',
            '**Part (b):** Line slope $m =$',
            '',
            answer_box(),
        ]

    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Dominant intersection.**',
            '',
            r'Every point $(x_i, y_i)$ in image space can lie on many candidate lines.'
            r' In polar form each candidate line through $(x_i, y_i)$ at angle $\theta$ satisfies',
            '',
            r'$$\rho = x_i\cos\theta + y_i\sin\theta$$',
            '',
            r'Plotting $\rho$ vs $\theta$ for every point gives a sinusoid. Lines shared by'
            r' multiple points produce intersections at the same $(\rho, \theta)$.'
            r' The majority of sinusoids converge at',
            '',
            fr'$$(\rho,\,\theta) = ({rho},\; {theta})$$',
            '',
            _mark(MARKS['accumulator']),
        ]
        lines += [
            '',
            '**Part (b) - Cartesian slope.**',
            '',
            r'The polar line from the origin to the candidate line has slope'
            r' $m_p = \sin\theta/\cos\theta$.'
            r' The candidate line is perpendicular to this polar line, so',
            '',
            r'$$m = -\frac{1}{m_p} = -\frac{\cos\theta}{\sin\theta}'
            fr'= -\frac{{\cos({theta})}}{{\sin({theta})}} = {m}$$',
            '',
            _mark(MARKS['peaks']),
        ]

    elif mode == 'answer':
        lines += [
            f'**Answer:** $m = {m}$',
            '',
            r'> **Hint:** Find the $(\rho, \theta)$ where the greatest number of sinusoids'
            r' intersect (the dominant vote). The line slope follows from'
            r' $m = -\dfrac{\cos\theta}{\sin\theta}$.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 – Recall the Hough parameterisation.**',
            '',
            r'Every point $(x_i, y_i)$ in image space can lie on many candidate lines.'
            r' In polar form each candidate line through $(x_i, y_i)$ at angle $\theta$ satisfies',
            '',
            r'$$\rho = x_i\cos\theta + y_i\sin\theta$$',
            '',
            r'Plotting $\rho$ vs $\theta$ for every point gives a sinusoid. Lines shared by'
            r' multiple points produce intersections at the same $(\rho, \theta)$.',
            '',
            '**Step 2 – Read the dominant intersection.**',
            '',
            r'The majority of sinusoids converge at',
            '',
            fr'$$(\rho,\,\theta) = ({rho},\; {theta})$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['accumulator'])]
        lines += [
            '',
            '**Step 3 – Convert to Cartesian slope.**',
            '',
            r'The polar line from the origin to the candidate line has slope'
            r' $m_p = \sin\theta/\cos\theta$.'
            r' The candidate line is perpendicular to this polar line, so',
            '',
            r'$$m = -\frac{1}{m_p} = -\frac{\cos\theta}{\sin\theta}'
            fr'= -\frac{{\cos({theta})}}{{\sin({theta})}} = {m}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['peaks'])]

    return '\n'.join(lines)
