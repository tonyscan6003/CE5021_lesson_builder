"""Otsu's optimal thresholding question generator.

Given a histogram of pixel intensities over 4 discrete levels (0, 1, 2, 3)
with pixel counts summing to 20, compute the inter-class variance for each
of the 3 candidate thresholds (t=0, t=1, t=2) and identify the optimal one.

Formula:
    σ²_B(t) = w₀(t) · w₁(t) · (μ₀(t) − μ₁(t))²

where:
    w₀(t) = sum of probabilities for levels ≤ t        (background weight)
    w₁(t) = 1 − w₀(t)                                  (foreground weight)
    μ₀(t) = weighted mean of levels ≤ t
    μ₁(t) = (μ_T − w₀·μ₀) / w₁     (derived from overall mean)
    μ_T   = sum(level_i × count_i) / 20

Modes:
    'blank'      – question only
    'answer'     – question + completed variance table + hint
    'full'       – full step-by-step solution (no mark annotations)
    'full_marks' – same as full, with mark annotations
"""
import numpy as np
from ._exam_boxes import answer_box, inline_box, cell_box

MARKS = {
    'class_stats': 5,   # class weights and means for all thresholds
    'variances':   3,   # σ²_B values for all thresholds
    'threshold':   2,   # identifying the optimal threshold
}

EXAM_COMPATIBLE = True


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    levels = [0, 1, 2, 3]
    N = 20

    for _ in range(500):
        counts = rng.multinomial(N, [0.25, 0.25, 0.25, 0.25])
        if any(cnt < 2 for cnt in counts):
            continue

        # Compute variances to check the optimal threshold is non-trivial
        probs = counts / N
        mu_T = float(np.sum(np.array(levels) * probs))

        variances = []
        valid = True
        for t in range(3):
            w0 = float(np.sum(probs[:t + 1]))
            w1 = 1.0 - w0
            if w0 < 1e-9 or w1 < 1e-9:
                valid = False
                break
            mu0 = float(np.sum(np.array(levels[:t + 1]) * probs[:t + 1])) / w0
            mu1 = (mu_T - w0 * mu0) / w1
            sigma2 = w0 * w1 * (mu0 - mu1) ** 2
            variances.append(sigma2)

        if not valid:
            continue

        optimal_t = int(np.argmax(variances))
        # Prefer non-trivial (t=1 or t=2 is optimal), but don't loop forever;
        # just skip if the max variance ties trivially at t=0 or t=2 only
        if optimal_t in (1, 2):
            return {'levels': levels, 'counts': counts.tolist()}

    # Fallback: return any valid result with min count ≥ 2
    for _ in range(200):
        counts = rng.multinomial(N, [0.25, 0.25, 0.25, 0.25])
        if all(cnt >= 2 for cnt in counts):
            return {'levels': levels, 'counts': counts.tolist()}

    return {'levels': levels, 'counts': [5, 5, 5, 5]}


def compute(params):
    levels = params['levels']
    counts = params['counts']
    N = 20
    probs = [cnt / N for cnt in counts]
    mu_T = sum(lv * p for lv, p in zip(levels, probs))

    thresholds = []
    for t in range(3):
        w0 = round(sum(probs[:t + 1]), 4)
        w1 = round(1.0 - w0, 4)
        if w0 < 1e-9:
            mu0 = 0.0
        else:
            mu0 = round(sum(levels[i] * probs[i] for i in range(t + 1)) / w0, 4)
        if w1 < 1e-9:
            mu1 = 0.0
        else:
            mu1 = round((mu_T - w0 * mu0) / w1, 4)
        sigma2_B = round(w0 * w1 * (mu0 - mu1) ** 2, 4)
        thresholds.append({
            't': t,
            'w0': w0,
            'w1': w1,
            'mu0': mu0,
            'mu1': mu1,
            'sigma2_B': sigma2_B,
        })

    optimal_t = max(thresholds, key=lambda x: x['sigma2_B'])['t']
    return {'thresholds': thresholds, 'optimal_t': optimal_t, 'mu_T': round(mu_T, 4)}


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    levels = p['levels']
    counts = p['counts']
    N = 20
    probs = [cnt / N for cnt in counts]
    thresholds = c['thresholds']
    optimal_t = c['optimal_t']
    mu_T = c['mu_T']

    # --- shared preamble: histogram table + question + formula ---------------
    hist_rows = [
        f'| {lv} | {cnt} | {cnt}/{N} |'
        for lv, cnt in zip(levels, counts)
    ]

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: Otsu\'s Optimal Thresholding'),
        '',
        'The histogram below shows the pixel count at each intensity level'
        ' for an image with 20 pixels and 4 discrete intensity levels.',
        '',
        '| Level | Count | Probability |',
        '| --- | --- | --- |',
    ]
    lines += hist_rows
    lines += [
        '',
        'Find the optimal Otsu threshold $t^*$ that maximises the'
        ' inter-class variance $\\sigma^2_B$.',
        '',
        r'$$\sigma^2_B(t) = w_0(t)\,w_1(t)\,\bigl(\mu_0(t) - \mu_1(t)\bigr)^2$$',
        '',
        'where $w_0(t)$ is the fraction of pixels with intensity $\\leq t$ (background),'
        ' $w_1(t) = 1 - w_0(t)$ (foreground), and $\\mu_0$, $\\mu_1$ are their'
        ' respective weighted means.',
        '',
    ]

    # -------------------------------------------------------------------------
    if mode == 'blank':
        lines.append('**Optimal threshold:** `___`')

    # -------------------------------------------------------------------------
    elif mode == 'exam':
        cbox = cell_box()
        lines += [
            '**Part (a):** Complete the class statistics and inter-class variance'
            ' for each threshold.',
            '',
            f'Overall mean $\\mu_T =$ {inline_box()}',
            '',
            '| $t$ | $w_0$ | $w_1$ | $\\mu_0$ | $\\mu_1$ | $\\sigma^2_B$ |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        for th in thresholds:
            lines.append(f'| {th["t"]} | {cbox} | {cbox} | {cbox} | {cbox} | {cbox} |')
        lines += [
            '',
            '**Part (b):** Optimal threshold $t^* =$',
            '',
            answer_box(),
        ]

    # -------------------------------------------------------------------------
    elif mode == 'exam_marks':
        lines += ['---', '', '### Solution', '']
        lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Part (a) - Class statistics and inter-class variance.**',
            '',
            f'Overall mean: $\\mu_T = {mu_T}$',
            '',
            '| $t$ | $w_0$ | $w_1$ | $\\mu_0$ | $\\mu_1$ | $\\sigma^2_B$ |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        for th in thresholds:
            lines.append(
                f'| {th["t"]} | {th["w0"]} | {th["w1"]}'
                f' | {th["mu0"]} | {th["mu1"]} | {th["sigma2_B"]} |'
            )
        lines += [
            '',
            _mark(MARKS['class_stats'] + MARKS['variances']),
            '',
            '**Part (b) - Optimal threshold.**',
            '',
        ]
        for th in thresholds:
            marker = ' ← maximum' if th['t'] == optimal_t else ''
            lines.append(f'- $t={th["t"]}$: $\\sigma^2_B = {th["sigma2_B"]}${marker}')
        lines += [
            '',
            f'The inter-class variance is maximised at $t = {optimal_t}$,'
            f' so the optimal Otsu threshold is $t^* = {optimal_t}$.',
            '',
            _mark(MARKS['threshold']),
        ]

    # -------------------------------------------------------------------------
    elif mode == 'answer':
        lines += [
            '**Variance table:**',
            '',
            '| $t$ | $w_0$ | $w_1$ | $\\mu_0$ | $\\mu_1$ | $\\sigma^2_B$ |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        for th in thresholds:
            lines.append(
                f'| {th["t"]} | {th["w0"]} | {th["w1"]}'
                f' | {th["mu0"]} | {th["mu1"]} | {th["sigma2_B"]} |'
            )
        lines += [
            '',
            f'**Optimal threshold:** $t^* = {optimal_t}$',
            '',
            '> **Hint:** For each candidate threshold $t \\in \\{{0,1,2\\}}$,'
            ' split the histogram into background (levels $\\leq t$) and'
            ' foreground (levels $> t$). Compute $w_0 = \\sum_{{i\\leq t}} p_i$,'
            ' $\\mu_0 = \\frac{1}{w_0}\\sum_{{i\\leq t}} i\\,p_i$, and'
            ' $\\mu_1 = (\\mu_T - w_0\\,\\mu_0)/w_1$ where'
            ' $\\mu_T = \\sum_i i\\,p_i$ is the overall mean.'
            ' The optimal threshold maximises $\\sigma^2_B$.',
        ]

    # -------------------------------------------------------------------------
    else:  # full / full_marks
        use_marks = marks
        lines += ['---', '', '### Solution', '']
        if use_marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        # Step 1: overall mean
        mean_terms = ' + '.join(
            f'{lv} \\times \\frac{{{cnt}}}{{{N}}}'
            for lv, cnt in zip(levels, counts)
        )
        lines += [
            '**Step 1 – Compute the overall mean $\\mu_T$.**',
            '',
            f'$$\\mu_T = {mean_terms} = {mu_T}$$',
            '',
        ]

        # Step 2: class stats and variances table
        lines += [
            '**Step 2 – Compute class statistics and inter-class variance'
            ' for each threshold.**',
            '',
            '| $t$ | $w_0$ | $w_1$ | $\\mu_0$ | $\\mu_1$ | $\\sigma^2_B$ |',
            '| --- | --- | --- | --- | --- | --- |',
        ]
        for th in thresholds:
            lines.append(
                f'| {th["t"]} | {th["w0"]} | {th["w1"]}'
                f' | {th["mu0"]} | {th["mu1"]} | {th["sigma2_B"]} |'
            )

        # Show working for each threshold
        lines += ['', '**Detailed working:**', '']
        for th in thresholds:
            t = th['t']
            w0, w1 = th['w0'], th['w1']
            mu0, mu1, sigma2 = th['mu0'], th['mu1'], th['sigma2_B']

            bg_levels = levels[:t + 1]
            bg_counts = counts[:t + 1]
            # w0 calculation
            w0_terms = ' + '.join(
                f'\\frac{{{cnt}}}{{{N}}}' for cnt in bg_counts
            )
            # mu0 calculation
            mu0_terms = ' + '.join(
                f'{lv}\\times\\frac{{{cnt}}}{{{N}}}' for lv, cnt in zip(bg_levels, bg_counts)
            )
            lines += [
                f'*$t={t}$ (background: levels $\\leq {t}$, foreground: levels $> {t}$)*',
                '',
                f'$$w_0 = {w0_terms} = {w0}, \\quad w_1 = 1 - {w0} = {w1}$$',
                '',
                f'$$\\mu_0 = \\frac{{1}}{{{w0}}}\\bigl({mu0_terms}\\bigr) = {mu0}$$',
                '',
                f'$$\\mu_1 = \\frac{{\\mu_T - w_0\\,\\mu_0}}{{w_1}}'
                f'= \\frac{{{mu_T} - {w0}\\times{mu0}}}{{{w1}}} = {mu1}$$',
                '',
                f'$$\\sigma^2_B({t}) = {w0}\\times{w1}\\times({mu0}-{mu1})^2 = {sigma2}$$',
                '',
            ]

        if use_marks:
            lines += [_mark(MARKS['class_stats'] + MARKS['variances']), '']

        # Step 3: optimal threshold
        best = thresholds[optimal_t]
        lines += [
            '**Step 3 – Identify the optimal threshold.**',
            '',
        ]
        for th in thresholds:
            marker = ' ← maximum' if th['t'] == optimal_t else ''
            lines.append(f'- $t={th["t"]}$: $\\sigma^2_B = {th["sigma2_B"]}${marker}')
        lines += [
            '',
            f'The inter-class variance is maximised at $t = {optimal_t}$,'
            f' so the optimal Otsu threshold is $t^* = {optimal_t}$.',
        ]
        if use_marks:
            lines += ['', _mark(MARKS['threshold'])]

    return '\n'.join(lines)
