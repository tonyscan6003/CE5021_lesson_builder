"""Generator for 1D convolution questions.

Mathematical convolution:
    (x * h)[n] = sum_{k=-inf}^{inf} x[k] h[n-k]

Output mode:
    'full'   – question text + complete step-by-step solution
    'answer' – question text + final answer only
    'blank'  – question text only
"""

import numpy as np
from ._utils import (
    fmt, cell, vec_list_str, index_range_str,
    pmatrix, md_table, product_sum_latex,
)
from ._exam_boxes import answer_box, inline_box

# Mark allocation per step — must sum to 10
MARKS = {
    'output_length': 2,  # output range identification
    'convolution': 8,    # full convolution computation
}

EXAM_COMPATIBLE = True


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


# ---------------------------------------------------------------------------
# Parameter generation
# ---------------------------------------------------------------------------

def generate_params(
    seed=None,
    x_len: int = 3,
    h_len: int = 3,
    x_start: int = -1,
    h_start: int = -1,
    val_lo: int = -3,
    val_hi: int = 3,
) -> dict:
    """Return a parameter dict for one 1D convolution question."""
    rng = np.random.default_rng(seed)

    def rand_nonzero(n: int):
        while True:
            v = rng.integers(val_lo, val_hi + 1, size=n)
            if np.any(v != 0):
                return v

    x_vals = rand_nonzero(x_len)
    h_vals = rand_nonzero(h_len)

    x_end = x_start + x_len - 1
    h_end = h_start + h_len - 1

    return {
        'x_vals': x_vals,
        'h_vals': h_vals,
        'x_start': x_start,
        'x_end': x_end,
        'h_start': h_start,
        'h_end': h_end,
        'n_out_start': x_start + h_start,
        'n_out_end': x_end + h_end,
    }


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def compute(params: dict) -> dict:
    """Compute convolution result and all intermediate values needed for
    the full step-by-step solution."""
    x_vals = params['x_vals']
    h_vals = params['h_vals']
    x_start, x_end = params['x_start'], params['x_end']
    h_start, h_end = params['h_start'], params['h_end']
    n_out_start, n_out_end = params['n_out_start'], params['n_out_end']

    # numpy gives the correct full discrete linear convolution
    result = np.convolve(x_vals, h_vals, mode='full').astype(int)
    n_out_range = list(range(n_out_start, n_out_end + 1))

    # Display k range: one padding element on each side of the output range
    k_min = n_out_start - 1
    k_max = n_out_end + 1
    k_range = list(range(k_min, k_max + 1))

    def get_x(k: int) -> int:
        idx = k - x_start
        return int(x_vals[idx]) if 0 <= idx < len(x_vals) else 0

    def get_h(arg: int) -> int:
        idx = arg - h_start
        return int(h_vals[idx]) if 0 <= idx < len(h_vals) else 0

    x_extended = [get_x(k) for k in k_range]

    # h[n-k] at each k, for each output n
    h_shifted = {n: [get_h(n - k) for k in k_range] for n in n_out_range}

    # For each n: the k range where x[k] or h[n-k] could be non-zero
    def active_k_range(n: int):
        k_lo = min(x_start, n - h_end)
        k_hi = max(x_end, n - h_start)
        return list(range(k_lo, k_hi + 1))

    # For each n: k evaluations to display (range where h[n-k] could be non-zero,
    # plus one extra k on each side to show the zero boundary)
    def h_eval_ks(n: int):
        return list(range(n - h_end - 1, n - h_start + 2))

    return {
        'result': result,
        'n_out_range': n_out_range,
        'k_range': k_range,
        'x_extended': x_extended,
        'h_shifted': h_shifted,
        'get_x': get_x,
        'get_h': get_h,
        'active_k_range': active_k_range,
        'h_eval_ks': h_eval_ks,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(params: dict, computed: dict, mode: str = 'full',
           question_num=None, marks=False) -> str:
    """Return the question (and optionally solution) as a markdown string."""
    p, c = params, computed
    result = c['result'].tolist()
    out_range = p['n_out_start'], p['n_out_end']
    h_range = p['h_start'], p['h_end']
    x_range = p['x_start'], p['x_end']

    out_len = len(c['n_out_range'])
    num_str = f' {question_num}' if question_num is not None else ''

    lines = [
        (f'## Q{question_num}' if (mode == 'exam' and question_num is not None)
         else f'## Question{num_str}: 1D Convolution (Mathematical)'),
        '',
        'Given the 1D convolution equation',
        '',
        r'$$',
        r'(x * h)[n] = \sum_{k=-\infty}^{\infty} x[k]\, h[n-k]',
        r'$$',
        '',
        (f'Perform convolution between the sequence '
         f'$x[n] = {vec_list_str(p["x_vals"])}$ '
         f'and the kernel $h[n] = {vec_list_str(p["h_vals"])}$ '
         f'defined at $n \\in {index_range_str(*h_range)}$.'),
        '',
        (f'Give your answer as a $1 \\times {out_len}$ matrix where the elements '
         f'of the matrix are the result of convolution $(x * h)[n]$ '
         f'for $n \\in {index_range_str(*out_range)}$:'),
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer:** `_______________`')
    elif mode == 'exam':
        lines += [
            '**Part (a):** State the output index range $n$ (i.e. the values'
            ' of $n$ for which $(x*h)[n]$ is non-zero).',
            '',
            f'$n \\in$ [{inline_box(width_pt=90)}, {inline_box(width_pt=90)}]',
            '',
            '**Part (b):** Compute the convolution result.',
            '',
            answer_box(),
        ]
    elif mode == 'exam_marks':
        lines += [
            '---',
            '',
            '### Solution',
            '',
            f'*(Total: {sum(MARKS.values())} marks)*',
            '',
            '**Part (a) - Output index range.**',
            '',
            f'$$n \\in {index_range_str(*out_range)}$$',
            '',
            _mark(MARKS['output_length']),
            '',
            '**Part (b) - Convolution.**',
            '',
            _solution(params, computed),
            '',
            _mark(MARKS['convolution']),
        ]
    elif mode == 'answer':
        lines += [
            '**Answer:**',
            '',
            f'$$C = {pmatrix(result)}$$',
            '',
            r'> **Hint:** Rewrite both sequences in terms of dummy variable $k$.'
            r' For each output index $n$, flip $h$ to get $h[-k]$, shift it to'
            r' $h[n-k]$, multiply element-wise with $x[k]$, and sum over all $k$.',
        ]
    else:
        lines += [
            '---',
            '',
            '### Solution',
            '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [_solution(params, computed)]
        if marks:
            lines += ['', _mark(MARKS['output_length']), '', _mark(MARKS['convolution'])]

    return '\n'.join(lines)


def _solution(params: dict, computed: dict) -> str:
    p, c = params, computed
    k_range = c['k_range']
    x_ext = c['x_extended']
    n_range = c['n_out_range']
    h_shifted = c['h_shifted']
    result = c['result'].tolist()

    # Table header row (shared across all tables)
    k_headers = [''] + [f'$k={k}$' for k in k_range]

    def make_x_row():
        return ['$x[k]$'] + [cell(v) for v in x_ext]

    def make_h_row(n):
        label = f'$h[{n}-k]$' if n != 0 else r'$h[-k]$'
        return [label] + [cell(v) for v in h_shifted[n]]

    x_row = make_x_row()

    lines = [
        '**Step 1:** Rewrite input and kernel in terms of the dummy variable $k$:',
        '',
        f'$x[k] = {vec_list_str(p["x_vals"])}$, '
        f'$k \\in {index_range_str(p["x_start"], p["x_end"])}$',
        '',
        f'$h[k] = {vec_list_str(p["h_vals"])}$, '
        f'$k \\in {index_range_str(p["h_start"], p["h_end"])}$',
        '',
        'Both sequences are zero outside their defined ranges.',
        '',
        md_table(k_headers, [x_row]),
        '',
        (f'The output range is $n \\in {index_range_str(p["n_out_start"], p["n_out_end"])}$. '
         f'We evaluate over $k \\in {index_range_str(min(k_range), max(k_range))}$.'),
        '',
    ]

    for step_idx, n in enumerate(n_range):
        step_num = step_idx + 2
        result_val = result[step_idx]
        h_row = h_shifted[n]

        lines += [
            f'**Step {step_num}: $n = {n}$**',
            '',
            f'Reversed/shifted kernel: $h(n-k) = h({n}-k)$',
            '',
        ]

        # Show individual k evaluations for the h-support range
        eval_lines = []
        for k in c['h_eval_ks'](n):
            arg = n - k
            hv = c['get_h'](arg)
            h_support = p['h_start'] <= arg <= p['h_end']
            in_note = '' if h_support else ' (outside defined range, so = 0)'
            eval_lines.append(
                f'- $k = {k}$: $h({n} - ({k})) = h({arg}) = {hv}${in_note}'
            )
        lines += eval_lines
        lines.append('')

        lines += [
            md_table(k_headers, [x_row, make_h_row(n)]),
            '',
        ]

        # Computation formula: show all terms in the "active" k range
        active_ks = c['active_k_range'](n)
        xs = [c['get_x'](k) for k in active_ks]
        hs = [c['get_h'](n - k) for k in active_ks]
        k_lo, k_hi = min(active_ks), max(active_ks)

        term_str = product_sum_latex(xs, hs)
        lines += [
            f'$$(x * h)[{n}] = \\sum_{{k={k_lo}}}^{{{k_hi}}} x[k]\\,h[{n}-k] '
            f'= {term_str} = {result_val}$$',
            '',
        ]

    # Summary table
    lines += [
        '**Summary table:**',
        '',
    ]
    summary_rows = [x_row]
    for n in n_range:
        summary_rows.append(make_h_row(n))

    # Result row: blank for padding k values, values for n_out_range positions
    result_cells = [''] + [cell(v) for v in result] + ['']
    summary_rows.append(['$(x * h)[n]$'] + result_cells)

    lines += [
        md_table(k_headers, summary_rows),
        '',
        (f'**Result:** $(x * h)[n] = {pmatrix(result)}$ '
         f'for $n \\in {index_range_str(p["n_out_start"], p["n_out_end"])}$'),
    ]

    return '\n'.join(lines)
