"""K-means clustering (one iteration) question generator.

Given 8 pixel intensity values and 2 initial centroids, perform one full
iteration of k-means:
  1. Assign each pixel to the nearest centroid (|pixel - centroid|).
     Ties are broken in favour of cluster 1.
  2. Compute new centroids as the mean of each cluster (rounded to 2 dp).

k = 2 always (two well-separated initial centroids keep the question tractable).
"""
import numpy as np

MARKS = {
    'assignment':    5,   # correct cluster assignment for all 8 pixels
    'new_centroids': 5,   # correct new centroid values
}


def _mark(n):
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)

    while True:
        pixels = rng.integers(0, 256, size=8).tolist()
        c1 = int(rng.integers(40, 101))   # c1 in [40, 100]
        c2 = int(rng.integers(150, 221))  # c2 in [150, 220]

        # Assign each pixel to the nearest centroid (ties -> cluster 1)
        assignments = []
        for px in pixels:
            d1 = abs(px - c1)
            d2 = abs(px - c2)
            assignments.append(1 if d1 <= d2 else 2)

        # Retry until both clusters are non-empty
        if 1 in assignments and 2 in assignments:
            break

    return {'pixels': pixels, 'centroids': [c1, c2]}


def compute(params):
    pixels = params['pixels']
    c1, c2 = params['centroids']

    assignments = []
    for px in pixels:
        d1 = abs(px - c1)
        d2 = abs(px - c2)
        assignments.append(1 if d1 <= d2 else 2)

    cluster1 = [px for px, a in zip(pixels, assignments) if a == 1]
    cluster2 = [px for px, a in zip(pixels, assignments) if a == 2]

    nc1 = round(sum(cluster1) / len(cluster1), 2)
    nc2 = round(sum(cluster2) / len(cluster2), 2)

    return {
        'assignments':   assignments,
        'new_centroids': [nc1, nc2],
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c    = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    pixels  = p['pixels']
    c1, c2  = p['centroids']
    assignments = c['assignments']
    nc1, nc2    = c['new_centroids']

    # Pixel intensity display string
    pix_latex = ',\\; '.join(f'p_{{{i+1}}}={v}' for i, v in enumerate(pixels))

    lines = [
        f'## Question{num_str}: K-Means Clustering (One Iteration)',
        '',
        'Perform **one iteration** of k-means clustering ($k=2$) on the following'
        ' 8 pixel intensity values.',
        '',
        f'**Pixel intensities:** ${pix_latex}$',
        '',
        f'**Initial centroids:** $c_1 = {c1},\\quad c_2 = {c2}$',
        '',
        '**Step 1:** Assign each pixel to the nearest centroid using'
        ' $|p_i - c_k|$. Break ties in favour of cluster 1.',
        '',
        '**Step 2:** Compute the new centroid for each cluster as the mean'
        ' of its assigned pixels (round to 2 decimal places).',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Cluster assignments:** `_______________`',
            '',
            '**New centroid $c_1$:** `___`'
            '$\\quad$'
            '**New centroid $c_2$:** `___`',
        ]

    elif mode == 'answer':
        cl1_idx   = [i + 1 for i, a in enumerate(assignments) if a == 1]
        cl2_idx   = [i + 1 for i, a in enumerate(assignments) if a == 2]
        cl1_names = ', '.join(f'$p_{{{i}}}$' for i in cl1_idx)
        cl2_names = ', '.join(f'$p_{{{i}}}$' for i in cl2_idx)

        lines += [
            f'**Cluster 1:** {cl1_names}',
            '',
            f'**Cluster 2:** {cl2_names}',
            '',
            f'**New centroids:** $c_1 = {nc1},\\quad c_2 = {nc2}$',
            '',
            '> **Hint:** Compute $|p_i - c_1|$ and $|p_i - c_2|$ for each pixel;'
            ' assign to the cluster with the smaller distance (ties go to cluster 1).'
            ' The new centroid equals the mean of all pixels assigned to that cluster.',
        ]

    else:   # full / full_marks
        lines += ['---', '', '### Solution', '']
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']

        # Step 1 -- assignment table
        lines += [
            '**Step 1 - Assign pixels to the nearest centroid.**',
            '',
            f'| Pixel | Value | $|p-c_1|$ ($c_1={c1}$) | $|p-c_2|$ ($c_2={c2}$) | Cluster |',
            '|-------|-------|------------------------|------------------------|---------|',
        ]
        for i, (pv, a) in enumerate(zip(pixels, assignments)):
            d1 = abs(pv - c1)
            d2 = abs(pv - c2)
            lines.append(f'| $p_{{{i+1}}}$ | ${pv}$ | ${d1}$ | ${d2}$ | $C_{{{a}}}$ |')

        if marks:
            lines += ['', _mark(MARKS['assignment'])]

        # Step 2 -- new centroids
        cl1_vals  = [pv for pv, a in zip(pixels, assignments) if a == 1]
        cl2_vals  = [pv for pv, a in zip(pixels, assignments) if a == 2]
        cl1_idx   = [i + 1 for i, a in enumerate(assignments) if a == 1]
        cl2_idx   = [i + 1 for i, a in enumerate(assignments) if a == 2]

        cl1_pnames = ',\\; '.join(f'p_{{{i}}}' for i in cl1_idx)
        cl2_pnames = ',\\; '.join(f'p_{{{i}}}' for i in cl2_idx)
        cl1_vsum   = ' + '.join(str(v) for v in cl1_vals)
        cl2_vsum   = ' + '.join(str(v) for v in cl2_vals)

        lines += [
            '',
            '**Step 2 - Compute new centroids.**',
            '',
            f'Cluster 1 — pixels $\\{{{cl1_pnames}\\}}$:',
            '',
            f'$$c_1^{{\\rm new}} = \\frac{{{cl1_vsum}}}{{{len(cl1_vals)}}} = {nc1}$$',
            '',
            f'Cluster 2 — pixels $\\{{{cl2_pnames}\\}}$:',
            '',
            f'$$c_2^{{\\rm new}} = \\frac{{{cl2_vsum}}}{{{len(cl2_vals)}}} = {nc2}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['new_centroids'])]

    return '\n'.join(lines)
