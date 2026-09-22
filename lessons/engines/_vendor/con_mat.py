"""Generator for Confusion Matrix Metrics questions (3 classes).

3x3 confusion matrix with labels A, B, C (rows=true, cols=predicted).
For each class: Accuracy, Precision, Recall.

Modes:
    'blank'  - question only
    'answer' - answer + hint
    'full'   - step-by-step solution

Pass marks=True to render() to show mark allocation per step (full mode only).
Total marks sum to 10.
"""
import numpy as np
from ._utils import pmatrix

# Mark allocation per step — must sum to 10
MARKS = {
    'counts': 4,    # TP/FP/FN/TN for all classes
    'accuracy': 2,  # accuracy per class
    'precision': 2, # precision per class
    'recall': 2,    # recall per class
}


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    mat = rng.integers(5, 26, size=(3, 3)).tolist()
    return {'mat': mat}


def compute(params):
    mat = np.array(params['mat'])
    total = int(mat.sum())
    results = []
    for i in range(3):
        TP = int(mat[i, i])
        FP = int(mat[:, i].sum()) - TP
        FN = int(mat[i, :].sum()) - TP
        TN = total - TP - FP - FN
        acc = round((TP + TN) / (TP + TN + FP + FN), 2)
        pre = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0.0
        rec = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0.0
        results.append({'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
                        'acc': acc, 'pre': pre, 'rec': rec})
    return {'results': results, 'total': total}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''
    mat = np.array(p['mat'])
    labels = ['A', 'B', 'C']

    mat_str = pmatrix(mat)
    results = c['results']

    def _res_mat():
        rows = []
        for r in results:
            rows.append(f'{r["acc"]} & {r["pre"]} & {r["rec"]}')
        inner = r' \\ '.join(rows)
        return r'\begin{pmatrix}' + inner + r'\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Confusion Matrix Metrics',
        '',
        'The following $3 \\times 3$ confusion matrix is given for classes A, B, C'
        ' (rows = true class, columns = predicted class):',
        '',
        '| | Pred A | Pred B | Pred C |',
        '|---|---|---|---|',
        f'| **True A** | {mat[0,0]} | {mat[0,1]} | {mat[0,2]} |',
        f'| **True B** | {mat[1,0]} | {mat[1,1]} | {mat[1,2]} |',
        f'| **True C** | {mat[2,0]} | {mat[2,1]} | {mat[2,2]} |',
        '',
        'For **each class** compute the **Accuracy** ($Acc_i$), **Precision** ($Pre_i$),'
        ' and **Recall** ($Rec_i$) using the one-vs-rest approach.',
        'Give answers rounded to 2 d.p. as a $3 \\times 3$ matrix'
        ' $[[Acc_A, Pre_A, Rec_A],[Acc_B, Pre_B, Rec_B],[Acc_C, Pre_C, Rec_C]]$.',
        '',
    ]

    if mode == 'blank':
        lines.append('**Answer matrix:** `_______________`')

    elif mode == 'answer':
        lines += [
            f'**Answer:** ${_res_mat()}$',
            '',
            '> **Hint:** For each class $i$ (one-vs-rest):'
            ' $TP$ = correct predictions for class $i$,'
            ' $FP$ = other classes predicted as $i$,'
            ' $FN$ = class $i$ predicted as other,'
            ' $TN$ = everything else.'
            ' Then $Acc=(TP+TN)/N$, $Pre=TP/(TP+FP)$, $Rec=TP/(TP+FN)$.',
        ]

    else:
        total = c['total']
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            f'Total samples: $N = {total}$',
            '',
        ]
        for i, (lbl, r) in enumerate(zip(labels, results)):
            lines += [
                f'**Class {lbl} (one-vs-rest):**',
                '',
                f'- $TP = {r["TP"]}$, $FP = {r["FP"]}$, $FN = {r["FN"]}$, $TN = {r["TN"]}$',
                f'- $Acc_{lbl} = (TP+TN)/N = ({r["TP"]}+{r["TN"]})/{total} = {r["acc"]}$',
                f'- $Pre_{lbl} = TP/(TP+FP) = {r["TP"]}/({r["TP"]}+{r["FP"]}) = {r["pre"]}$',
                f'- $Rec_{lbl} = TP/(TP+FN) = {r["TP"]}/({r["TP"]}+{r["FN"]}) = {r["rec"]}$',
                '',
            ]
        if marks:
            lines += [_mark(MARKS['counts']), '']
        lines += [
            '**Answer matrix:**',
            '',
            f'$$\\begin{{pmatrix}}'
            f'{results[0]["acc"]} & {results[0]["pre"]} & {results[0]["rec"]} \\\\'
            f' {results[1]["acc"]} & {results[1]["pre"]} & {results[1]["rec"]} \\\\'
            f' {results[2]["acc"]} & {results[2]["pre"]} & {results[2]["rec"]}'
            f'\\end{{pmatrix}}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['accuracy']), '', _mark(MARKS['precision']), '', _mark(MARKS['recall'])]

    return '\n'.join(lines)
