"""Generator for Object Detection Precision & Recall questions (two parts).

Random 5x4 IoU and Confidence matrices.
T_IoU=0.50, T_C=0.20 (fixed). Random N_GT (20-35).
Part (a): Precision P = TP/(TP+FP).
Part (b): Recall R = TP/(TP+FN).

Modes:
    'blank'  - question only
    'answer' - all part answers + hint
    'full'   - step-by-step solution for all parts
"""
import numpy as np
from ._utils import pmatrix


# Mark allocation per step — must sum to 10
MARKS = {
    'counts': 4,    # TP/FP/FN indicator matrices and counts
    'precision': 3, # precision computation
    'recall': 3,    # recall computation
}


def _mark(n):
    """Return a mark annotation line."""
    return f'> *({n} mark{"s" if n != 1 else ""})*'


def generate_params(seed=None):
    rng = np.random.default_rng(seed)
    iou = [[round(float(rng.uniform(0.1, 0.9)), 2) for _ in range(4)] for _ in range(5)]
    conf = [[round(float(rng.uniform(0.1, 0.9)), 2) for _ in range(4)] for _ in range(5)]
    N_GT = int(rng.integers(20, 36))
    return {'iou': iou, 'conf': conf, 'N_GT': N_GT, 'T_IoU': 0.50, 'T_C': 0.20}


def compute(params):
    iou = np.array(params['iou'])
    conf = np.array(params['conf'])
    T_IoU = params['T_IoU']
    T_C = params['T_C']
    N_GT = params['N_GT']

    I_C = conf > T_C
    I_IoU = iou > T_IoU
    TP = int(np.sum(I_C & I_IoU))
    FP = int(np.sum(I_C & ~I_IoU))
    FN = N_GT - TP

    P = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0.0
    R = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0.0

    return {
        'TP': TP, 'FP': FP, 'FN': FN,
        'P': P, 'R': R,
        'I_C': I_C.tolist(), 'I_IoU': I_IoU.tolist(),
    }


def render(params, computed, mode='full', question_num=None, marks=False):
    p, c = params, computed
    num_str = f' {question_num}' if question_num is not None else ''

    def _mat_str(mat):
        rows = [' & '.join(str(v) for v in row) for row in mat]
        return r'\begin{pmatrix}' + r' \\ '.join(rows) + r'\end{pmatrix}'

    def _bool_mat(mat):
        rows = [' & '.join('1' if v else '0' for v in row) for row in mat]
        return r'\begin{pmatrix}' + r' \\ '.join(rows) + r'\end{pmatrix}'

    lines = [
        f'## Question{num_str}: Object Detection Precision and Recall',
        '',
        'For a set of detections, the following $5 \\times 4$ matrices give the'
        ' IoU and Confidence scores (5 predicted boxes, 4 ground truth boxes):',
        '',
        f'$$\\text{{IoU}} = {_mat_str(p["iou"])}$$',
        '',
        f'$$\\text{{Conf}} = {_mat_str(p["conf"])}$$',
        '',
        f'Thresholds: $T_{{\\text{{IoU}}}} = {p["T_IoU"]}$, $T_C = {p["T_C"]}$.',
        f'Number of ground truth objects: $N_{{GT}} = {p["N_GT"]}$.',
        '',
        'Using indicator matrices $I_C = (\\text{Conf} > T_C)$ and $I_{{\\text{{IoU}}}} = (\\text{{IoU}} > T_{{\\text{{IoU}}}})$:',
        '',
        '$$TP = \\sum (I_C \\wedge I_{\\text{IoU}}), \\quad'
        ' FP = \\sum (I_C \\wedge \\neg I_{\\text{IoU}}), \\quad'
        ' FN = N_{GT} - TP$$',
        '',
        '**Part (a):** Compute the **Precision** $P = TP/(TP+FP)$. (Round to 2 d.p.)',
        '',
        '**Part (b):** Compute the **Recall** $R = TP/(TP+FN)$. (Round to 2 d.p.)',
        '',
    ]

    if mode == 'blank':
        lines += [
            '**Part (a):** $P =$ `___`',
            '',
            '**Part (b):** $R =$ `___`',
        ]

    elif mode == 'answer':
        lines += [
            f'**Part (a):** $P = {c["P"]}$',
            '',
            f'**Part (b):** $R = {c["R"]}$',
            '',
            f'*(TP={c["TP"]}, FP={c["FP"]}, FN={c["FN"]})*',
            '',
            '> **Hint:** Build boolean indicator matrices for confidence and IoU thresholds.'
            ' TP = detections passing both; FP = passing confidence but not IoU; FN = GT objects not detected.',
        ]

    else:
        lines += [
            '---', '',
            '### Solution', '',
        ]
        if marks:
            lines += [f'*(Total: {sum(MARKS.values())} marks)*', '']
        lines += [
            '**Step 1 - Build indicator matrices.**',
            '',
            f'$$I_C = {_bool_mat(c["I_C"])}$$',
            '',
            f'$$I_{{\\text{{IoU}}}} = {_bool_mat(c["I_IoU"])}$$',
            '',
            '**Step 2 - Count TP, FP, FN.**',
            '',
            f'$$TP = {c["TP"]}, \\quad FP = {c["FP"]}, \\quad FN = {p["N_GT"]} - {c["TP"]} = {c["FN"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['counts'])]
        lines += [
            '',
            '**Part (a) - Precision:**',
            '',
            f'$$P = \\frac{{TP}}{{TP+FP}} = \\frac{{{c["TP"]}}}{{{c["TP"]}+{c["FP"]}}} = {c["P"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['precision'])]
        lines += [
            '',
            '**Part (b) - Recall:**',
            '',
            f'$$R = \\frac{{TP}}{{TP+FN}} = \\frac{{{c["TP"]}}}{{{c["TP"]}+{c["FN"]}}} = {c["R"]}$$',
        ]
        if marks:
            lines += ['', _mark(MARKS['recall'])]

    return '\n'.join(lines)
