from collections import defaultdict

import networkx as nx
import numpy as np
import torch
from seqeval.metrics.v1 import _prf_divide


# adapted from https://github.com/chakki-works/seqeval
def extract_tp_actual_correct(y_true, y_pred):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)

    for type_name, (start, end), idx in y_true:
        entities_true[type_name].add((start, end, idx))
    for type_name, (start, end), idx in y_pred:
        entities_pred[type_name].add((start, end, idx))

    target_names = sorted(set(entities_true.keys()) |
                          set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(
            entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum, target_names


def flatten_for_eval(y_true, y_pred):
    """
    y_true: list of true spans list[list[spans]]
    y_pred: list of pred spans list[list[spans]]
    """

    all_true = []
    all_pred = []

    for i, (true, pred) in enumerate(zip(y_true, y_pred)):
        all_true.extend(
            [t + [i] for t in true]
        )
        all_pred.extend(
            [p + [i] for p in pred]
        )
    return all_true, all_pred


# adapted from https://github.com/chakki-works/seqeval
def compute_prf(y_true, y_pred, average='micro'):
    y_true, y_pred = flatten_for_eval(y_true, y_pred)

    pred_sum, tp_sum, true_sum, target_names = extract_tp_actual_correct(
        y_true, y_pred)

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])

    precision = _prf_divide(
        numerator=tp_sum,
        denominator=pred_sum,
        metric='precision',
        modifier='predicted',
        average=average,
        warn_for=('precision', 'recall', 'f-score'),
        zero_division='warn'
    )

    recall = _prf_divide(
        numerator=tp_sum,
        denominator=true_sum,
        metric='recall',
        modifier='true',
        average=average,
        warn_for=('precision', 'recall', 'f-score'),
        zero_division='warn'
    )

    denominator = precision + recall
    denominator[denominator == 0.] = 1
    f_score = 2 * (precision * recall) / denominator

    return {'precision': precision[0], 'recall': recall[0], 'f_score': f_score[0]}
