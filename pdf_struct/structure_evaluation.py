from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, \
    recall_score, precision_score

from pdf_struct.transition_predictor import DocumentWithFeatures, ListAction


def create_hierarchy_matrix(document: DocumentWithFeatures) -> np.array:
    # (i, j) shows the relationship (-1: no relationship, 0: same paragraph,
    # 1: same level, 2: line j is below line i)
    # Note that it will return upper triangle matrix --- (i, j) (i < j) will
    # be given but not (i, j) (i > j) except for the last column/row
    # m[-1, :] and m[:, -1] are used to denote "-1" root node, which will be
    # removed in the final step
    m = np.full((len(document.labels) + 1, len(document.labels) + 1), -1, dtype=int)
    m[0, -1] = 1
    last_i = None
    last_l = None
    last_p = None
    for i, (l, p) in enumerate(zip(document.labels, document.pointers)):
        if l == ListAction.ELIMINATE:
            continue
        if last_i is None:
            last_i, last_l, last_p = i, l, p
            continue
        if last_l == ListAction.CONTINUOUS:
            m[last_i, i] = 0
            m[np.where(m[:, last_i] == 0)[0], i] = 0
            m[np.where(m[:, last_i] == 1)[0], i] = 1
            m[np.where(m[:, last_i] == 2)[0], i] = 2
        elif last_l == ListAction.SAME_LEVEL:
            m[last_i, i] = 1
            m[np.where(m[:, last_i] == 0)[0], i] = 1
            m[np.where(m[:, last_i] == 1)[0], i] = 1
            m[np.where(m[:, last_i] == 2)[0], i] = 2
        elif last_l == ListAction.DOWN:
            m[last_i, i] = 2
            m[np.where(m[:, last_i] == 0)[0], i] = 2
            m[np.where(m[:, last_i] == 2)[0], i] = 2
        elif last_l == ListAction.UP:
            # it should work OK even when last_p == -1
            m[last_p, i] = 1
            m[np.where(m[:, last_p] == 0)[0], i] = 1
            m[np.where(m[:, last_p] == 1)[0], i] = 1
            m[np.where(m[:, last_p] == 2)[0], i] = 2
        last_i, last_l, last_p = i, l, p
    # last label will simply be ignored, which is a right behavior
    return m[:-1, :-1]


def _calc_metrics(ys_true: List[List[int]], ys_pred: List[List[int]],
                 labels: List[str]):
    ys_true = [np.array(yi) for yi in ys_true]
    ys_pred = [np.array(yi) for yi in ys_pred]

    metrics = []
    accuracies = []
    for y_pred, y_true in zip(ys_pred, ys_true):
        metrics.append([
            (precision_score(y_true == j, y_pred == j, zero_division=0),
             recall_score(y_true == j, y_pred == j),
             f1_score(y_true == j, y_pred == j))
            if np.any(y_true == j) else
            (np.nan, np.nan, np.nan)
            for j in (0, 1, 2, 3, 4)])
        accuracies.append(accuracy_score(y_true, y_pred))

    metrics = np.array(metrics)
    ys_true = np.concatenate(ys_true)
    ys_pred = np.concatenate(ys_pred)

    def _get_metrics(rel):
        return {
            'micro': {
                'precision': precision_score(ys_true == rel, ys_pred == rel),
                'recall': recall_score(ys_true == rel, ys_pred == rel),
                'f1': f1_score(ys_true == rel, ys_pred == rel)
            },
            'macro': {
                'precision': np.nanmean(metrics[:, rel, 0]),
                'recall': np.nanmean(metrics[:, rel, 1]),
                'f1': np.nanmean(metrics[:, rel, 2])
            }
        }

    ret = {label: _get_metrics(i) for i, label in enumerate(labels)}
    ret['accuracy'] = {
            'micro': accuracy_score(ys_true, ys_pred),
            'macro': np.mean(accuracies)
    }
    return ret



def evaluate_structure(documents_true: List[DocumentWithFeatures], documents_pred: List[DocumentWithFeatures]):
    ms_true = [create_hierarchy_matrix(d).flatten() for d in documents_true]
    ms_pred = [create_hierarchy_matrix(d).flatten() for d in documents_pred]
    return _calc_metrics(ms_true, ms_pred, ['same_paragraph', 'same_level', 'parent_child'])


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    width = int(np.log10(np.max(cm))) + 1
    tmpl = f'{{:>{width}}}'
    row = '|   | ' + ' | '.join(tmpl.format(n) for n in range(len(cm))) + ' |'
    print(row)
    print(f'{"|".join("-" * len(h) for h in row.split("|"))}')
    for i, cmi in enumerate(cm):
        print(f'| {i} | ' + ' | '.join(tmpl.format(c) for c in cmi) + ' |')


def evaluate_labels(documents_true: List[DocumentWithFeatures], documents_pred: List[DocumentWithFeatures]):
    ys_pred = [np.array([l.value for l in d.labels]) for d in documents_pred]
    ys_true = [np.array([l.value for l in d.labels]) for d in documents_true]
    print_confusion_matrix(np.concatenate(ys_true), np.concatenate(ys_pred))
    metrics = _calc_metrics(
        ys_true, ys_pred,
        [ListAction(0).name, ListAction(1).name, ListAction(2).name, ListAction(3).name, ListAction(4).name])
    ys_pred = [
        np.isin(yi, (ListAction.DOWN.value, ListAction.UP.value, ListAction.SAME_LEVEL.value))
        for yi in ys_pred]
    ys_true = [
        np.isin(yi, (ListAction.DOWN.value, ListAction.UP.value, ListAction.SAME_LEVEL.value))
        for yi in ys_true]
    metrics.update({
        'paragraph_boundary': {
            'micro': {
                'precision': precision_score(np.concatenate(ys_true), np.concatenate(ys_pred)),
                'recall': recall_score(np.concatenate(ys_true), np.concatenate(ys_pred)),
                'f1': f1_score(np.concatenate(ys_true), np.concatenate(ys_pred))
            },
            'macro': {
                'precision': np.mean([
                    precision_score(y_true, y_pred)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
                'recall': np.mean([
                    recall_score(y_true, y_pred)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
                'f1': np.mean([
                    f1_score(y_true, y_pred)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
            }
        }
    })
    return metrics
