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
                'precision': precision_score(ys_true == rel, ys_pred == rel, zero_division=0),
                'recall': recall_score(ys_true == rel, ys_pred == rel),
                'f1': f1_score(ys_true == rel, ys_pred == rel, zero_division=0)
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
    # Since hierarchy matrix is a upper triangle matrix, feeding full matrix
    # with create_hierarchy_matrix(d).flatten() will give higher accuracy than it should do
    ms_true = [
        np.concatenate([r[i + 1:] for i, r in enumerate(create_hierarchy_matrix(d))])
        for d in documents_true]
    ms_pred = [
        np.concatenate([r[i + 1:] for i, r in enumerate(create_hierarchy_matrix(d))])
        for d in documents_pred]
    return _calc_metrics(ms_true, ms_pred, ['same_paragraph', 'same_level', 'parent_child'])


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    list_actions = [l for l in sorted(ListAction, key=lambda l: l.value) if l.value >= 0]
    index_mapping = {
        label_idx: ind for ind, label_idx in enumerate(sorted(set(y_true) | set(y_pred)))}
    tmpls = []
    for la in list_actions:
        if la.value in index_mapping:
            width_num = int(np.log10(max(1, np.max(cm[:, index_mapping[la.value]])))) + 1
            width = max(width_num, len(la.name))
        else:
            width = len(la.name)
        tmpls.append(f'{{:>{width}}}')
    tmpl = f' {{:<{max((len(la.name) for la in list_actions))}}} | ' + ' | '.join(tmpls) + ' |'

    row = tmpl.replace('>', '<').format(*([''] + [la.name for la in list_actions]))
    print(row)
    print(f'{"|".join("-" * len(h) for h in row.split("|"))}')
    for la in list_actions:
        print(tmpl.format(*(
            [la.name] +
            [cm[index_mapping[la.value]][index_mapping[la2.value]]
             if la.value in index_mapping and la2.value in index_mapping else 0
             for la2 in list_actions])))
    print('(i-th row and j-th column entry indicates the number of samples with '
          'true label being i-th class and predicted label being j-th class.)')


def evaluate_labels(documents_true: List[DocumentWithFeatures], documents_pred: List[DocumentWithFeatures], confusion_matrix=True):
    ys_pred = [np.array([l.value for l in d.labels]) for d in documents_pred]
    ys_true = [np.array([l.value for l in d.labels]) for d in documents_true]
    if confusion_matrix:
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
                'precision': precision_score(np.concatenate(ys_true), np.concatenate(ys_pred), zero_division=0),
                'recall': recall_score(np.concatenate(ys_true), np.concatenate(ys_pred)),
                'f1': f1_score(np.concatenate(ys_true), np.concatenate(ys_pred))
            },
            'macro': {
                'precision': np.mean([
                    precision_score(y_true, y_pred, zero_division=0)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
                'recall': np.mean([
                    recall_score(y_true, y_pred)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
                'f1': np.mean([
                    f1_score(y_true, y_pred, zero_division=0)
                    for y_true, y_pred in zip(ys_true, ys_pred)
                    if np.any(y_true)]),
            }
        }
    })
    return metrics
