import numpy as np
from typing import List
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

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


def evaluate_structure(documents_true: List[DocumentWithFeatures], documents_pred: List[DocumentWithFeatures]):
    ms_true = [create_hierarchy_matrix(d).flatten() for d in documents_true]
    ms_pred = [create_hierarchy_matrix(d).flatten() for d in documents_pred]
    metrics = []
    accuracies = []
    for m_true, m_pred in zip(ms_true, ms_pred):
        metrics.append([
            (precision_score(m_true == j, m_pred == j, zero_division=0),
             recall_score(m_true == j, m_pred == j),
             f1_score(m_true == j, m_pred == j))
            if np.any(m_true == j) else
            (np.nan, np.nan, np.nan)
            for j in (0, 1, 2)])
        accuracies.append(accuracy_score(m_true, m_pred))

    metrics = np.array(metrics)
    ms_true = np.concatenate(ms_true)
    ms_pred = np.concatenate(ms_pred)

    def _get_metrics(rel):
        return {
            'micro': {
                'precision': precision_score(ms_true == rel, ms_pred == rel),
                'recall': recall_score(ms_true == rel, ms_pred == rel),
                'f1': f1_score(ms_true == rel, ms_pred == rel)
            },
            'macro': {
                'precision': np.nanmean(metrics[:, rel, 0]),
                'recall': np.nanmean(metrics[:, rel, 1]),
                'f1': np.nanmean(metrics[:, rel, 2])
            }
        }

    return {
        'same_paragraph': _get_metrics(0),
        'same_level': _get_metrics(1),
        'parent_child': _get_metrics(2),
        'accuracy': {
            'micro': accuracy_score(ms_true, ms_pred),
            'macro': np.mean(accuracies)
        }
    }
