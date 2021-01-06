import copy
import random
from collections import Counter
from typing import List

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold

from pdf_struct.structure_reconstructor import construct_hierarchy
from pdf_struct.transition_labels import ListAction, DocumentWithFeatures


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    width = int(np.log10(np.max(cm))) + 1
    tmpl = f'{{:>{width}}}'
    row = '|   | ' + ' | '.join(tmpl.format(n) for n in range(len(cm))) + ' |'
    print(row)
    print(f'|{"|".join("-" * len(h) for h in row.split("|"))}|')
    for i, cmi in enumerate(cm):
        print(f'| {i} | ' + ' | '.join(tmpl.format(c) for c in cmi) + ' |')


def levels_to_pointer(levels: List[int]):
    # this is here to convert old level-based notation to pointer notation
    latest_levels = [0]
    latest_pointers = [-1]
    pointers = []
    for i, l in enumerate(levels):
        assert len(latest_levels) == len(latest_pointers)
        if latest_levels[-1] > l:
            # UP must have happened
            while latest_levels[-1] > l:
                latest_levels.pop()
                latest_pointers.pop()
            if latest_levels[-1] == l:
                pointers.append(latest_pointers[-1])
                latest_pointers[-1] = i
                continue
        if latest_levels[-1] < l:
            # DOWN must have happened
            latest_levels.append(l)
            latest_pointers.append(i)
        else:
            # consecutive or continuous lines
            latest_pointers[-1] = -1 if i == 0 else i
        pointers.append(None)
    pointers.append(-1)
    pointers = pointers[1:]
    assert len(levels) == len(pointers)
    return pointers


def k_fold_train_predict(documents: list, n_splits: int=5) -> List[DocumentWithFeatures]:
    print(f'Extracted {sum(map(lambda d: len(d.feats), documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    test_indices = []
    predicted_documents = []
    random.seed(123)
    np.random.seed(123)
    for train_index, test_index in KFold(n_splits=n_splits).split(X=documents):
        X_train = np.array(sum([documents[j].feats for j in train_index], []), dtype=np.float64)
        y_train = np.array([l.value for j in train_index for l in documents[j].labels], dtype=int)
        X_test = np.array(sum([documents[j].feats for j in test_index], []), dtype=np.float64)

        clf = RandomForestClassifier().fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        test_indices.append(test_index)
        cum_j = 0
        for j in test_index:
            d = copy.deepcopy(documents[j])
            d.labels = [ListAction(yi) for yi in y_pred[cum_j:cum_j + len(documents[j].feats)]]
            cum_j += len(documents[j].feats)
            d.pointers = levels_to_pointer(construct_hierarchy(d, d.labels))
            predicted_documents.append(d)
    predicted_documents = [
        predicted_documents[j] for j in np.argsort(np.concatenate(test_indices))]
    y_pred = np.array([l.value for d in predicted_documents for l in d.labels])
    y_true = np.array([l.value for d in documents for l in d.labels])
    print(f'Done prediction. Accuracy={accuracy_score(y_true, y_pred)}.')
    print_confusion_matrix(y_true, y_pred)

    return predicted_documents
