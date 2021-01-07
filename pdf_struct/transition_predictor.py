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


def k_fold_train_predict(documents: List[DocumentWithFeatures], n_splits: int=5) -> List[DocumentWithFeatures]:
    print(f'Extracted {sum(map(lambda d: len(d.feats), documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    test_indices = []
    predicted_documents = []
    random.seed(123)
    np.random.seed(123)
    for train_index, test_index in KFold(n_splits=n_splits).split(X=documents):
        test_indices.append(test_index)

        # First, classify transition between consecutive lines
        X_train = np.array(sum([documents[j].feats for j in train_index], []), dtype=np.float64)
        y_train = np.array([l.value for j in train_index for l in documents[j].labels], dtype=int)
        X_test = np.array(sum([documents[j].feats for j in test_index], []), dtype=np.float64)

        clf = RandomForestClassifier().fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Next, predict pointers
        X_train = np.array(
            [f[2] for j in train_index for f in documents[j].pointer_feats],
            dtype=np.float64)
        y_train = np.array(
            [f[0] == documents[j].pointers[f[1]] for j in train_index for f in documents[j].pointer_feats],
            dtype=int)
        clf_ptr = RandomForestClassifier().fit(X_train, y_train)

        cum_j = 0
        for doc_idx in test_index:
            d = copy.deepcopy(documents[doc_idx])
            d.labels = [ListAction(yi) for yi in y_pred[cum_j:cum_j + len(documents[doc_idx].feats)]]

            d.feature_extractor.init_state()
            for i in range(len(d.text_boxes)):
                tb1 = d.text_boxes[i - 1] if i != 0 else None
                tb2 = d.text_boxes[i]
                if d.labels[i] == ListAction.ELIMINATE:
                    tb3 = d.text_boxes[i + 1] if i + 1 < len(d.text_boxes) else None
                    tb4 = d.text_boxes[i + 2] if i + 2 < len(d.text_boxes) else None
                else:
                    tb3, tb4 = None, None
                    for j in range(i + 1, len(d.text_boxes)):
                        if d.labels[j] != ListAction.ELIMINATE:
                            tb3 = d.text_boxes[j]
                            break
                    for j in range(j + 1, len(d.text_boxes)):
                        if d.labels[j] != ListAction.ELIMINATE:
                            tb4 = d.text_boxes[j]
                        break
                # still execute extract_features even if d.labels[i] != ListAction.ELIMINATE
                # to make the state consistent
                feat = d.feature_extractor.extract_features(tb1, tb2, tb3, tb4)
                if d.labels[i] != ListAction.ELIMINATE:
                    d.labels[i] = ListAction(clf.predict(np.array([feat]))[0])

            pointers = []
            for j in range(len(d.labels)):
                X_test_ptr = []
                ptr_candidates = []
                if d.labels[j] == ListAction.UP:
                    for i in range(j):
                        if d.labels[i] == ListAction.DOWN:
                            feat = d.feature_extractor.extract_pointer_features(
                                d.text_boxes, d.labels[:j], i, j)
                            X_test_ptr.append(feat)
                            ptr_candidates.append(i)
                if len(X_test_ptr) > 0:
                    pointers.append(ptr_candidates[np.argmax(clf_ptr.predict_proba(np.array(X_test_ptr))[:, 1])])
                else:
                    pointers.append(-1)
            d.pointers = pointers
            predicted_documents.append(d)
            cum_j += len(documents[doc_idx].feats)
    predicted_documents = [
        predicted_documents[j] for j in np.argsort(np.concatenate(test_indices))]
    y_pred = np.array([l.value for d in predicted_documents for l in d.labels])
    y_true = np.array([l.value for d in documents for l in d.labels])
    print(f'Done prediction. Accuracy={accuracy_score(y_true, y_pred)}.')
    print_confusion_matrix(y_true, y_pred)

    return predicted_documents
