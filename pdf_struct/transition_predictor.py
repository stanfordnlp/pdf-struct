import copy
import random
from typing import List, Optional
from collections import defaultdict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from pdf_struct.transition_labels import ListAction, DocumentWithFeatures


def train_classifiers(documents: List[DocumentWithFeatures], used_features: Optional[List[int]]=None):
    used_features = None if used_features is None else np.array(used_features)

    # First, classify transition between consecutive lines
    X_train = np.array(sum([d.feats for d in documents], []),
                       dtype=np.float64)
    y_train = np.array(
        [l.value for d in documents for l in d.labels], dtype=int)
    if used_features is not None:
        X_train = X_train[:, used_features]

    clf = RandomForestClassifier().fit(X_train, y_train)

    # Next, predict pointers
    X_train = np.array(
        [f[2] for d in documents for f in d.pointer_feats],
        dtype=np.float64)
    y_train = np.array(
        [f[0] == d.pointers[f[1]] for d in documents for f in
         d.pointer_feats],
        dtype=int)
    clf_ptr = RandomForestClassifier().fit(X_train, y_train)
    return clf, clf_ptr


def predict_with_classifiers(clf, clf_ptr, documents: List[DocumentWithFeatures], used_features: Optional[List[int]]=None) -> List[DocumentWithFeatures]:
    used_features = None if used_features is None else np.array(used_features)

    X_test = np.array(sum([d.feats_test for d in documents], []),
                      dtype=np.float64)
    if used_features is not None:
        X_test = X_test[:, used_features]

    y_pred = clf.predict(X_test)
    predicted_documents = []
    cum_j = 0
    for document in documents:
        d = copy.deepcopy(document)
        d.labels = [ListAction(yi) for yi in
                    y_pred[cum_j:cum_j + len(document.feats)]]

        d.feature_extractor.init_state()
        for i in range(len(d.text_blocks)):
            tb1 = d.text_blocks[i - 1] if i != 0 else None
            tb2 = d.text_blocks[i]
            if d.labels[i] == ListAction.ELIMINATE:
                tb3 = d.text_blocks[i + 1] if i + 1 < len(
                    d.text_blocks) else None
                tb4 = d.text_blocks[i + 2] if i + 2 < len(
                    d.text_blocks) else None
            else:
                tb3 = None
                for j in range(i + 1, len(d.text_blocks)):
                    if d.labels[j] != ListAction.ELIMINATE:
                        tb3 = d.text_blocks[j]
                        break
                tb4 = d.text_blocks[j + 1] if j + 1 < len(
                    d.text_blocks) else None
            # still execute extract_features even if d.labels[i] != ListAction.ELIMINATE
            # to make the state consistent
            feat = np.array([
                d.feature_extractor.extract_features(tb1, tb2, tb3, tb4)])
            if used_features is not None:
                feat = feat[:, used_features]
            if d.labels[i] != ListAction.ELIMINATE:
                d.labels[i] = ListAction(clf.predict(feat)[0])

        pointers = []
        for j in range(len(d.labels)):
            X_test_ptr = []
            ptr_candidates = []
            if d.labels[j] == ListAction.UP:
                for i in range(j):
                    if d.labels[i] == ListAction.DOWN:
                        feat = d.feature_extractor.extract_pointer_features(
                            d.text_blocks, d.labels[:j], i, j)
                        X_test_ptr.append(feat)
                        ptr_candidates.append(i)
            if len(X_test_ptr) > 0:
                pointers.append(ptr_candidates[np.argmax(
                    clf_ptr.predict_proba(np.array(X_test_ptr))[:, 1])])
            else:
                pointers.append(-1)
        d.pointers = pointers
        predicted_documents.append(d)
        cum_j += len(document.feats)
    return predicted_documents


def k_fold_train_predict(documents: List[DocumentWithFeatures], n_splits: int=5, used_features: Optional[List[int]]=None) -> List[DocumentWithFeatures]:
    test_indices = []
    predicted_documents = []

    cv_documents = defaultdict(list)
    for i, document in enumerate(documents):
        cv_documents[document.cv_key].append((document, i))
    cv_documents = list(cv_documents.values())

    random.seed(123)
    np.random.seed(123)
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(X=cv_documents):
        test_indices.append([ind for j in test_index for _, ind in cv_documents[j]])
        documents_train = [document for j in train_index for document, ind in cv_documents[j]]
        documents_test = [document for j in test_index for document, ind in cv_documents[j]]
        clf, clf_ptr = train_classifiers(documents_train, used_features)
        predicted_documents.extend(
            predict_with_classifiers(clf, clf_ptr, documents_test, used_features))
    predicted_documents = [
        predicted_documents[j] for j in np.argsort(np.concatenate(test_indices))]

    return predicted_documents
