import copy
import random
from typing import List, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

from pdf_struct.transition_labels import ListAction, DocumentWithFeatures


def k_fold_train_predict(documents: List[DocumentWithFeatures], n_splits: int=5, used_features: Optional[List[int]]=None) -> List[DocumentWithFeatures]:
    used_features = None if used_features is None else np.array(used_features)
    test_indices = []
    predicted_documents = []
    random.seed(123)
    np.random.seed(123)
    for train_index, test_index in KFold(n_splits=n_splits, shuffle=True).split(X=documents):
        test_indices.append(test_index)

        # First, classify transition between consecutive lines
        X_train = np.array(sum([documents[j].feats for j in train_index], []), dtype=np.float64)
        y_train = np.array([l.value for j in train_index for l in documents[j].labels], dtype=int)
        X_test = np.array(sum([documents[j].feats for j in test_index], []), dtype=np.float64)
        if used_features is not None:
            X_train = X_train[:, used_features]
            X_test = X_test[:, used_features]

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
            for i in range(len(d.text_blocks)):
                tb1 = d.text_blocks[i - 1] if i != 0 else None
                tb2 = d.text_blocks[i]
                if d.labels[i] == ListAction.ELIMINATE:
                    tb3 = d.text_blocks[i + 1] if i + 1 < len(d.text_blocks) else None
                    tb4 = d.text_blocks[i + 2] if i + 2 < len(d.text_blocks) else None
                else:
                    tb3 = None
                    for j in range(i + 1, len(d.text_blocks)):
                        if d.labels[j] != ListAction.ELIMINATE:
                            tb3 = d.text_blocks[j]
                            break
                    tb4 = d.text_blocks[j + 1] if j + 1 < len(d.text_blocks) else None
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
                    pointers.append(ptr_candidates[np.argmax(clf_ptr.predict_proba(np.array(X_test_ptr))[:, 1])])
                else:
                    pointers.append(-1)
            d.pointers = pointers
            predicted_documents.append(d)
            cum_j += len(documents[doc_idx].feats)
    predicted_documents = [
        predicted_documents[j] for j in np.argsort(np.concatenate(test_indices))]

    return predicted_documents
