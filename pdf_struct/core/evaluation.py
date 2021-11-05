# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import Counter
from typing import List, Type

import numpy as np
import tqdm
from sklearn.metrics import accuracy_score

from pdf_struct.core import predictor
from pdf_struct.core.document import Document
from pdf_struct.core.feature_extractor import BaseFeatureExtractor
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels


def _create_prediction_jsons(documents, documents_pred) -> List[dict]:
    predictions = []
    for d, d_p in zip(documents, documents_pred):
        assert d.path == d_p.path
        transition_prediction_accuracy = accuracy_score(
            np.array([l.value for l in d.labels]),
            np.array([l.value for l in d_p.labels])
        )
        predictions.append({
            'path': d.path,
            'texts': d.texts,
            'features': d.feature_array,
            'transition_prediction_accuracy': transition_prediction_accuracy,
            'ground_truth': {
                'labels': [l.name for l in d.labels],
                'pointers': d.pointers
            },
            'prediction': {
                'labels': [l.name for l in d_p.labels],
                'pointers': d_p.pointers
            }
        })
    return predictions


def evaluate(documents: List[Document],
             feature_extractor_cls: Type[BaseFeatureExtractor],
             k_folds: int,
             prediction: bool=False):
    print('Extracting features from documents')
    documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

    print(f'Extracted {sum(map(lambda d: d.n_blocks, documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    print(f'Extracted {documents[0].n_features}-dim features and '
          f'{documents[0].n_pointer_features}-dim pointer features.')
    documents_pred = predictor.k_fold_train_predict(
        documents, n_splits=k_folds)

    metrics = {
        'structure': evaluate_structure(documents, documents_pred),
        'labels': evaluate_labels(documents, documents_pred)
    }
    if prediction:
        prediction_jsons = _create_prediction_jsons(documents, documents_pred)
        return metrics, prediction_jsons
    else:
        return metrics
