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

from typing import List, Tuple, Optional, Dict
from itertools import chain

from pdf_struct.core.transition_labels import ListAction


class TextBlock(object):
    def __init__(self, text: str):
        self.text: str = text


class Document(object):
    def __init__(self,
                 path: str,
                 texts: List[str],
                 text_blocks: List[TextBlock],
                 labels: Optional[List[ListAction]],
                 pointers: Optional[List[Optional[int]]],
                 cv_key: str):
        assert labels is None or len(texts) == len(labels)
        self.path: str = path
        self.texts: List[str] = texts
        self.text_blocks: List[TextBlock] = text_blocks
        # Ground-truth/predicted labels
        self.labels: Optional[List[ListAction]] = labels
        # Ground-truth/predicted pointer labels
        self.pointers: Optional[List[Optional[int]]] = pointers
        # Key to use for CV partitioning
        self.cv_key: str = cv_key

    def set_features(self,
                     feats: Optional[Dict[str, Dict[str, List[float]]]],
                     feats_test: Dict[str, Dict[str, List[float]]],
                     pointer_feats: Optional[Dict[str, Dict[str, List[float]]]],
                     pointer_candidates: Optional[List[Tuple[int, int]]],
                     feature_extractor):
        # features to be used at train time. This is created with an access
        # to the labels
        self.feats: Optional[Dict[str, Dict[str, List[float]]]] = feats
        self.feature_array = self._get_feature_matrix(feats)
        # features to be used at test time. This is created without an access
        # to the labels
        self.feats_test: Dict[str, Dict[str, List[float]]] = feats_test
        self.feature_array_test = self._get_feature_matrix(feats_test)

        self.pointer_feats: Optional[Dict[str, Dict[str, List[float]]]] = pointer_feats
        if pointer_feats is not None and len(pointer_feats) > 0:
            self.pointer_feats_array = self._get_feature_matrix(pointer_feats)
        else:
            self.pointer_feats_array = []
        self.pointer_candidates: Optional[List[Tuple[int, int]]] = pointer_candidates
        self.feature_extractor = feature_extractor

    @property
    def n_blocks(self):
        return len(self.texts)

    @property
    def n_features(self):
        assert self.feats is not None and 'self.feats accessed before set'
        return sum(map(len, self.feats.values()))

    @property
    def n_pointer_features(self):
        return None if self.pointer_feats is None else sum(map(len, self.pointer_feats.values()))

    def get_feature_names(self):
        return [k for k, _ in self._unpack_features(self.feats_test)]

    @staticmethod
    def _get_feature_matrix(feats) -> Optional[List[List[float]]]:
        if feats is None:
            return None
        n_blocks = len(list(list(feats.values())[0].values())[0])
        # List of list of size (n_blocks, n_feats)
        features = [[] for _ in range(n_blocks)]
        for _, feature in Document._unpack_features(feats):
            for j, f in enumerate(feature):
                features[j].append(float(f))
        return features

    @staticmethod
    def get_feature_array(features):
        return [v for _, v in Document._unpack_features(features)]

    @staticmethod
    def _unpack_features(features):
        return [
            (k, v) for k, v in sorted(
                chain(*[fg.items() for fg in features.values()]),
                key=lambda k_v: k_v[0])
        ]
