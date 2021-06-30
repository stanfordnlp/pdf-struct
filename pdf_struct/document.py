from typing import List, Tuple, Optional, Dict
from itertools import chain

from pdf_struct.transition_labels import ListAction


class TextBlock(object):
    def __init__(self, text: str):
        self.text: str = text


class Document(object):
    def __init__(self,
                 path: str,
                 feats: Optional[Dict[str, Dict[str, List[float]]]],
                 feats_test: Dict[str, Dict[str, List[float]]],
                 texts: List[str],
                 labels: Optional[List[ListAction]],
                 pointers: Optional[List[Optional[int]]],
                 pointer_feats: Optional[Dict[str, Dict[str, List[float]]]],
                 pointer_candidates: Optional[List[Tuple[int, int]]],
                 feature_extractor,
                 text_blocks: List[TextBlock],
                 cv_key: str):
        assert len(texts) == len(labels)
        self.path: str = path
        # features to be used at train time. This is created with an access
        # to the labels
        self.feats: Optional[Dict[str, Dict[str, List[float]]]] = feats
        self.feature_array = self._create_feature_array(feats)
        # features to be used at test time. This is created without an access
        # to the labels
        self.feats_test: Dict[str, Dict[str, List[float]]] = feats_test
        self.feature_array_test = self._create_feature_array(feats_test)
        self.texts: List[str] = texts
        # Ground-truth/predicted labels
        self.labels: Optional[List[ListAction]] = labels
        # Ground-truth/predicted pointer labels
        self.pointers: Optional[List[Optional[int]]] = pointers
        # this can be None at inference, because it is calculated on the run
        self.pointer_feats: Optional[Dict[str, Dict[str, List[float]]]] = pointer_feats
        if len(pointer_feats) > 0:
            self.pointer_feats_array = self._create_feature_array(pointer_feats)
        else:
            self.pointer_feats_array = []
        self.pointer_candidates: Optional[List[Tuple[int, int]]] = pointer_candidates
        self.feature_extractor = feature_extractor
        self.text_blocks: List[TextBlock] = text_blocks
        # Key to use for CV partitioning
        self.cv_key: str = cv_key

    @property
    def n_blocks(self):
        return len(self.texts)

    @property
    def n_features(self):
        return sum(map(len, self.feats.values()))

    @staticmethod
    def _create_feature_array(feats) -> Optional[List[List[float]]]:
        if feats is None:
            return None
        n_blocks = len(list(list(feats.values())[0].values())[0])
        # List of list of size (n_blocks, n_feats)
        features = [[] for _ in range(n_blocks)]
        for feature in Document.unpack_features(feats):
            for j, f in enumerate(feature):
                features[j].append(float(f))
        return features

    @staticmethod
    def unpack_features(features):
        return [
            v for _, v in sorted(
                chain(*[fg.items() for fg in features.values()]),
                key=lambda k_v: k_v[0])
        ]
