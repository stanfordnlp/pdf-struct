import glob
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional

import tqdm

from pdf_struct.utils import get_filename


class TextBlock(object):
    def __init__(self, text: str):
        self.text: str = text


class ListAction(Enum):
    EXCLUDED = -1
    CONTINUOUS = 0
    SAME_LEVEL = 1
    DOWN = 2
    UP = 3
    ELIMINATE = 4
    ADDRESS = 5

    @staticmethod
    def contains(key: str) -> bool:
        return key in {'c', 'a', 'b', 's', 'd', 'e', 'x'}

    @classmethod
    def from_key(
            cls, key: str, pointer: Optional[int], use_address: bool=False) -> 'ListAction':
        if pointer is not None:
            if key == 'e' or key == 'x':
                raise ValueError(f'Cannot use a pointer with {key}')
            if pointer == -1 and key != 's':
                raise ValueError(f'Cannot use -1 with {key}')
            return cls.UP
        if use_address and key == 'a':
            return cls.ADDRESS
        if key == 'c' or key == 'a':
            return cls.CONTINUOUS
        # annotated block or same_level
        if key == 'b' or key == 's':
            return cls.SAME_LEVEL
        if key == 'x':
            return cls.EXCLUDED
        if key == 'd':
            return cls.DOWN
        if key == 'e':
            return cls.ELIMINATE
        raise ValueError(f'Unknown key {key}')


class DocumentWithFeatures(object):
    def __init__(self, path: str, feats: List[List[float]], texts: List[str],
                 labels: List[ListAction], pointers: List[Optional[int]],
                 pointer_feats: List[Tuple[int, int, List[float]]],
                 feature_extractor: 'pdf_struct.features.BaseFeatureExtractor',
                 text_blocks: List[TextBlock]):
        assert len(feats) == len(texts) == len(labels)
        self.path: str = path
        self.feats: List[List[float]] = feats
        self.texts: List[str] = texts
        self.labels: List[ListAction] = labels
        self.pointers: List[Optional[int]] = pointers
        self.pointer_feats: List[Tuple[int, int, List[float]]] = pointer_feats
        self.feature_extractor: 'pdf_struct.features.BaseFeatureExtractor' = feature_extractor
        self.text_blocks: List[TextBlock] = text_blocks

    @staticmethod
    def _filter_text_blocks(text_blocks, labels, pointers):
        _labels = []
        _pointers = []
        _text_boxes = []
        for i in range(len(labels)):
            if labels[i] != ListAction.EXCLUDED:
                _labels.append(labels[i])
                if pointers[i] is None:
                    p = None
                elif pointers[i] == -1:
                    p = -1
                else:
                    p = pointers[i]
                    assert p >= 0
                _pointers.append(p)
                _text_boxes.append(text_blocks[i])
            else:
                pointers_tmp = []
                for p in pointers:
                    assert p != i
                    if p is None:
                        pointers_tmp.append(None)
                    elif p > i:
                        pointers_tmp.append(p - 1)
                    else:
                        pointers_tmp.append(p)
                pointers = pointers_tmp
        return _text_boxes, _labels, _pointers

    @staticmethod
    def _extract_features(feature_extractor_cls, text_blocks, labels, pointers, dummy_feats):
        if dummy_feats:
            feature_extractor = None
            # Do not assign None because some functions relies on its length
            feats = [[]] * len(text_blocks)
            pointer_feats = []
        else:
            feature_extractor = feature_extractor_cls(text_blocks)
            feats = list(feature_extractor.extract_features_all(text_blocks, labels))
            pointer_feats = []
            for j, p in enumerate(pointers):
                if p is not None:
                    assert p >= 0
                    for i in range(j):
                        if labels[i] == ListAction.DOWN:
                            feat = feature_extractor.extract_pointer_features(
                                text_blocks, labels[:j], i, j)
                            pointer_feats.append((i, j, feat))
        return feature_extractor, feats, pointer_feats



def _load_anno(in_path) -> List[Tuple[ListAction, Optional[int]]]:
    ret = []
    root_line_indices = set()
    root_flg = True
    with open(in_path, 'r') as fin:
        for i, line in enumerate(fin):
            line = line.rstrip('\n').split('\t')
            if len(line) != 3:
                raise ValueError(
                    f'Invalid line "{line}" in {i + 1}-th line of "{in_path}".')
            if not ListAction.contains(line[2]):
                raise ValueError(
                    f'Invalid label "{line[2]}" in {i + 1}-th line of "{in_path}".')
            ptr = int(line[1])
            if not (-1 <= ptr <= i):
                raise ValueError(
                    f'Invalid pointer "{line[1]}" in {i + 1}-th line of "{in_path}".')
            if ptr == 0:
                ptr = None
            elif ptr > 0:
                ptr = ptr - 1
            if ptr is not None and ptr > 0 and ret[ptr][0] != ListAction.DOWN:
                raise ValueError(
                    f'Pointer is pointing at "{ret[ptr][0]}" in {i + 1}-th line of "{in_path}".')
            try:
                l = ListAction.from_key(line[2], ptr)
            except ValueError as e:
                raise ValueError(f'{e} in {i + 1}-th line of "{in_path}".')
            if ptr is not None and ptr in root_line_indices:
                root_flg = True
            if root_flg:
                root_line_indices.add(i)
            if ptr == -1:
                ptr = max(root_line_indices)
            if l == ListAction.DOWN:
                root_flg = False
            if ptr == i:
                print('Pointer pointing at root when it is already in root in '
                      f'{i + 1}-th line of "{in_path}". Turning it into SAME_LEVEL.')
                ptr = None
                l = ListAction.SAME_LEVEL
            ret.append((l, ptr))
    return ret


AnnoListType = Dict[str, List[Tuple[ListAction, Optional[int]]]]


def load_annos(base_dir: str) -> AnnoListType:
    annos = dict()
    for path in tqdm.tqdm(glob.glob(os.path.join(base_dir, '*.tsv'))):
        a = _load_anno(path)
        filename = get_filename(path)
        annos[filename] = a
    return annos
