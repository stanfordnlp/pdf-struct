import glob
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

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
    def __init__(self, path: str,
                 feats: Optional[List[List[float]]],
                 feats_test: List[List[float]],
                 texts: List[str],
                 labels: Optional[List[ListAction]],
                 pointers: Optional[List[Optional[int]]],
                 pointer_feats: Optional[List[Tuple[int, int, List[float]]]],
                 feature_extractor: 'pdf_struct.features.BaseFeatureExtractor',
                 text_blocks: List[TextBlock],
                 cv_key: str):
        assert len(feats) == len(texts) == len(labels)
        self.path: str = path
        # features to be used at train time. This is created with an access
        # to the labels
        self.feats: Optional[List[List[float]]] = feats
        # features to be used at test time. This is created without an access
        # to the labels
        self.feats_test: List[List[float]] = feats_test
        self.texts: List[str] = texts
        # Ground-truth/predicted labels
        self.labels: Optional[List[ListAction]] = labels
        # Ground-truth/predicted pointer labels
        self.pointers: Optional[List[Optional[int]]] = pointers
        # this can be None at inference, because it is calculated on the run
        self.pointer_feats: Optional[List[Tuple[int, int, List[float]]]] = pointer_feats
        self.feature_extractor: 'pdf_struct.features.BaseFeatureExtractor' = feature_extractor
        self.text_blocks: List[TextBlock] = text_blocks
        # Key to use for CV partitioning
        self.cv_key: str = cv_key

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
                    if p is None:
                        pointers_tmp.append(None)
                    elif p > i:
                        pointers_tmp.append(p - 1)
                    else:
                        pointers_tmp.append(p)
                pointers = pointers_tmp
        return _text_boxes, _labels, _pointers

    @staticmethod
    def _extract_features(feature_extractor, text_blocks, labels):
        feats = list(feature_extractor.extract_features_all(text_blocks, labels))
        return feats

    @staticmethod
    def _extract_pointer_features(feature_extractor, text_blocks, labels, pointers):
        pointer_feats = []
        for j, p in enumerate(pointers):
            if p is not None:
                assert p >= 0
                for i in range(j):
                    if labels[i] == ListAction.DOWN:
                        feat = feature_extractor.extract_pointer_features(
                            text_blocks, labels[:j], i, j)
                        pointer_feats.append((i, j, feat))
        return pointer_feats

    @classmethod
    def _extract_all_features(cls, feature_extractor_func, text_blocks, labels, pointers, dummy_feats):
        assert labels is not None
        feature_extractor = feature_extractor_func(text_blocks)
        if dummy_feats:
            feature_extractor = None
            feats = [[]] * len(text_blocks)
            feats_test = [[]] * len(text_blocks)
            pointer_feats = []
        else:
            feats = cls._extract_features(
                feature_extractor, text_blocks, labels)
            feats_test = cls._extract_features(
                feature_extractor, text_blocks, None)
            pointer_feats = cls._extract_pointer_features(
                feature_extractor, text_blocks, labels, pointers)
        return feature_extractor, feats, feats_test, pointer_feats


def _load_anno(in_path: str, lines: List[str], offset: int) -> List[Tuple[ListAction, Optional[int]]]:
    ret = []
    root_line_indices = set()
    root_flg = True
    for i, line in enumerate(lines):
        line_num = i + 1 + offset  # for debugging
        line = line.rstrip('\n').split('\t')
        if len(line) != 3:
            raise ValueError(
                f'Invalid line "{line}" in {line_num}-th line of "{in_path}".')
        if not ListAction.contains(line[2]):
            raise ValueError(
                f'Invalid label "{line[2]}" in {line_num}-th line of "{in_path}".')
        ptr = int(line[1])
        if offset > 0 and ptr > 0:
            ptr -= offset
        if not (-1 <= ptr <= i):
            raise ValueError(
                f'Invalid pointer "{line[1]}" in {line_num}-th line of "{in_path}".')
        if ptr == 0:
            ptr = None
        elif ptr > 0:
            ptr = ptr - 1
        if ptr is not None and ptr > 0 and ret[ptr][0] != ListAction.DOWN:
            raise ValueError(
                f'Pointer is pointing at "{ret[ptr][0]}" in {line_num}-th line of "{in_path}".')
        try:
            l = ListAction.from_key(line[2], ptr)
        except ValueError as e:
            raise ValueError(f'{e} in {line_num}-th line of "{in_path}".')
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
                  f'{line_num}-th line of "{in_path}". Turning it into SAME_LEVEL.')
            ptr = None
            l = ListAction.SAME_LEVEL
        ret.append((l, ptr))
    return ret


AnnoListType = Dict[str, List[Tuple[ListAction, Optional[int]]]]


def load_annos(base_dir: str) -> AnnoListType:
    annos = dict()
    for path in tqdm.tqdm(glob.glob(os.path.join(base_dir, '*.tsv'))):
        with open(path, 'r') as fin:
            lines = [l for l in fin]
        a = _load_anno(path, lines, offset=0)
        filename = get_filename(path)
        annos[filename] = a
    return annos


def load_hocr_annos(base_dir: str) -> AnnoListType:
    annos = defaultdict(list)
    for path in tqdm.tqdm(glob.glob(os.path.join(base_dir, '*.tsv'))):
        filename = get_filename(path)
        with open(path, 'r') as fin:
            cur_id = None
            cur_idx = 0
            for i, line in enumerate(fin):
                if line[:5] != cur_id:
                    if cur_id is not None:
                        assert len(lines) > 1
                        a = _load_anno(path, lines, offset=cur_idx)
                        annos[filename].extend(a)
                    lines = []
                    cur_id = line[:5]
                    cur_idx = i
                lines.append(line)
        assert len(lines) > 1
        a = _load_anno(path, lines, offset=cur_idx)
        annos[filename].extend(a)
    return dict(annos)
