import glob
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional

import tqdm

from pdf_struct.utils import get_filename


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
                 labels: List[ListAction], pointers: List[Optional[int]]):
        assert len(feats) == len(texts) == len(labels)
        self.path: str = path
        self.feats: List[List[float]] = feats
        self.texts: List[str] = texts
        self.labels: List[ListAction] = labels
        self.pointers: List[Optional[int]] = pointers


def _load_anno(in_path) -> List[Tuple[ListAction, Optional[int]]]:
    ret = []
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
