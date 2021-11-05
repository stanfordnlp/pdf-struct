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

import glob
import os
from enum import Enum
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import tqdm

from pdf_struct.core.utils import get_filename


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


def filter_text_blocks(text_blocks, labels, pointers):
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
                assert len(_pointers) > p >= 0
            _pointers.append(p)
            _text_boxes.append(text_blocks[i])
        else:
            pointers_tmp = []
            for p in pointers:
                if p is None:
                    pointers_tmp.append(None)
                elif p > len(_labels):
                    pointers_tmp.append(p - 1)
                else:
                    pointers_tmp.append(p)
            pointers = pointers_tmp
    return _text_boxes, _labels, _pointers
