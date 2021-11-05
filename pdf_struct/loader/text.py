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
from typing import List, Optional

import regex as re
import tqdm

from pdf_struct.core.document import TextBlock, Document
from pdf_struct.core.preprocessing import preprocess_text
from pdf_struct.core.transition_labels import ListAction, \
    AnnoListType, filter_text_blocks
from pdf_struct.core.utils import get_filename, groupwise


class TextLine(TextBlock):
    _PAT_INDENT = re.compile(' *')

    def __init__(self, l_before, l, l_next):
        super(TextLine, self).__init__(l.strip())
        self.indent = self._get_indent(l)
        self.width = self._get_line_width(l)
        self.top_spacing = l_before is None or len(l_before.strip()) == 0
        self.bottom_spacing = l_next is None or len(l_next.strip()) == 0
        self.text_orig = l

    @staticmethod
    def _get_indent(text: str) -> int:
        m = TextLine._PAT_INDENT.match(text)
        return len(m.group(0))

    @staticmethod
    def _get_line_width(text: str) -> int:
        return len(text.rstrip())

    @classmethod
    def from_lines(cls, lines: List[str]) -> List['TextLine']:
        text_lines = []
        lines = [preprocess_text(l) for l in lines]
        for l_before, l, l_next in groupwise(lines, 3):
            # ignore empty line, but use it determin top/bottom spacing of
            # adjacent line
            if l is not None and len(l.strip()) > 0:
                text_lines.append(cls(l_before, l, l_next))
        return text_lines


class TextDocumentLoadingError(ValueError):
    pass


def load_document(path: str, labels: Optional[List[ListAction]], pointers: Optional[List[int]]):
    with open(path, 'r') as fin:
        text_boxes = TextLine.from_lines([line for line in fin])
    if len(text_boxes) == 0:
        raise TextDocumentLoadingError('No text boxes found.')

    if labels is not None:
        assert pointers is not None
        if len(labels) != len(text_boxes):
            raise TextDocumentLoadingError('Number of rows does not match labels.')

        text_boxes, labels, pointers = filter_text_blocks(text_boxes, labels, pointers)

    texts = [tb.text for tb in text_boxes]

    return Document(path, texts, text_boxes, labels, pointers, path)


def load_from_directory(base_dir: str, annos: AnnoListType) -> List[Document]:
    paths = glob.glob(os.path.join(base_dir, '*.txt'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(load_document(
                path, [a[0] for a in anno], [a[1] for a in anno]))
        except TextDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents


def create_training_data(in_path, out_path):
    with open(in_path) as fin:
        text_lines = TextLine.from_lines([line for line in fin])

    if len(text_lines) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    with open(out_path, 'w') as fout:
        for line in text_lines:
            fout.write(f'{line.text}\t0\t\n')
