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
import operator
import os
import re
from functools import reduce
from typing import Tuple, List, Set

import tqdm
from bs4 import BeautifulSoup

from pdf_struct.core.document import Document, TextBlock
from pdf_struct.core.preprocessing import preprocess_text
from pdf_struct.core.transition_labels import ListAction, AnnoListType, \
    filter_text_blocks
from pdf_struct.core.utils import get_filename


class SpanBox(TextBlock):
    def __init__(self,
                 text: str,
                 bbox: Tuple[float, float, float, float],
                 block_ids: Set[str],
                 cell_size: Tuple[float, float]):
        super(SpanBox, self).__init__(text)
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        # This is NOT the same as the original hocr bbox
        self.bbox: Tuple[float, float, float, float] = bbox
        self.blocks: Set[str] = block_ids
        # cell_size is geometry of cell [width, height] in points
        assert cell_size[0] >= 0 and cell_size[1] >= 0
        self.cell_size: Tuple[float, float] = cell_size

    @classmethod
    def merge_continuous_lines(cls, text_blocks: List['SpanBox'],
                               threshold=0.5, space_size=4):
        assert len(set(map(type, text_blocks))) == 1
        if len(text_blocks) <= 1:
            return text_blocks
        text_blocks = sorted(
            text_blocks, key=lambda b: (-b.bbox[1], b.bbox[0]))
        merged_text_blocks = []
        i = 0
        while i < (len(text_blocks) - 1):
            tbi = text_blocks[i]
            # aggregate text boxes in same line then merge
            same_line_boxes = [tbi]
            for j in range(i + 1, len(text_blocks)):
                tbj = text_blocks[j]
                # text_blocks[j]'s y_bottom is always lower than text_blocks[i]'s
                span = max(tbi.bbox[3], tbj.bbox[3]) - tbj.bbox[1]
                overlap = min(tbi.bbox[3], tbj.bbox[3]) - tbi.bbox[1]
                if overlap / span > threshold:
                    same_line_boxes.append(tbj)
                    continue
                else:
                    # stop scanning for same line for efficiency
                    break
            if len(same_line_boxes) > 1:
                # sort left to right
                same_line_boxes = sorted(same_line_boxes, key=lambda b: b.bbox[0])
                text = same_line_boxes[0].text.strip('\n')
                bbox = same_line_boxes[0].bbox
                for tbk in same_line_boxes[1:]:
                    spaces = max(tbk.bbox[0] - bbox[2], 0)
                    text += int(spaces // space_size) * ' '
                    text += tbk.text.strip('\n')
                    bbox = [
                        bbox[0],
                        min(bbox[1], tbk.bbox[1]),
                        max(bbox[2], tbk.bbox[2]),
                        max(bbox[3], tbk.bbox[3])
                    ]
                blocks = reduce(operator.or_, (b.blocks for b in same_line_boxes))
                merged_text_blocks.append(SpanBox(text, bbox, blocks, tbi.cell_size))
            else:
                merged_text_blocks.append(tbi)
            i = j
        # if len(same_line_boxes) == 1 in last loop, text_blocks[-1] will be missing
        if len(same_line_boxes) == 1:
            merged_text_blocks.append(text_blocks[-1])
        return merged_text_blocks


_RE_PAGE = re.compile(r'page_([1-9][0-9]*)')


def _extract_attr_from_title(title: str):
    attributes = dict()
    for t in title.strip().split(';'):
        t = t.strip().split(' ')
        if t[0] in attributes:
            raise OSError(f'Duplicate entry in title ({title})')
        attributes[t[0]] = t[1:]
    return attributes


def parse_hocr(html_doc: str) -> List[List[SpanBox]]:
    soup = BeautifulSoup(html_doc, 'html.parser')
    span_boxes = []
    for page in soup.find_all("div", class_="ocr_page"):
        page_num = int(_RE_PAGE.match(page['id']).group(1))
        for table in page.find_all('table'):
            # do some table-specific operations
            for td in table.find_all('td'):
                spans = td.find_all('span')
                if len(spans) <= 1:
                    continue
                td_attr = _extract_attr_from_title(td['title'])
                # original bbox is [x_left, y_top, x_right, y_bottom] in points with
                # left top being [0, 0, 0, 0]
                td_bbox = list(map(int, td_attr['bbox']))
                cell_size = (td_bbox[2] - td_bbox[0], td_bbox[3] - td_bbox[1])
                span_boxes_td = []
                for i, span in enumerate(spans):
                    span_attr = _extract_attr_from_title(span['title'])
                    span_bbox = list(map(int, span_attr['bbox']))
                    # transform bbox to be compatible with SpanBox
                    trans_bbox = (
                        span_bbox[0] - td_bbox[0],
                        td_bbox[3] - span_bbox[3],
                        span_bbox[2] - td_bbox[0],
                        td_bbox[3] - span_bbox[1]
                    )
                    assert 0 <= trans_bbox[0] <= trans_bbox[2]
                    assert 0 <= trans_bbox[1] <= trans_bbox[3]
                    text = preprocess_text(span.text.strip('\n'))
                    # use span['title'] as a pseudo ID for now
                    span_boxes_td.append(
                        SpanBox(text, trans_bbox, {span['title']}, cell_size))
                # HOCR's geometry unit is point, so ex = 4pt
                span_boxes_td = SpanBox.merge_continuous_lines(span_boxes_td, space_size=4)
                if len(span_boxes_td) > 1:
                    span_boxes.append(span_boxes_td)
    return span_boxes


class HOCRDocumentLoadingError(ValueError):
    pass


def load_document(path: str, labels: List[ListAction], pointers: List[int]) -> List['Document']:
    with open(path) as fin:
        html_doc = fin.read()
    span_boxes_lst = parse_hocr(html_doc)
    if len(labels) != sum(map(len, span_boxes_lst)):
        raise HOCRDocumentLoadingError('Number of rows does not match labels.')

    documents = []
    cur_index = 0
    for span_boxes in span_boxes_lst:
        _labels = labels[cur_index:cur_index + len(span_boxes)]
        _pointers = pointers[cur_index:cur_index + len(span_boxes)]
        cur_index += len(span_boxes)

        text_boxes, _labels, _pointers = filter_text_blocks(span_boxes, _labels, _pointers)
        if len(text_boxes) < 2:
            # could be that whole cell is excluded
            continue
        texts = [tb.text for tb in text_boxes]

        documents.append(
            Document(path, texts, text_boxes, _labels, _pointers, path))
    return documents


def load_from_directory(base_dir: str, annos: AnnoListType) -> List[Document]:
    paths = glob.glob(os.path.join(base_dir, '*.hocr'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.extend(load_document(
                path, [a[0] for a in anno], [a[1] for a in anno]))
        except HOCRDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents


def create_training_data(in_path, out_path):
    with open(in_path) as fin:
        html_doc = fin.read()
    span_boxes_lst = parse_hocr(html_doc)

    if len(span_boxes_lst) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    with open(out_path, 'w') as fout:
        for i, span_boxes in enumerate(span_boxes_lst):
            for span in span_boxes:
                fout.write(f'{i:>05d} {span.text}\t0\t\n')
