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
import uuid
from functools import reduce
from typing import Generator, Tuple, List, Set, Optional

import tqdm
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure, LTChar
from pdfminer.pdfdocument import PDFDocument as _PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from pdf_struct.core.document import Document, TextBlock
from pdf_struct.core.preprocessing import preprocess_text
from pdf_struct.core.transition_labels import ListAction, AnnoListType, \
    filter_text_blocks
from pdf_struct.core.utils import get_filename


class PDFDocumentLoadingError(ValueError):
    pass


class TextBox(TextBlock):
    def __init__(self, text: str, bbox: Tuple[float, float, float, float],
                 blocks: Set[str], page: int):
        super(TextBox, self).__init__(text)
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bbox: Tuple[float, float, float, float] = bbox
        self.page: int = page
        self.blocks: Set[str] = blocks

    @staticmethod
    def merge_continuous_lines(text_blocks: List['TextBox'], threshold=0.5, space_size=4):
        assert len(set(map(type, text_blocks))) == 1
        if len(text_blocks) <= 1:
            return text_blocks
        text_blocks = sorted(
            text_blocks,
            key=lambda b: (b.page, -b.bbox[1], b.bbox[0]))
        merged_text_blocks = []
        i = 0
        while i < (len(text_blocks) - 1):
            tbi = text_blocks[i]
            # aggregate text boxes in same line then merge
            same_line_boxes = [tbi]
            for j in range(i + 1, len(text_blocks)):
                tbj = text_blocks[j]
                if tbi.page != tbj.page:
                    break
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
                merged_text_blocks.append(TextBox(text, bbox, blocks, tbi.page))
            else:
                merged_text_blocks.append(tbi)
            i = j
        # if len(same_line_boxes) == 1 in last loop, text_blocks[-1] will be missing
        if len(same_line_boxes) == 1:
            merged_text_blocks.append(text_blocks[-1])
        return merged_text_blocks

def parse_pdf(fs) -> Generator[TextBox, None, None]:
    doc = _PDFDocument(PDFParser(fs))
    rsrcmgr = PDFResourceManager()
    laparams = LAParams()
    device = PDFPageAggregator(rsrcmgr, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page_idx, page in enumerate(PDFPage.create_pages(doc)):
         interpreter.process_page(page)
         layout = device.get_result()
         yield from parse_layout(layout, page_idx + 1, None)


def parse_layout(layout, page: int, block: str):
    for lt_obj in layout:
        if isinstance(lt_obj, (LTTextLine, LTChar)):
            text = preprocess_text(lt_obj.get_text().strip('\n'))
            if block is None:
                block = uuid.uuid4().hex
            if len(text.strip()) > 0:
                yield TextBox(text, lt_obj.bbox, {block}, page)
        elif isinstance(lt_obj, LTTextBox):
            block = uuid.uuid4().hex
            yield from parse_layout(lt_obj, page, block)
        elif isinstance(lt_obj, LTFigure):
            yield from parse_layout(lt_obj, page, block)


def load_document(path: str, labels: Optional[List[ListAction]], pointers: Optional[List[int]]):
    with open(path, 'rb') as fin:
        text_boxes = list(parse_pdf(fin))
    if len(text_boxes) == 0:
        raise PDFDocumentLoadingError('No text boxes found.')
    # Space size is about 4pt
    text_boxes = TextBox.merge_continuous_lines(text_boxes, space_size=4)

    if labels is not None:
        assert pointers is not None

        if len(labels) != len(text_boxes):
            raise PDFDocumentLoadingError('Number of rows does not match labels.')

        text_boxes, labels, pointers = filter_text_blocks(text_boxes, labels, pointers)

    texts = [tb.text for tb in text_boxes]

    return Document(path, texts, text_boxes, labels, pointers, path)


def load_from_directory(base_dir: str, annos: AnnoListType) -> List[Document]:
    paths = glob.glob(os.path.join(base_dir, '*.pdf'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(load_document(
                path, [a[0] for a in anno], [a[1] for a in anno]))
        except PDFDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents


def create_training_data(in_path, out_path):
    with open(in_path, 'rb') as fin:
        text_boxes = list(parse_pdf(fin))

    if len(text_boxes) == 0:
        raise RuntimeError(f'No text boxes found for document "{in_path}".')

    text_boxes = TextBox.merge_continuous_lines(
        text_boxes, space_size=4)

    with open(out_path, 'w') as fout:
        for tb in text_boxes:
            fout.write(f'{tb.text}\t0\t\n')
