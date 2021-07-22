import glob
import os
from typing import List

import regex as re
import tqdm

from pdf_struct.document import TextBlock, Document
from pdf_struct.listing import get_text_body_indent
from pdf_struct.preprocessing import preprocess_text
from pdf_struct.text.features import PlainTextFeatureExtractor
from pdf_struct.transition_labels import ListAction, \
    AnnoListType, filter_text_blocks
from pdf_struct.utils import get_filename, groupwise


class TextLine(TextBlock):
    _PAT_INDENT = re.compile(' *')

    def __init__(self, l_before, l, l_next):
        super(TextLine, self).__init__(l.strip())
        self.indent = self._get_indent(l)
        self.width = self._get_line_width(l)
        self.top_spacing = l_before is None or len(l_before.strip()) == 0
        self.bottom_spacing = l_next is None or len(l_next.strip()) == 0
        self.body_indent = get_text_body_indent(l)

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


def load_text_document(path: str, labels: List[ListAction], pointers: List[int], dummy_feats: bool=False):
    with open(path, 'r') as fin:
        text_boxes = TextLine.from_lines([line for line in fin])
    if len(text_boxes) == 0:
        raise TextDocumentLoadingError('No text boxes found.')
    if len(labels) != len(text_boxes):
        raise TextDocumentLoadingError('Number of rows does not match labels.')

    text_boxes, labels, pointers = filter_text_blocks(text_boxes, labels, pointers)
    texts = [tb.text for tb in text_boxes]

    feature_extractor, feats, feats_test, pointer_feats, pointer_candidates = PlainTextFeatureExtractor.initialize_and_extract_all_features(
        text_boxes, labels, pointers, dummy_feats, text_boxes)

    return Document(path, feats, feats_test, texts, labels, pointers,
                    pointer_feats, pointer_candidates, feature_extractor,
                    text_boxes, path)

def load_text_document_for_prediction(path: str):
    with open(path, 'r') as fin:
        text_lines = TextLine.from_lines([line for line in fin])
    if len(text_lines) == 0:
        raise TextDocumentLoadingError('No text boxes found.')

    texts = [tb.text for tb in text_lines]

    feature_extractor = PlainTextFeatureExtractor(text_lines)
    feats_test = feature_extractor.extract_features_all(text_lines, None)

    return Document(path, None, feats_test, texts, None, None, None, None,
                    feature_extractor, text_lines, path)


def load_texts_from_directory(base_dir: str, annos: AnnoListType, dummy_feats: bool=False) -> List[Document]:
    paths = glob.glob(os.path.join(base_dir, '*.txt'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(load_text_document(
                path, [a[0] for a in anno], [a[1] for a in anno], dummy_feats=dummy_feats))
        except TextDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents
