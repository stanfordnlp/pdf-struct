import glob
import os
from typing import List

import tqdm

from pdf_struct.pdf.features import extract_features
from pdf_struct.pdf.parser import parse_pdf, merge_continuous_lines, TextBox
from pdf_struct.transition_predictor import DocumentWithFeatures


class PDFDocumentLoadingError(ValueError):
    pass


class PDFDocumentWithFeatures(DocumentWithFeatures):
    def __init__(self, path: str, feats: list, texts: List[str], text_boxes: List[TextBox]):
        self._path = path
        self._feats = feats
        self._texts = texts
        self._text_boxes = text_boxes

    @property
    def text_boxes(self) -> List[TextBox]:
        return self._text_boxes

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as fin:
            text_boxes = list(parse_pdf(fin))
        if len(text_boxes) == 0:
            raise PDFDocumentLoadingError('No text boxes found.')
        text_boxes = merge_continuous_lines(text_boxes)
        texts = [tb.text for tb in text_boxes]
        feats = list(extract_features(text_boxes))
        return cls(path, feats, texts, text_boxes)


def load_pdfs(base_dir: str) -> List[PDFDocumentWithFeatures]:
    paths = glob.glob(os.path.join(base_dir, '*.pdf'))
    documents = []
    for path in tqdm.tqdm(paths):
        try:
            documents.append(PDFDocumentWithFeatures.load(path))
        except PDFDocumentLoadingError:
            print(f'Loading "{path}" failed due to no valid text box found')
    return documents
