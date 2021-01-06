import glob
import os
from typing import List, Tuple

import tqdm

from pdf_struct.pdf.features import PDFFeatureExtractor
from pdf_struct.pdf.parser import parse_pdf, merge_continuous_lines, TextBox
from pdf_struct.transition_labels import DocumentWithFeatures, ListAction, \
    AnnoListType
from pdf_struct.utils import get_filename


class PDFDocumentLoadingError(ValueError):
    pass


class PDFDocumentWithFeatures(DocumentWithFeatures):
    def __init__(self, path: str, feats: list, texts: List[str],
                 labels: List[ListAction], pointers: List[int],
                 pointer_feats: List[Tuple[int, int, List[float]]],
                 feature_extractor, text_boxes: List[TextBox]):
        super(PDFDocumentWithFeatures, self).__init__(
            path, feats, texts, labels, pointers, pointer_feats, feature_extractor)
        self._text_boxes = text_boxes

    @property
    def text_boxes(self) -> List[TextBox]:
        return self._text_boxes

    @classmethod
    def load(cls, path: str, labels: List[ListAction], pointers: List[int]):
        with open(path, 'rb') as fin:
            text_boxes = list(parse_pdf(fin))
        if len(text_boxes) == 0:
            raise PDFDocumentLoadingError('No text boxes found.')
        text_boxes = merge_continuous_lines(text_boxes)
        if len(labels) != len(text_boxes):
            raise PDFDocumentLoadingError('Number of rows does not match labels.')

        texts = []
        _labels = []
        _pointers = []
        _text_boxes = []
        n_removed = 0
        for i in range(len(labels)):
            if labels[i] != ListAction.EXCLUDED:
                texts.append(text_boxes[i].text)
                _labels.append(labels[i])
                if pointers[i] is None:
                    p = None
                elif pointers[i] == -1:
                    p = -1
                else:
                    p = pointers[i] - n_removed
                _pointers.append(p)
                _text_boxes.append(text_boxes[i])
            else:
                n_removed += 1
        feature_extractor = PDFFeatureExtractor(_text_boxes)
        feats = list(feature_extractor.extract_features_all(_text_boxes))
        pointer_feats = []
        for j, p in enumerate(_pointers):
            if p is not None:
                assert p >= 0
                for i in range(j):
                    if _labels[i] == ListAction.DOWN:
                        feat = feature_extractor.extract_pointer_features(
                            _text_boxes, _labels[:j], i, j)
                        pointer_feats.append((i, j, feat))

        return cls(path, feats, texts, _labels, _pointers, pointer_feats,
                   feature_extractor, _text_boxes)


def load_pdfs(base_dir: str, annos: AnnoListType) -> List[PDFDocumentWithFeatures]:
    paths = glob.glob(os.path.join(base_dir, '*.pdf'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(PDFDocumentWithFeatures.load(
                path, [a[0] for a in anno], [a[1] for a in anno]))
        except PDFDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents
