import glob
import os
from typing import List, Tuple

import tqdm

from pdf_struct.pdf.features import PDFFeatureExtractor
from pdf_struct.pdf.parser import parse_pdf, TextBox
from pdf_struct.transition_labels import DocumentWithFeatures, ListAction, \
    AnnoListType
from pdf_struct.utils import get_filename
from pdf_struct.bbox import merge_continuous_lines


class PDFDocumentLoadingError(ValueError):
    pass


class PDFDocumentWithFeatures(DocumentWithFeatures):
    def __init__(self, path: str, feats: list, texts: List[str],
                 labels: List[ListAction], pointers: List[int],
                 pointer_feats: List[Tuple[int, int, List[float]]],
                 feature_extractor, text_boxes: List[TextBox]):
        super(PDFDocumentWithFeatures, self).__init__(
            path, feats, texts, labels, pointers, pointer_feats, feature_extractor, text_boxes)

    @classmethod
    def load(cls, path: str, labels: List[ListAction], pointers: List[int], dummy_feats: bool=False):
        with open(path, 'rb') as fin:
            text_boxes = list(parse_pdf(fin))
        if len(text_boxes) == 0:
            raise PDFDocumentLoadingError('No text boxes found.')
        # Space size is about 4pt
        text_boxes = merge_continuous_lines(text_boxes, space_size=4)
        if len(labels) != len(text_boxes):
            raise PDFDocumentLoadingError('Number of rows does not match labels.')

        text_boxes, labels, pointers = cls._filter_text_blocks(text_boxes, labels, pointers)
        texts = [tb.text for tb in text_boxes]

        feature_extractor, feats, pointer_feats = cls._extract_features(
            PDFFeatureExtractor, text_boxes, labels, pointers, dummy_feats)

        return cls(path, feats, texts, labels, pointers, pointer_feats,
                   feature_extractor, text_boxes)


def load_pdfs(base_dir: str, annos: AnnoListType, dummy_feats: bool=False) -> List[PDFDocumentWithFeatures]:
    paths = glob.glob(os.path.join(base_dir, '*.pdf'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(PDFDocumentWithFeatures.load(
                path, [a[0] for a in anno], [a[1] for a in anno], dummy_feats=dummy_feats))
        except PDFDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents
