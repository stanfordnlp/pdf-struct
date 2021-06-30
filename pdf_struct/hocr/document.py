import glob
import os
from typing import List

import tqdm

from pdf_struct.document import Document
from pdf_struct.hocr.features import HOCRFeatureExtractor
from pdf_struct.hocr.parser import parse_hocr
from pdf_struct.transition_labels import ListAction, AnnoListType, filter_text_blocks
from pdf_struct.utils import get_filename


class HOCRDocumentLoadingError(ValueError):
    pass


class HOCRDocumentWithFeatures(Document):
    @classmethod
    def load(cls, path: str, labels: List[ListAction], pointers: List[int], dummy_feats: bool=False) -> List['HOCRDocumentWithFeatures']:
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

            feature_extractor, feats, feats_test, pointer_feats, pointer_candidates = HOCRFeatureExtractor.initialize_and_extract_all_features(
                text_boxes, _labels, _pointers, dummy_feats, text_boxes
            )
            documents.append(cls(
                path, feats, feats_test, texts, _labels, _pointers, pointer_feats, pointer_candidates,
                feature_extractor, text_boxes, path))
        return documents

    @classmethod
    def load_pred(cls, path: str) -> List['HOCRDocumentWithFeatures']:
        with open(path) as fin:
            html_doc = fin.read()
        span_boxes_lst = parse_hocr(html_doc)

        documents = []
        for span_boxes in span_boxes_lst:
            texts = [tb.text for tb in span_boxes]

            feature_extractor = HOCRFeatureExtractor(span_boxes)
            feats_test = feature_extractor.extract_features_all(
                span_boxes, None)
            documents.append(cls(
                path, None, feats_test, texts, None, None, None, None,
                feature_extractor, span_boxes, path))
        return documents


def load_hocr(base_dir: str, annos: AnnoListType, dummy_feats: bool=False) -> List[HOCRDocumentWithFeatures]:
    paths = glob.glob(os.path.join(base_dir, '*.hocr'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.extend(HOCRDocumentWithFeatures.load(
                path, [a[0] for a in anno], [a[1] for a in anno], dummy_feats=dummy_feats))
        except HOCRDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents
