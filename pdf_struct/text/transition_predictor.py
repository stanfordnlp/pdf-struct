import glob
import os
from typing import List

import tqdm

from pdf_struct.text.features import extract_features
from pdf_struct.text.parser import TextLine
from pdf_struct.transition_predictor import DocumentWithFeatures


class TextDocumentWithFeatures(DocumentWithFeatures):
    def __init__(self, path: str, feats: list, texts: List[str]):
        self._path = path
        self._feats = feats
        self._texts = texts

    @classmethod
    def load(cls, path: str):
        with open(path, 'r') as fin:
            text_lines = TextLine.from_lines([line for line in fin])
        texts = [tb.text for tb in text_lines]
        feats = list(extract_features(text_lines))
        return cls(path, feats, texts)


def load_texts(base_dir: str) -> List[TextDocumentWithFeatures]:
    paths = glob.glob(os.path.join(base_dir, '*.txt'))
    documents = []
    for path in tqdm.tqdm(paths):
        documents.append(TextDocumentWithFeatures.load(path))
    return documents
