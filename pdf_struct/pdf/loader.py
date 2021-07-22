import glob
import os
import uuid
from typing import Generator, Tuple, List, Set

import tqdm
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure
from pdfminer.pdfdocument import PDFDocument as _PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from pdf_struct.bbox import merge_continuous_lines
from pdf_struct.document import Document, TextBlock
from pdf_struct.pdf.features import PDFFeatureExtractor
from pdf_struct.preprocessing import preprocess_text
from pdf_struct.transition_labels import ListAction, AnnoListType, \
    filter_text_blocks
from pdf_struct.utils import get_filename


class PDFDocumentLoadingError(ValueError):
    pass


class TextBox(TextBlock):
    def __init__(self, text: str, bbox: Tuple[float, float, float, float],
                 page: int, blocks: Set[str]):
        super(TextBox, self).__init__(text)
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bbox: Tuple[float, float, float, float] = bbox
        self.page: int = page
        self.blocks: Set[str] = blocks


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
        if isinstance(lt_obj, LTTextLine):
            text = preprocess_text(lt_obj.get_text().strip('\n'))
            if block is None:
                block = uuid.uuid4().hex
            if len(text.strip()) > 0:
                yield TextBox(text, lt_obj.bbox, page, {block})
        elif isinstance(lt_obj, LTTextBox):
            block = uuid.uuid4().hex
            yield from parse_layout(lt_obj, page, block)
        elif isinstance(lt_obj, LTFigure):
            pass


def load_pdf_document(path: str, labels: List[ListAction], pointers: List[int],
                      dummy_feats: bool=False):
    with open(path, 'rb') as fin:
        text_boxes = list(parse_pdf(fin))
    if len(text_boxes) == 0:
        raise PDFDocumentLoadingError('No text boxes found.')
    # Space size is about 4pt
    text_boxes = merge_continuous_lines(text_boxes, space_size=4)
    if len(labels) != len(text_boxes):
        raise PDFDocumentLoadingError('Number of rows does not match labels.')

    text_boxes, labels, pointers = filter_text_blocks(text_boxes, labels, pointers)
    texts = [tb.text for tb in text_boxes]

    feature_extractor, feats, feats_test, pointer_feats, pointer_candidates = \
        PDFFeatureExtractor.initialize_and_extract_all_features(
            text_boxes, labels, pointers, dummy_feats, text_boxes)

    return Document(path, feats, feats_test, texts, labels, pointers,
                    pointer_feats, pointer_candidates, feature_extractor,
                    text_boxes, path)


def load_pdf_document_for_prediction(path: str):
    with open(path, 'rb') as fin:
        text_boxes = list(parse_pdf(fin))
    if len(text_boxes) == 0:
        raise PDFDocumentLoadingError('No text boxes found.')
    # Space size is about 4pt
    text_boxes = merge_continuous_lines(text_boxes, space_size=4)

    texts = [tb.text for tb in text_boxes]

    feature_extractor = PDFFeatureExtractor(text_boxes)
    feats_test = feature_extractor.extract_features_all(text_boxes, None)

    return Document(path, None, feats_test, texts, None, None, None, None,
                    feature_extractor, text_boxes, path)


def load_pdfs_from_directory(base_dir: str, annos: AnnoListType, dummy_feats: bool=False) -> List[Document]:
    paths = glob.glob(os.path.join(base_dir, '*.pdf'))
    # filter first for tqdm to work properly
    paths = [path for path in paths if get_filename(path) in annos]
    documents = []
    for path in tqdm.tqdm(paths):
        anno = annos[get_filename(path)]
        try:
            documents.append(load_pdf_document(
                path, [a[0] for a in anno], [a[1] for a in anno], dummy_feats=dummy_feats))
        except PDFDocumentLoadingError as e:
            print(f'Loading "{path}" failed. {e}')
    return documents
