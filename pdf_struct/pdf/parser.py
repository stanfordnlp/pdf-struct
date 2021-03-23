from typing import NamedTuple, Generator, Tuple, List, Set
import uuid
from functools import reduce
import operator

from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox, LTTextLine, LTFigure
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

from pdf_struct.preprocessing import preprocess_text
from pdf_struct.transition_labels import TextBlock


class TextBox(TextBlock):
    def __init__(self, text: str, bbox: Tuple[float, float, float, float], page: int, blocks: Set[str]):
        super(TextBox, self).__init__(text)
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bbox: Tuple[float, float, float, float] = bbox
        self.page: int = page
        self.blocks: Set[str] = blocks


def parse_pdf(fs) -> Generator[TextBox, None, None]:
    doc = PDFDocument(PDFParser(fs))
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
