import re
from typing import Tuple, List, Set

from bs4 import BeautifulSoup

from pdf_struct.preprocessing import preprocess_text
from pdf_struct.transition_labels import TextBlock
from pdf_struct.bbox import merge_continuous_lines


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
                span_boxes_td = merge_continuous_lines(span_boxes_td, space_size=4)
                if len(span_boxes_td) > 1:
                    span_boxes.append(span_boxes_td)
    return span_boxes
