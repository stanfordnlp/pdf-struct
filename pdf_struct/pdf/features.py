from typing import Optional, List
from functools import reduce

import regex as re
import numpy as np
import editdistance

from pdf_struct import features
from pdf_struct.clustering import get_margins, cluster_positions
from pdf_struct.listing import NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.pdf.parser import TextBox, get_margins
from pdf_struct.utils import pairwise


def get_pdf_margin(clusters, n_pages):
    # I don't think header/footer rows exceed 3 respectively
    min_occurances = n_pages * (3 + 3) + 1
    return get_margins(clusters, min_occurances)


def _gt(tb: Optional[TextBox]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class PDFFeatureExtractor(features.BaseFeatureExtractor):
    def __init__(self, text_boxes: List[TextBox]):
        self.text_boxes = text_boxes
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bboxes = np.array([tb.bbox for tb in text_boxes])
        self.pages = np.array([tb.page for tb in text_boxes])

        horizontal_thresh = 10  # 10 points = 1em
        line_spacing_thresh = 2  # 2 points = 1ex / 2
        n_pages = len(set(t.page for t in text_boxes))
        clusters_l, self.mappings_l = cluster_positions(
            [b.bbox[0] for b in text_boxes], horizontal_thresh)
        self.left_margin = get_pdf_margin(clusters_l, n_pages)
        self.right_margin = get_pdf_margin(
            cluster_positions([b.bbox[2] for b in text_boxes],
                              horizontal_thresh)[0][::-1],
            n_pages)
        clusters_s, mappings_s = cluster_positions(
            [b1.bbox[1] - b2.bbox[1]
             for b1, b2 in pairwise(sorted(text_boxes, key=lambda b: (
            b.page, -b.bbox[1], b.bbox[0])))
             if b1.page == b2.page],
            line_spacing_thresh
        )
        self.line_spacing = max(clusters_s, key=lambda c: len(c))
        self.multi_level_numbered_list = None
        page_top = self.bboxes[:, 3].max()
        page_bottom = self.bboxes[:, 1].min()
        header_footer_ratio = 0.15
        self.header_thresh = page_top - header_footer_ratio * (page_top - page_bottom)
        self.footer_thresh = page_bottom + header_footer_ratio * (page_top - page_bottom)

    def similar_position_similar_text(self, tb: TextBox):
        # FIXME: this is O(n^2) operation when called for each tb
        iou_thresh = 0.5
        editdistance_thresh = 0.1
        overlap_widths = (np.minimum(tb.bbox[2], self.bboxes[:, 2]) -
                          np.maximum(tb.bbox[0], self.bboxes[:, 0]))
        overlap_heights = (np.minimum(tb.bbox[3], self.bboxes[:, 3]) -
                           np.maximum(tb.bbox[1], self.bboxes[:, 1]))
        span_widths = (np.maximum(tb.bbox[2], self.bboxes[:, 2]) -
                       np.minimum(tb.bbox[0], self.bboxes[:, 0]))
        span_heights = (np.maximum(tb.bbox[3], self.bboxes[:, 3]) -
                        np.minimum(tb.bbox[1], self.bboxes[:, 1]))
        # not exacly iou but its approximation
        iou = (overlap_widths * overlap_heights) / (span_widths * span_heights)
        mask = reduce(np.logical_and, (
            overlap_widths > 0, overlap_heights > 0, iou > iou_thresh,
            tb.page != self.pages))
        for j in np.where(mask)[0]:
            tb2 = self.text_boxes[j]
            d = editdistance.eval(tb.text, tb2.text) / max(len(tb.text), len(tb2.text))
            if editdistance_thresh > d:
                return True
        return False

    def header_region(self, tb: TextBox):
        return tb.bbox[3] > self.header_thresh

    def footer_region(self, tb: TextBox):
        return tb.bbox[1] < self.footer_thresh

    # PDF specific features using PDF info
    def line_break(self, tb1: TextBox, tb2: TextBox):
        if tb1 is None:
            return True
        return tb1.bbox[2] not in self.right_margin

    def indent(self, tb1: TextBox, tb2: TextBox):
        if tb1 is None or tb2 is None:
            return 3
        if self.mappings_l[tb1.bbox[0]] < self.mappings_l[tb2.bbox[0]]:
            return 1
        if self.mappings_l[tb1.bbox[0]] > self.mappings_l[tb2.bbox[0]]:
            return 2
        return 0

    def centered(self, tb: TextBox):
        if tb is None:
            return False
        if tb.bbox[0] in self.left_margin:
            return False
        right_space = self.right_margin.mean - tb.bbox[2]
        left_space = tb.bbox[0] - self.left_margin.mean
        return abs(right_space - left_space) < 20

    def left_aligned(self, tb: TextBox):
        return tb.bbox[0] in self.left_margin

    def extra_line_space(self, tb1: TextBox, tb2: TextBox):
        if tb1 is None or tb2 is None:
            return True
        ls = tb1.bbox[1] - tb2.bbox[1]
        return ls not in self.line_spacing

    def dict_like(self, tb: TextBox):
        if tb is None:
            return False
        return ':' in tb.text and tb.bbox[2] not in self.right_margin

    def page_like(self, tb: TextBox):
        if tb is None:
            return False
        return ((tb.bbox[1] < 100 or tb.bbox[3] > 700) and
                re.search('[1-9]', tb.text) is not None)

    def page_like2(self, tb: TextBox):
        if tb is None:
            return False
        return ((tb.bbox[1] < 100 or tb.bbox[3] > 700) and
                re.search('page [1-9]', tb.text, flags=re.IGNORECASE) is not None and
                len(tb.text.replace(' ', '')) < 10)

    def page_change(self, tb1: TextBox, tb2: TextBox):
        if tb1 is None or tb2 is None:
            return True
        return tb1.page != tb2.page

    def extract_features(self, tb1: TextBox, tb2: TextBox, tb3: TextBox, tb4: TextBox):
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = self.multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(tb3.text))

        feat = (
            self.line_break(tb1, tb2),
            self.line_break(tb2, tb3),
            self.indent(tb1, tb2),
            self.indent(tb2, tb3),
            features.space_separated(_gt(tb2)),
            features.space_separated(_gt(tb3)),
            self.centered(tb2),
            self.centered(tb3),
            self.extra_line_space(tb1, tb2),
            self.extra_line_space(tb2, tb3),
            self.similar_position_similar_text(tb2),
            self.page_change(tb1, tb2),
            self.page_change(tb2, tb3),
            self.footer_region(tb2),
            self.header_region(tb2)
        )
        return list(map(float, feat))
