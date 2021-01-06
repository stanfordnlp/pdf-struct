from typing import Optional, List

import regex as re

from pdf_struct import features
from pdf_struct.clustering import get_margins, cluster_positions
from pdf_struct.listing import \
    MultiLevelNumberedList, NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.pdf.parser import TextBox, get_margins
from pdf_struct.utils import pairwise, groupwise


def get_pdf_margin(clusters, n_pages):
    # I don't think header/footer rows exceed 3 respectively
    min_occurances = n_pages * (3 + 3) + 1
    return get_margins(clusters, min_occurances)


def _gt(tb: Optional[TextBox]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class PDFFeatureExtractor(object):
    def __init__(self, text_boxes: List[TextBox]):
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

    def init_state(self):
        self.multi_level_numbered_list = MultiLevelNumberedList()

    def extract_features_all(self, text_boxes: List[TextBox]):
        self.init_state()
        for tb1, tb2, tb3, tb4 in groupwise(text_boxes, 4):
            if tb2 is None:
                continue
            yield self.extract_features(tb1, tb2, tb3, tb4)
        self.multi_level_numbered_list = None

    def extract_features(self, tb1: TextBox, tb2: TextBox, tb3: TextBox, tb4: TextBox):
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = self.multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(tb3.text))
        if tb3 is None or tb4 is None:
            loss_diff_next = 0.
            loss_diff_prev = 0.
        else:
            loss_diff_next = compare_losses(tb3.text, tb4.text, prev=tb2.text)
            loss_diff_prev = compare_losses(tb3.text, tb2.text, next=tb4.text)

        feat = (
            features.whereas(_gt(tb2), _gt(tb3)),
            features.colon_ish(_gt(tb1), _gt(tb1)),
            features.punctuated(_gt(tb1), _gt(tb1)),
            features.punctuated(_gt(tb1), _gt(tb1)),
            self.line_break(tb1, tb2),
            self.line_break(tb2, tb3),
            features.list_ish(_gt(tb1), _gt(tb1)),
            self.indent(tb1, tb2),
            self.indent(tb2, tb3),
            features.therefore(_gt(tb1), _gt(tb1)),
            features.therefore(_gt(tb1), _gt(tb1)),
            features.all_capital(_gt(tb1)),
            features.all_capital(_gt(tb1)),
            features.mask_continuation(_gt(tb1), _gt(tb1)),
            features.mask_continuation(_gt(tb1), _gt(tb1)),
            features.space_separated(_gt(tb1)),
            features.space_separated(_gt(tb1)),
            self.centered(tb2),
            self.centered(tb3),
            self.extra_line_space(tb1, tb2),
            self.extra_line_space(tb2, tb3),
            self.dict_like(tb2),
            self.dict_like(tb3),
            self.page_like(tb1),
            self.page_like(tb2),
            self.page_like(tb3),
            self.page_like2(tb1),
            self.page_like2(tb2),
            self.page_like2(tb3),
            self.page_change(tb1, tb2),
            self.page_change(tb2, tb3),
            loss_diff_next,
            loss_diff_prev,
            numbered_list_state.value
        )
        return list(map(float, feat))
