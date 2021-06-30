from typing import Optional, List

import numpy as np
import regex as re

from pdf_struct.clustering import cluster_positions
from pdf_struct.clustering import get_margins
from pdf_struct.feature_extractor import BaseFeatureExtractor
from pdf_struct.hocr.parser import SpanBox
from pdf_struct.listing import NumberedListState, SectionNumber
from pdf_struct.utils import pairwise


def _gt(tb: Optional[SpanBox]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


def longest_common_substring(s1, s2):
    # adopted from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
    return s1[x_longest - longest: x_longest]


class HOCRFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, text_boxes: List[SpanBox]):
        self.text_boxes = text_boxes
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bboxes = np.array([tb.bbox for tb in text_boxes])

        # take 5% points in order to remove outliers
        self.char_per_pt = sorted(
            (len(tb.text) / (tb.bbox[2] - tb.bbox[0])
            for tb in text_boxes))[int(len(text_boxes) * 0.05)]
        assert self.char_per_pt > 0

        horizontal_thresh = 8  # 10 points = 1em
        line_spacing_thresh = 2  # 2 points = 1ex / 2
        clusters_l, self.mappings_l = cluster_positions(
            [b.bbox[0] for b in text_boxes], horizontal_thresh)
        self.left_margin = get_margins(clusters_l, 0)
        self.right_margin = get_margins(
            cluster_positions(
                [b.bbox[2] for b in text_boxes], horizontal_thresh)[0][::-1],
            0)
        clusters_s, mappings_s = cluster_positions(
            [b1.bbox[1] - b2.bbox[1]
             for b1, b2 in pairwise(sorted(text_boxes, key=lambda b: (-b.bbox[1], b.bbox[0])))],
            line_spacing_thresh
        )
        self.line_spacing = max(clusters_s, key=lambda c: len(c))
        self.multi_level_numbered_list = None
        # if text is surrounded by parentheses or not.
        # we allow partial parenthesis at start/end because OCR sometimes
        # fail to extract those.
        self._RE_PARENTHESIZED = re.compile(
            r'(?:^\([^\)]+\)$)|(?:^[^\(]+\)$)|(?:^\([^\)]+$)'
        )

    # HOCR specific features using PDF info
    def indent(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return 3
        if self.mappings_l[tb1.bbox[0]] < self.mappings_l[tb2.bbox[0]]:
            return 1
        if self.mappings_l[tb1.bbox[0]] > self.mappings_l[tb2.bbox[0]]:
            return 2
        return 0

    def centered(self, tb: SpanBox):
        # classify if it is centered by looking at other texts
        if tb is None:
            return False
        if tb.bbox[0] in self.left_margin:
            return False
        right_space = self.right_margin.mean - tb.bbox[2]
        left_space = tb.bbox[0] - self.left_margin.mean
        return abs(right_space - left_space) < 20

    def centered_cell(self, tb: SpanBox):
        # classify if it is centered by purely looking at cell geometry
        if tb is None:
            return False
        right_space = tb.cell_size[0] - tb.bbox[2]
        left_space = tb.bbox[0]
        return abs(right_space - left_space) < 20

    def left_aligned(self, tb: SpanBox):
        if tb is None:
            return False
        return tb.bbox[0] in self.left_margin

    def right_aligned(self, tb: SpanBox):
        if tb is None:
            return False
        return tb.bbox[0] in self.right_margin

    def stretched(self, tb: SpanBox):
        return (len(tb.text) / (tb.bbox[2] - tb.bbox[0]) - self.char_per_pt) > 1

    def extra_line_space(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return True
        ls = tb1.bbox[1] - tb2.bbox[1]
        return ls not in self.line_spacing

    def parenthesized(self, tb):
        # whether or not it is surrounded by parenthesis
        if tb is None:
            return False
        return self._RE_PARENTHESIZED.search(tb.text) is not None

    def common_substrings(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return False
        return len(longest_common_substring(tb1.text, tb2.text)) >= 2

    def dict_nobu(self, tb: SpanBox):
        if tb is None:
            return False
        return 'の部' in tb.text

    def dict_sonota(self, tb: SpanBox):
        if tb is None:
            return False
        return 'その他' in tb.text

    def dict_uchi(self, tb: SpanBox):
        if tb is None:
            return False
        return 'うち' in tb.text

    def dict_gokei(self, tb: SpanBox):
        if tb is None:
            return False
        return '合計' in tb.text

    def extract_features(self, tb1: SpanBox, tb2: SpanBox, tb3: SpanBox, tb4: SpanBox):
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = self.multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(tb3.text))

        feat = (
            self.indent(tb1, tb2),
            self.indent(tb2, tb3),
            self.centered(tb2),
            self.centered(tb3),
            self.centered_cell(tb2),
            self.centered_cell(tb3),
            self.left_aligned(tb2),
            self.left_aligned(tb3),
            self.right_aligned(tb2),
            self.right_aligned(tb3),
            self.stretched(tb2),
            self.extra_line_space(tb1, tb2),
            self.extra_line_space(tb2, tb3),
            self.parenthesized(tb2),
            self.parenthesized(tb3),
            self.common_substrings(tb1, tb2),
            self.common_substrings(tb2, tb3),
            self.dict_nobu(tb2),
            self.dict_sonota(tb2),
            self.dict_sonota(tb3),
            self.dict_uchi(tb2),
            self.dict_uchi(tb3),
            self.dict_gokei(tb2),
            self.dict_gokei(tb3),
            numbered_list_state.value
        )
        return list(map(float, feat))
