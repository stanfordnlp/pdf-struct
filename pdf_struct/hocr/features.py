from typing import Optional, List

import numpy as np

from pdf_struct import features
from pdf_struct.clustering import cluster_positions
from pdf_struct.clustering import get_margins
from pdf_struct.hocr.parser import SpanBox
from pdf_struct.listing import NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.utils import pairwise


def _gt(tb: Optional[SpanBox]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class HOCRFeatureExtractor(features.BaseFeatureExtractor):
    def __init__(self, text_boxes: List[SpanBox]):
        self.text_boxes = text_boxes
        # bbox is [x_left, y_bottom, x_right, y_top] in points with
        # left bottom being [0, 0, 0, 0]
        self.bboxes = np.array([tb.bbox for tb in text_boxes])

        horizontal_thresh = 10  # 10 points = 1em
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

    # PDF specific features using PDF info
    def line_break(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None:
            return True
        return tb1.bbox[2] not in self.right_margin

    def indent(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return 3
        if self.mappings_l[tb1.bbox[0]] < self.mappings_l[tb2.bbox[0]]:
            return 1
        if self.mappings_l[tb1.bbox[0]] > self.mappings_l[tb2.bbox[0]]:
            return 2
        return 0

    def centered(self, tb: SpanBox):
        if tb is None:
            return False
        if tb.bbox[0] in self.left_margin:
            return False
        right_space = self.right_margin.mean - tb.bbox[2]
        left_space = tb.bbox[0] - self.left_margin.mean
        return abs(right_space - left_space) < 20

    def left_aligned(self, tb: SpanBox):
        return tb.bbox[0] in self.left_margin

    def extra_line_space(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return True
        ls = tb1.bbox[1] - tb2.bbox[1]
        return ls not in self.line_spacing

    def extract_features(self, tb1: SpanBox, tb2: SpanBox, tb3: SpanBox, tb4: SpanBox):
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = self.multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(tb3.text))

        feat = (
            features.punctuated(_gt(tb1), _gt(tb2)),
            features.punctuated(_gt(tb2), _gt(tb3)),
            self.line_break(tb1, tb2),
            self.line_break(tb2, tb3),
            self.indent(tb1, tb2),
            self.indent(tb2, tb3),
            features.therefore(_gt(tb2), _gt(tb3)),
            features.all_capital(_gt(tb2)),
            features.all_capital(_gt(tb3)),
            features.mask_continuation(_gt(tb1), _gt(tb2)),
            features.mask_continuation(_gt(tb2), _gt(tb3)),
            features.space_separated(_gt(tb2)),
            features.space_separated(_gt(tb3)),
            self.centered(tb2),
            self.centered(tb3),
            self.extra_line_space(tb1, tb2),
            self.extra_line_space(tb2, tb3),
            numbered_list_state.value
        )
        return list(map(float, feat))
