# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, List

import numpy as np
import regex as re

from pdf_struct.core.clustering import cluster_positions, get_margins
from pdf_struct.core.feature_extractor import BaseFeatureExtractor, \
    single_input_feature, pairwise_feature, pointer_feature, feature
from pdf_struct.core.utils import pairwise
from pdf_struct.features.lexical import longest_common_substring
from pdf_struct.features.listing import NumberedListState, SectionNumberJa, \
    MultiLevelNumberedList
from pdf_struct.loader.hocr import SpanBox


def _gt(tb: Optional[SpanBox]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


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

    def _indent(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return 3
        if self.mappings_l[tb1.bbox[0]] < self.mappings_l[tb2.bbox[0]]:
            return 1
        if self.mappings_l[tb1.bbox[0]] > self.mappings_l[tb2.bbox[0]]:
            return 2
        return 0

    @pairwise_feature([(0, 1), (1, 2)])
    def indent(self, tb1, tb2):
        return self._indent(tb1, tb2)

    @single_input_feature([1, 2])
    def centered(self, tb: SpanBox):
        # classify if it is centered by looking at other texts
        if tb is None:
            return False
        if tb.bbox[0] in self.left_margin:
            return False
        right_space = self.right_margin.mean - tb.bbox[2]
        left_space = tb.bbox[0] - self.left_margin.mean
        return abs(right_space - left_space) < 20

    @single_input_feature([1, 2])
    def centered_cell(self, tb: SpanBox):
        # classify if it is centered by purely looking at cell geometry
        if tb is None:
            return False
        right_space = tb.cell_size[0] - tb.bbox[2]
        left_space = tb.bbox[0]
        return abs(right_space - left_space) < 20

    @single_input_feature([1, 2])
    def left_aligned(self, tb: SpanBox):
        if tb is None:
            return False
        return tb.bbox[0] in self.left_margin

    @single_input_feature([1, 2])
    def right_aligned(self, tb: SpanBox):
        if tb is None:
            return False
        return tb.bbox[0] in self.right_margin

    @single_input_feature([1])
    def stretched(self, tb: SpanBox):
        return (len(tb.text) / (tb.bbox[2] - tb.bbox[0]) - self.char_per_pt) > 1

    @pairwise_feature([(0, 1), (1, 2)])
    def extra_line_space(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return True
        ls = tb1.bbox[1] - tb2.bbox[1]
        return ls not in self.line_spacing

    @single_input_feature([1, 2])
    def parenthesized(self, tb):
        # whether or not it is surrounded by parenthesis
        if tb is None:
            return False
        return self._RE_PARENTHESIZED.search(tb.text) is not None

    @pairwise_feature([(0, 1), (1, 2)])
    def common_substrings(self, tb1: SpanBox, tb2: SpanBox):
        if tb1 is None or tb2 is None:
            return False
        return len(longest_common_substring(tb1.text, tb2.text)) >= 2

    @single_input_feature([1])
    def dict_nobu(self, tb: SpanBox):
        if tb is None:
            return False
        return 'の部' in tb.text

    @single_input_feature([1, 2])
    def dict_sonota(self, tb: SpanBox):
        if tb is None:
            return False
        return 'その他' in tb.text

    @single_input_feature([1, 2])
    def dict_uchi(self, tb: SpanBox):
        if tb is None:
            return False
        return 'うち' in tb.text

    @single_input_feature([1, 2])
    def dict_gokei(self, tb: SpanBox):
        if tb is None:
            return False
        return '合計' in tb.text

    @feature()
    def numbered_list_state(self, tb1, tb2, tb3, tb4, states):
        if states is None:
            states = MultiLevelNumberedList()
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = states.try_append(
                SectionNumberJa.extract_section_number(tb3.text))
        return {
            'value': numbered_list_state.value,
            'states': states}

    @pointer_feature()
    def pointer_section_number(self, head_tb, tb1, tb2, tb3):
        if tb3 is None:
            return {
                '3_next_of_1': -1,
                '3_next_of_head': -1
            }
        else:
            section_numbers1 = SectionNumberJa.extract_section_number(tb1.text)
            section_number_head = SectionNumberJa.extract_section_number(head_tb.text)
            section_numbers3 = SectionNumberJa.extract_section_number(tb3.text)
            return {
                '3_next_of_1': SectionNumberJa.is_any_next_of(section_numbers3, section_numbers1),
                '3_next_of_head': SectionNumberJa.is_any_next_of(
                    section_numbers3, section_number_head)
            }

    @pointer_feature()
    def pointer_indent(self, head_tb, tb1, tb2, tb3):
        feats = {
            '1_2': self._indent(tb1, tb2),
            'head_2': self._indent(head_tb, tb2)
        }
        if tb3 is None:
            feats.update({
                '1_3': -1,
                'head_3': -1
            })
        else:
            feats.update({
                '1_3': self._indent(tb1, tb3),
                'head_3': self._indent(head_tb, tb3)
            })
        return feats

    @pointer_feature()
    def pointer_left_aligned(self, head_tb, tb1, tb2, tb3):
        return {
            '1': self.left_aligned(tb1),
            '3': True if tb3 is None else self.left_aligned(tb3),
            'head': self.left_aligned(head_tb)
        }
