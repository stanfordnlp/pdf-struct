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

from functools import reduce
from typing import Optional

import editdistance
import numpy as np
import regex as re

from pdf_struct.core.clustering import cluster_positions, get_margins
from pdf_struct.core.feature_extractor import BaseFeatureExtractor, \
    pointer_feature, feature, single_input_feature, pairwise_feature
from pdf_struct.core.utils import pairwise
from pdf_struct.features import lexical
from pdf_struct.features.listing import MultiLevelNumberedList, SectionNumber, \
    NumberedListState
from pdf_struct.features.lm import compare_losses


def get_pdf_margin(clusters, n_pages):
    # I don't think header/footer rows exceed 3 respectively
    min_occurances = n_pages * (3 + 3) + 1
    return get_margins(clusters, min_occurances)


def _gt(tb) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class BasePDFFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, text_boxes):
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

    @single_input_feature([1])
    def similar_position_similar_text(self, tb):
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

    @single_input_feature([1])
    def header_region(self, tb):
        return bool(tb.bbox[3] > self.header_thresh)

    @single_input_feature([1])
    def footer_region(self, tb):
        return bool(tb.bbox[1] < self.footer_thresh)

    @pairwise_feature([(0, 1), (1, 2)])
    def line_break(self, tb1, tb2):
        if tb1 is None:
            return True
        return tb1.bbox[2] not in self.right_margin

    def _indent(self, tb1, tb2):
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
    def centered(self, tb):
        if tb is None:
            return False
        if tb.bbox[0] in self.left_margin:
            return False
        right_space = self.right_margin.mean - tb.bbox[2]
        left_space = tb.bbox[0] - self.left_margin.mean
        return abs(right_space - left_space) < 20

    def left_aligned(self, tb):
        return tb.bbox[0] in self.left_margin

    @pairwise_feature([(0, 1), (1, 2)])
    def extra_line_space(self, tb1, tb2):
        if tb1 is None or tb2 is None:
            return True
        ls = tb1.bbox[1] - tb2.bbox[1]
        return ls not in self.line_spacing

    @pairwise_feature([(0, 1), (1, 2)])
    def page_change(self, tb1, tb2):
        if tb1 is None or tb2 is None:
            return True
        return tb1.page != tb2.page

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


class PDFContractEnFeatureExtractor(BasePDFFeatureExtractor):
    @single_input_feature([1, 2])
    def dict_like(self, tb):
        if tb is None:
            return False
        return ':' in tb.text and tb.bbox[2] not in self.right_margin

    @single_input_feature([0, 1, 2])
    def page_like(self, tb):
        if tb is None:
            return False
        return ((tb.bbox[1] < 100 or tb.bbox[3] > 700) and
                re.search('[1-9]', tb.text) is not None)

    @single_input_feature([0, 1, 2])
    def page_like2(self, tb):
        if tb is None:
            return False
        return ((tb.bbox[1] < 100 or tb.bbox[3] > 700) and
                re.search('page [1-9]', tb.text, flags=re.IGNORECASE) is not None and
                len(tb.text.replace(' ', '')) < 10)

    @feature()
    def numbered_list_state(self, tb1, tb2, tb3, tb4, states):
        if states is None:
            states = MultiLevelNumberedList()
        if tb3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = states.try_append(
                SectionNumber.extract_section_number(tb3.text))
        return {
            'value': numbered_list_state.value,
            'states': states}

    @single_input_feature([2])
    def whereas(self, tb):
        return lexical.whereas(_gt(tb))

    @single_input_feature([2])
    def therefore(self, tb):
        return lexical.therefore(_gt(tb))

    @single_input_feature([0, 1])
    def colon_ish(self, tb):
        return lexical.colon_ish(_gt(tb))

    @single_input_feature([0, 1])
    def punctuated(self, tb):
        return lexical.punctuated(_gt(tb))

    @single_input_feature([1])
    def list_ish(self, tb):
        return lexical.list_ish(_gt(tb))

    @pairwise_feature([(0, 1), (1, 2)])
    def mask_continuation(self, tb1, tb2):
        return lexical.mask_continuation(_gt(tb1), _gt(tb2))

    @single_input_feature([1, 2])
    def all_capital(self, tb):
        return lexical.all_capital(_gt(tb))

    @single_input_feature([1, 2])
    def space_separated(self, tb):
        return lexical.space_separated(_gt(tb))

    @pointer_feature()
    def pointer_section_number(self, head_tb, tb1, tb2, tb3):
        if tb3 is None:
            return {
                '3_next_of_1': -1,
                '3_next_of_head': -1
            }
        else:
            section_numbers1 = SectionNumber.extract_section_number(tb1.text)
            section_number_head = SectionNumber.extract_section_number(head_tb.text)
            section_numbers3 = SectionNumber.extract_section_number(tb3.text)
            return {
                '3_next_of_1': SectionNumber.is_any_next_of(section_numbers3, section_numbers1),
                '3_next_of_head': SectionNumber.is_any_next_of(
                    section_numbers3, section_number_head)
            }


class PDFContractEnFeatureExtractorWithLM(PDFContractEnFeatureExtractor):

    @feature()
    def language_model_coherence(self, tb1, tb2, tb3, tb4):
        if tb1 is None or tb3 is None:
            loss_diff_next = 0.
            loss_diff_prev = 0.
        else:
            loss_diff_next = compare_losses('en', tb2.text, tb3.text, prev=tb1.text)
            loss_diff_prev = compare_losses('en', tb2.text, tb1.text, next=tb3.text)
        return {
            'loss_diff_next': loss_diff_next,
            'loss_diff_prev': loss_diff_prev
        }
