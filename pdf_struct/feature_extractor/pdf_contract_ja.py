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

from typing import Optional

import regex as re

from pdf_struct.core.feature_extractor import \
    pointer_feature, feature, single_input_feature, pairwise_feature
from pdf_struct.feature_extractor.pdf_contract import BasePDFFeatureExtractor
from pdf_struct.features import lexical
from pdf_struct.features.listing import MultiLevelNumberedList, SectionNumberJa, \
    NumberedListState
from pdf_struct.features.lm import compare_losses


def _gt(tb) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class PDFContractJaFeatureExtractor(BasePDFFeatureExtractor):
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
                re.search('(?:page|ページ|P\\.?) [1-9]', tb.text, flags=re.IGNORECASE) is not None and
                len(tb.text.replace(' ', '')) < 10)

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

    @single_input_feature([0, 1])
    def colon_ish(self, tb):
        if tb is None:
            return False
        return tb.text.strip()[-1] in {'-', ';', ':', ',', '、'}

    @single_input_feature([0, 1])
    def punctuated(self, tb):
        if tb is None:
            return True
        text = tb.text.strip()
        return text.endswith('.') or text.endswith('。')

    @pairwise_feature([(0, 1), (1, 2)])
    def mask_continuation(self, tb1, tb2):
        return lexical.mask_continuation(_gt(tb1), _gt(tb2))

    @single_input_feature([1, 2])
    def space_separated(self, tb):
        return lexical.space_separated(_gt(tb))

    @single_input_feature([0, 1])
    def space_emphasis(self, tb):
        if tb is None:
            return False
        return re.match('^(?:[^ ] +)+[^ ]$', tb.text.strip()) is not None

    @single_input_feature([0, 1])
    def parenthesis_emphasis(self, tb):
        if tb is None:
            return False
        text = tb.text.strip()
        m = re.search('<[^>]+>|\\([^\\)]+\\)|\\[[^\\]]+\\]|【[^】]+】|〔[^〕]+〕',
                      text)
        if m is None:
            return False
        # we compare length becase we sometimes have stuff like "1. (定義)"
        return (m.span()[1] - m.span()[0]) > (len(text) / 2)

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


class PDFContractJaFeatureExtractorWithLM(PDFContractJaFeatureExtractor):

    @feature()
    def language_model_coherence(self, tb1, tb2, tb3, tb4):
        if tb1 is None or tb3 is None:
            loss_diff_next = 0.
            loss_diff_prev = 0.
        else:
            loss_diff_next = compare_losses('ja', tb2.text, tb3.text, prev=tb1.text)
            loss_diff_prev = compare_losses('ja', tb2.text, tb1.text, next=tb3.text)
        return {
            'loss_diff_next': loss_diff_next,
            'loss_diff_prev': loss_diff_prev
        }
