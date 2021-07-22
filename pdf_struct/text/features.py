from typing import Optional

import regex as re

from pdf_struct import features
from pdf_struct.clustering import get_margins, cluster_positions
from pdf_struct.feature_extractor import BaseFeatureExtractor, single_input_feature, pairwise_feature, feature, pointer_feature
from pdf_struct.listing import NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.listing import MultiLevelNumberedList, SectionNumber, \
    NumberedListState


def _gt(tb) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class PlainTextFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, text_lines):
        self.right_margin = get_margins(
            cluster_positions([l.width for l in text_lines], 8)[0][::-1], 5)

    @pairwise_feature([(0, 1), (1, 2)])
    def line_break(self, t1, t2):
        if t1 is None:
            return True
        return t1.width not in self.right_margin

    def left_aligned(self, tb):
        return tb.indent == 0

    def _indent(self, t1, t2):
        if t1 is None or t2 is None:
            return 3
        if t1.indent < t2.indent:
            return 1
        if t1.indent > t2.indent:
            return 2
        return 0

    @pairwise_feature([(0, 1), (1, 2)])
    def indent(self, tb1, tb2):
        return self._indent(tb1, tb2)

    @pairwise_feature([(0, 1), (1, 2)])
    def indent_body(self, t1, t2):
        if t1 is None or t2 is None:
            return 3
        if t1.body_indent < t2.body_indent:
            return 1
        if t1.body_indent > t2.body_indent:
            return 2
        return 0

    @single_input_feature([1, 2])
    def centered(self, t):
        if t is None:
            return False
        if t.indent == 0:
            return False
        right_space = self.right_margin.mean - t.width
        left_space = t.indent
        return abs(right_space - left_space) < 8

    @single_input_feature([1, 2])
    def extra_line_space(self, t1):
        if t1 is None:
            return -1
        return t1.top_spacing

    @single_input_feature([1, 2])
    def dict_like(self, t):
        if t is None:
            return False
        return ':' in t.text and t.width not in self.right_margin

    @single_input_feature([0, 1, 2])
    def page_like1(self, t):
        if t is None:
            return False
        return re.search('page [1-9]|PAGE', t.text) is not None

    @single_input_feature([0, 1, 2])
    def page_like2(self, t):
        if t is None:
            return False
        return re.search('[0-9]/[1-9]|- ?[0-9]+ ?-', t.text) is not None

    @single_input_feature([0, 1, 2])
    def horizontal_line(self, t):
        if t is None:
            return False
        charset = set(t.text.strip())
        charset.discard(' ')
        return len(charset) == 1 and len(t.text.strip()) >= 3 and charset.pop() in set('*-=#%_+')

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

    @feature()
    def language_model_coherence(self, tb1, tb2, tb3, tb4):
        if tb1 is None or tb3 is None:
            loss_diff_next = 0.
            loss_diff_prev = 0.
        else:
            loss_diff_next = compare_losses(tb2.text, tb3.text, prev=tb1.text)
            loss_diff_prev = compare_losses(tb2.text, tb1.text, next=tb3.text)
        return {
            'loss_diff_next': loss_diff_next,
            'loss_diff_prev': loss_diff_prev
        }

    @pairwise_feature([(1, 2)])
    def whereas(self, tb1, tb2):
        return features.whereas(_gt(tb1), _gt(tb2))

    @pairwise_feature([(1, 2)])
    def therefore(self, tb1, tb2):
        return features.therefore(_gt(tb1), _gt(tb2))

    @pairwise_feature([(0, 1), (1, 2)])
    def colon_ish(self, tb1, tb2):
        return features.colon_ish(_gt(tb1), _gt(tb2))

    @pairwise_feature([(0, 1), (1, 2)])
    def punctuated(self, tb1, tb2):
        return features.punctuated(_gt(tb1), _gt(tb2))

    @pairwise_feature([(1, 2)])
    def list_ish(self, tb1, tb2):
        return features.list_ish(_gt(tb1), _gt(tb2))

    @pairwise_feature([(0, 1), (1, 2)])
    def mask_continuation(self, tb1, tb2):
        return features.mask_continuation(_gt(tb1), _gt(tb2))

    @single_input_feature([1, 2])
    def all_capital(self, tb):
        return features.all_capital(_gt(tb))

    @single_input_feature([1, 2])
    def space_separated(self, tb):
        return features.space_separated(_gt(tb))

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
