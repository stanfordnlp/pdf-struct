from typing import List, Optional

import regex as re

from pdf_struct import features
from pdf_struct.clustering import get_margins, cluster_positions
from pdf_struct.listing import \
    NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.text.parser import TextLine


def _gt(tb: Optional[TextLine]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class PlainTextFeatureExtractor(features.BaseFeatureExtractor):
    def __init__(self, text_lines: List[TextLine]):
        self.right_margin = get_margins(
            cluster_positions([l.width for l in text_lines], 8)[0][::-1], 5)

    def line_break(self, t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None:
            return True
        return t1.width not in self.right_margin

    def left_aligned(self, tb: TextLine):
        return tb.indent == 0

    def indent(self, t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None or t2 is None:
            return 3
        if t1.indent < t2.indent:
            return 1
        if t1.indent > t2.indent:
            return 2
        return 0

    @staticmethod
    def indent_body(t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None or t2 is None:
            return 3
        if t1.body_indent < t2.body_indent:
            return 1
        if t1.body_indent > t2.body_indent:
            return 2
        return 0

    def centered(self, t: Optional[TextLine]):
        if t is None:
            return False
        if t.indent == 0:
            return False
        right_space = self.right_margin.mean - t.width
        left_space = t.indent
        return abs(right_space - left_space) < 8

    @staticmethod
    def extra_line_space(t1: Optional[TextLine]):
        if t1 is None:
            return -1
        return t1.top_spacing

    def dict_like(self, t: Optional[TextLine]):
        if t is None:
            return False
        return ':' in t.text and t.width not in self.right_margin

    @staticmethod
    def page_like1(t: Optional[TextLine]):
        if t is None:
            return False
        return re.search('page [1-9]|PAGE', t.text) is not None

    @staticmethod
    def page_like2(t: Optional[TextLine]):
        if t is None:
            return False
        return re.search('[0-9]/[1-9]|- ?[0-9]+ ?-', t.text) is not None

    @staticmethod
    def horizontal_line(t: Optional[TextLine]):
        if t is None:
            return False
        charset = set(t.text.strip())
        charset.discard(' ')
        return len(charset) == 1 and len(t.text.strip()) >= 3 and charset.pop() in set('*-=#%_+')

    def extract_features(self, t1: TextLine, t2: TextLine, t3: TextLine, t4: TextLine):
        if t3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = self.multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(t3.text))

        feat = (
            self.line_break(t1, t2),
            self.line_break(t2, t3),
            self.indent(t1, t2),
            self.indent(t2, t3),
            features.space_separated(_gt(t2)),
            features.space_separated(_gt(t3)),
            self.centered(t2),
            self.centered(t3),
            self.extra_line_space(t2),
            self.extra_line_space(t3)
        )
        return list(map(float, feat))
