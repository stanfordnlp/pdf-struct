from typing import List, Optional

import regex as re

from pdf_struct import features
from pdf_struct.clustering import get_margins, cluster_positions
from pdf_struct.listing import \
    MultiLevelNumberedList, NumberedListState, SectionNumber
from pdf_struct.lm import compare_losses
from pdf_struct.text.parser import TextLine
from pdf_struct.utils import groupwise


def _gt(tb: Optional[TextLine]) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


def extract_features(texts: List[TextLine]):
    right_margin = get_margins(
        cluster_positions([l.width for l in texts], 8)[0][::-1],
        5)

    # Text specific features using text info
    def line_break(t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None:
            return True
        return t1.width not in right_margin

    def indent(t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None or t2 is None:
            return 3
        if t1.indent < t2.indent:
            return 1
        if t1.indent > t2.indent:
            return 2
        return 0

    def indent_body(t1: Optional[TextLine], t2: Optional[TextLine]):
        if t1 is None or t2 is None:
            return 3
        if t1.body_indent < t2.body_indent:
            return 1
        if t1.body_indent > t2.body_indent:
            return 2
        return 0

    def centered(t: Optional[TextLine]):
        if t is None:
            return False
        if t.indent == 0:
            return False
        right_space = right_margin.mean - t.width
        left_space = t.indent
        return abs(right_space - left_space) < 8

    def extra_line_space(t1: Optional[TextLine]):
        if t1 is None:
            return -1
        return t1.top_spacing

    def dict_like(t: Optional[TextLine]):
        if t is None:
            return False
        return ':' in t.text and t.width not in right_margin

    def page_like1(t: Optional[TextLine]):
        if t is None:
            return False
        return re.search('page [1-9]|PAGE', t.text) is not None

    def page_like2(t: Optional[TextLine]):
        if t is None:
            return False
        return re.search('[0-9]/[1-9]|- ?[0-9]+ ?-', t.text) is not None

    def horizontal_line(t: Optional[TextLine]):
        if t is None:
            return False
        charset = set(t.text.strip())
        charset.discard(' ')
        return len(charset) == 1 and len(t.text.strip()) >= 3 and charset.pop() in set('*-=#%_+')

    multi_level_numbered_list = MultiLevelNumberedList()
    for t1, t2, t3, t4 in groupwise(texts, 4):
        if t2 is None:
            continue
        if t3 is None:
            numbered_list_state = NumberedListState.DOWN
        else:
            numbered_list_state = multi_level_numbered_list.try_append(
                SectionNumber.extract_section_number(t3.text))
        if t3 is None or t4 is None:
            loss_diff_next = 0.
            loss_diff_prev = 0.
        else:
            loss_diff_next = compare_losses(t3.text, t4.text, prev=t2.text)
            loss_diff_prev = compare_losses(t3.text, t2.text, next=t4.text)

        feat = (
            features.whereas(_gt(t2), _gt(t3)),
            features.colon_ish(_gt(t1), _gt(t2)),
            features.colon_ish(_gt(t2), _gt(t3)),
            features.punctuated(_gt(t1), _gt(t2)),
            features.punctuated(_gt(t2), _gt(t3)),
            line_break(t1, t2),
            line_break(t2, t3),
            features.list_ish(_gt(t2), _gt(t3)),
            indent(t1, t2),
            indent(t2, t3),
            indent_body(t1, t2),
            indent_body(t2, t3),
            features.therefore(_gt(t2), _gt(t3)),
            features.all_capital(_gt(t2)),
            features.all_capital(_gt(t3)),
            features.mask_continuation(_gt(t1), _gt(t2)),
            features.mask_continuation(_gt(t2), _gt(t3)),
            features.space_separated(_gt(t2)),
            features.space_separated(_gt(t3)),
            centered(t2),
            centered(t3),
            extra_line_space(t2),
            extra_line_space(t3),
            dict_like(t2),
            dict_like(t3),
            page_like1(t1),
            page_like1(t2),
            page_like1(t3),
            page_like2(t1),
            page_like2(t2),
            page_like2(t3),
            horizontal_line(t1),
            horizontal_line(t2),
            horizontal_line(t3),
            loss_diff_next,
            loss_diff_prev,
            numbered_list_state.value
        )
        yield list(map(float, feat))
