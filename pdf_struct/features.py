# Features that are universal to text and PDF
from typing import Optional, List

import regex as re

from pdf_struct.listing import MultiLevelNumberedList, SectionNumber
from pdf_struct.transition_labels import TextBlock, ListAction


def _gt(tb) -> Optional[str]:
    # get text
    return None if tb is None else tb.text


class BaseFeatureExtractor(object):
    def init_state(self):
        self.multi_level_numbered_list = MultiLevelNumberedList()

    def extract_features(self, tb1: TextBlock, tb2: TextBlock, tb3: TextBlock, tb4: TextBlock):
        raise NotImplementedError('extract_features should be over written')

    def extract_features_all(self, text_blocks: List[TextBlock], actions: Optional[List[ListAction]]):
        self.init_state()
        for i in range(len(text_blocks)):
            tb1 = text_blocks[i - 1] if i != 0 else None
            tb2 = text_blocks[i]
            if actions is None or actions[i] == ListAction.ELIMINATE:
                tb3 = text_blocks[i + 1] if i + 1 < len(text_blocks) else None
                tb4 = text_blocks[i + 2] if i + 2 < len(text_blocks) else None
            else:
                tb3 = None
                for j in range(i + 1, len(text_blocks)):
                    if actions[j] != ListAction.ELIMINATE:
                        tb3 = text_blocks[j]
                        break
                tb4 = text_blocks[j + 1] if j + 1 < len(text_blocks) else None
            yield self.extract_features(tb1, tb2, tb3, tb4)
        self.multi_level_numbered_list = None

    def indent(self, tb1, tb2):
        raise NotImplementedError()

    def left_aligned(self, tb1):
        raise NotImplementedError

    def extract_pointer_features(self, text_blocks: List[TextBlock], list_actions: List[ListAction], i: int, j: int):
        # extract features for classifying whether j-th pointer (which
        # determines level at (j+1)-th line) should point at i-th line
        assert 0 <= i < j < len(text_blocks)

        n_downs = len([a for a in list_actions[j:i:-1] if a == ListAction.DOWN])
        n_ups = len([a for a in list_actions[j:i:-1] if a == ListAction.UP])

        for k in range(i, 0, -1):
            if list_actions[k - 1] in {ListAction.UP, ListAction.DOWN,
                                       ListAction.SAME_LEVEL}:
                head_tb = text_blocks[k]
                break
        else:
            head_tb = text_blocks[0]

        if j + 1 >= len(text_blocks):
            tb1, tb2 = text_blocks[i], text_blocks[j]

            feat = (
                -1,
                -1,
                self.indent(tb1, tb2),
                -1,
                self.indent(head_tb, tb2),
                -1,
                self.left_aligned(tb1),
                True,
                self.left_aligned(head_tb),
                n_downs,
                n_ups,
                n_ups - n_downs
            )
        else:
            tb1, tb2, tb3 = text_blocks[i], text_blocks[j], text_blocks[j+1]
            section_numbers1 = SectionNumber.extract_section_number(tb1.text)
            section_number_head = SectionNumber.extract_section_number(head_tb.text)
            section_numbers3 = SectionNumber.extract_section_number(tb3.text)

            feat = (
                SectionNumber.is_any_next_of(section_numbers3, section_numbers1),
                SectionNumber.is_any_next_of(
                    section_numbers3, section_number_head),
                self.indent(tb1, tb2),
                self.indent(tb1, tb3),
                self.indent(head_tb, tb2),
                self.indent(head_tb, tb3),
                self.left_aligned(tb1),
                self.left_aligned(tb3),
                self.left_aligned(head_tb),
                n_downs,
                n_ups,
                n_ups - n_downs
            )
        return list(map(float, feat))



def whereas(text1: Optional[str], text2: Optional[str]):
    if text2 is None:
        return False
    return text2.strip().lower().startswith('whereas')


def colon_ish(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return False
    return text1.strip()[-1] in {'-', ';', ':', ','}


def punctuated(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return True
    return text1.strip().endswith('.')


_PAT_LIST_ISH = re.compile('(?<!^)(?:,|;|and|or) *$', flags=re.IGNORECASE)


def list_ish(text1: Optional[str], text2: Optional[str]):
    if text1 is None:
        return False
    return _PAT_LIST_ISH.search(text1) is not None


_PAT_THEREFORE = re.compile('now[ ,]+ therefore', flags=re.IGNORECASE)


def therefore(text1: Optional[str], text2: Optional[str]):
    if text2 is None:
        return False
    m = _PAT_THEREFORE.match(text2.strip())
    return m is not None


def all_capital(text: Optional[str]):
    if text is None:
        return -1
    # allow non letters to be there
    return text.upper() == text


def mask_continuation(text1: Optional[str], text2: Optional[str]):
    if text1 is None or text2 is None:
        return False
    return text1.strip().endswith('__') and text2.strip().startswith('__')


def space_separated(text: Optional[str]):
    if text is None:
        return False
    n_spaces = text.strip().count(' ')
    return n_spaces / len(text)
