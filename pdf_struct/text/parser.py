from typing import List

import regex as re

from pdf_struct.listing import get_text_body_indent
from pdf_struct.preprocessing import preprocess_text
from pdf_struct.transition_labels import TextBlock
from pdf_struct.utils import groupwise


class TextLine(TextBlock):
    _PAT_INDENT = re.compile(' *')

    def __init__(self, l_before, l, l_next):
        super(TextLine, self).__init__(l.strip())
        self.indent = self._get_indent(l)
        self.width = self._get_line_width(l)
        self.top_spacing = l_before is None or len(l_before.strip()) == 0
        self.bottom_spacing = l_next is None or len(l_next.strip()) == 0
        self.body_indent = get_text_body_indent(l)

    @staticmethod
    def _get_indent(text: str) -> int:
        m = TextLine._PAT_INDENT.match(text)
        return len(m.group(0))

    @staticmethod
    def _get_line_width(text: str) -> int:
        return len(text.rstrip())

    @classmethod
    def from_lines(cls, lines: List[str]) -> List['TextLine']:
        text_lines = []
        lines = [preprocess_text(l) for l in lines]
        for l_before, l, l_next in groupwise(lines, 3):
            # ignore empty line, but use it determin top/bottom spacing of
            # adjacent line
            if l is not None and len(l.strip()) > 0:
                text_lines.append(cls(l_before, l, l_next))
        return text_lines
