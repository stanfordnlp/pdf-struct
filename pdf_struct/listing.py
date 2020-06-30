from enum import Enum
from enum import Enum
from typing import List, Set, Tuple, Optional, Union, NamedTuple
import itertools

import regex as re


def roman_to_int(expr: str):
    """ Convert a Roman numeral to an integer.
    Adopted from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s24.html
    """
    expr = expr.upper()
    nums = {'M':1000, 'D':500, 'C':100, 'L':50, 'X':10, 'V':5, 'I':1}
    sum = 0
    for i in range(len(expr)):
        try:
            value = nums[expr[i]]
            # If the next place holds a larger number, this value is negative
            if i+1 < len(expr) and nums[expr[i+1]] > value:
                sum -= value
            else: sum += value
        except KeyError:
            raise ValueError(f'expr is not a valid Roman numeral:{expr}')
    return sum


# Only allows until 39
_RE_ROMAN = 'X{0,2}I[XV]|V?I{0,3}'
_TMPLS_RE_ENUM = ['\({num}\)', '{num}\. ', '{num}\)', '{num}  ', '§{num}', '{num}:']

_PATS_ALPH_LOWER = [
    re.compile(tmpl.format(num='(?P<num>[a-z])')) for tmpl in _TMPLS_RE_ENUM]
_PATS_ALPH_UPPER = [
    re.compile(tmpl.format(num='(?P<num>[A-Z])')) for tmpl in _TMPLS_RE_ENUM]
_PATS_NUM = [
    re.compile(tmpl.format(num='(?P<num>[1-9][0-9]*)')) for tmpl in _TMPLS_RE_ENUM]
_PATS_ROMAN_LOWER = [
    re.compile(tmpl.format(num=f'(?P<num>{_RE_ROMAN.lower()})')) for tmpl in _TMPLS_RE_ENUM]
_PATS_ROMAN_UPPER = [
    re.compile(tmpl.format(num=f'(?P<num>{_RE_ROMAN})')) for tmpl in _TMPLS_RE_ENUM]
_RE_NUM_MULTILEVEL = '[1-9][0-9]*(?:[\.-][1-9][0-9]*)*[\.-](?P<num>[1-9][0-9]*)'
_PATS_NUM_MULTILEVEL = [
    re.compile(tmpl.format(num=_RE_NUM_MULTILEVEL)) for tmpl in _TMPLS_RE_ENUM]
_RE_BULLET_POINTS = '[・\-*+•‣⁃○∙◦⦾⦿\uE000-\uF8FF]'
_PAT_BULLET_POINTS = re.compile(_RE_BULLET_POINTS)
_RE_ALL_LIST_NUM = (f'(?:[a-zA-Z]|[1-9][0-9]*|(?:{_RE_ROMAN.lower()})|(?:{_RE_ROMAN})|'
                    '(?:[1-9][0-9]*(?:[\.-][1-9][0-9]*)*[\.-][1-9][0-9]*))')
_PAT_ALL_LISTING = re.compile(
    ' *(?:{}|{})? *'.format(
        '|'.join(tmpl.format(num=_RE_ALL_LIST_NUM) for tmpl in _TMPLS_RE_ENUM),
        _RE_BULLET_POINTS
    )
)


def get_text_body_indent(text: str):
    m = _PAT_ALL_LISTING.match(text)
    return len(m.group(0))


def alphabet_to_int(expr: str):
    expr = expr.lower()
    return ord(expr) - ord('a') + 1


class SectionNumberType(Enum):
    NUM = 1
    ALPH_UPPER = 2
    ALPH_LOWER = 3
    ROMAN_UPPER = 4
    ROMAN_LOWER = 5
    NUM_MULTILEVEL = 6
    BULLET_POINT = 7


class SectionNumber(NamedTuple):
    number: Optional[int]
    section_number_type: Tuple[SectionNumberType, Union[str, int]]

    def is_next_of(self, other: 'SectionNumber') -> bool:
        return other.next() == self

    def next(self) -> 'SectionNumber':
        if self.number is None:
            # bullet points
            return self
        else:
            # Numbered list
            return SectionNumber(self.number + 1, self.section_number_type)

    @staticmethod
    def extract_section_number(text: str) -> Set['SectionNumber']:
        text = text.strip()
        candidates = set()

        def _extract_section_number(
                patterns: list, num_type: SectionNumberType, conv_fun):
            for i, pat in enumerate(patterns):
                m = pat.match(text)
                if m is not None:
                    s = SectionNumber(conv_fun(m.group('num')), (num_type, i))
                    candidates.add(s)

        _extract_section_number(_PATS_NUM, SectionNumberType.NUM, int)
        _extract_section_number(
            _PATS_ROMAN_UPPER, SectionNumberType.ROMAN_UPPER, roman_to_int)
        _extract_section_number(
            _PATS_ROMAN_LOWER, SectionNumberType.ROMAN_LOWER, roman_to_int)
        _extract_section_number(
            _PATS_ALPH_UPPER, SectionNumberType.ALPH_UPPER, alphabet_to_int)
        _extract_section_number(
            _PATS_ALPH_LOWER, SectionNumberType.ALPH_LOWER, alphabet_to_int)
        _extract_section_number(
            _PATS_NUM_MULTILEVEL, SectionNumberType.NUM_MULTILEVEL, int)

        m = _PAT_BULLET_POINTS.match(text)
        if m is not None:
            candidates.add(
                SectionNumber(None, (SectionNumberType.BULLET_POINT, m.group(0))))
        return candidates


class NumberedListState(Enum):
    NO_NUM = 0
    CONSECUTIVE = 1
    DOWN = 2
    UP = 3
    UNKNOWN = 4


class MultiLevelNumberedList(object):
    def __init__(self):
        self._numbered_list: List[SectionNumber] = []

    def try_append(self, candidates: Set[SectionNumber]) -> NumberedListState:
        if len(candidates) == 0:
            return NumberedListState.NO_NUM
        for i, section_number_pre in enumerate(self._numbered_list[::-1]):
            for section_number in candidates:
                if section_number.is_next_of(section_number_pre):
                    self._numbered_list = self._numbered_list[:-i-1]
                    self._numbered_list.append(section_number)
                    if i == 0:
                        return NumberedListState.CONSECUTIVE
                    else:
                        return NumberedListState.UP
        # No valid continuation found... check if it is a new level
        for section_number in candidates:
            if section_number.number is None or section_number.number <= 1:
                self._numbered_list.append(section_number)
                return NumberedListState.DOWN
        return NumberedListState.UNKNOWN


