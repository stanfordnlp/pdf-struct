from enum import Enum
from inspect import signature
from typing import List, Set, Optional, Union, Pattern, Type, Callable

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
_TMPLS_RE_ENUM = [
    '\({num}\)', '{num}\. ', '{num}\)', '{num}  ', '§{num}', '{num}:',
    'ARTICLE *{num}', 'Article *{num}', 'SECTION *{num}', 'Section *{num}']

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


class _SectionPatternRegisterer(type):

    def __init__(cls, name, bases, attrs):
        cls._patterns = []
        for key, val in attrs.items():
            properties = getattr(val, '_prop', None)
            if properties is not None:
                cls._patterns.append({'func': val, 'name': properties['name']})


def section_pattern(name: Optional[str] = None):
    if not (name is None or isinstance(name, str)):
        raise ValueError(f'name must be str but {type(name)} was given.')
    def _body(func: Callable[[str], Optional[Union[str, int]]]):
        nonlocal name
        n_params = len(signature(func).parameters)
        if n_params != 1:
            raise TypeError(
                '@section_pattern() must receive a function with 1 argument.')
        if name is None:
            name = func.__name__

        func._prop = {'name': name}
        return func

    return _body


def register_section_pattern(name: str, patterns: List[Pattern[str]], convert_fun: Callable[[str], int]):

    def _body(cls: Type[BaseSectionNumber]):
        nonlocal patterns, name, convert_fun
        if not issubclass(cls, BaseSectionNumber):
            raise TypeError(
                '@register_section_pattern may only be used on a class inheriting BaseSectionNumber')
        for pattern in patterns:

            def _extract_section_number_factory(pattern):
                # use factory pattern to embed pattern into _extract_section_number
                def _extract_section_number(text: str):
                    m = pattern.match(text)
                    return None if m is None else convert_fun(m.group('num'))
                return _extract_section_number

            extract_section_number = _extract_section_number_factory(pattern)
            extract_section_number._prop = {'name': name}
            cls._patterns.append(
                {'func': extract_section_number,
                 'name': name,
                 'pattern': pattern})
        return cls

    return _body


class BaseSectionNumber(object, metaclass=_SectionPatternRegisterer):

    def __init__(self, section_number_type: str, number: Union[int, str]):
        if not isinstance(section_number_type, str):
            raise TypeError(f'section_number_type must be str but {type(section_number_type)} was given.')
        if not isinstance(number, (int, str)):
            raise TypeError(f'number must be int or str but {type(number)} was given.')
        self.section_number_type: str = section_number_type
        self.number: Optional[int] = number

    def __eq__(self, other):
        return (type(self) == type(other) and
                self.number == other.number and
                self.section_number_type == other.section_number_type)

    @classmethod
    def extract_section_number(cls, text: str) -> List['BaseSectionNumber']:
        text = text.strip()
        candidates = []
        for pattern in cls._patterns:
            section_number = pattern['func'](text)
            if section_number is not None:
                candidates.append(cls(pattern['name'], section_number))
        return candidates

    def is_next_of(self, other: 'BaseSectionNumber') -> bool:
        return other.next() == self

    def next(self) -> 'BaseSectionNumber':
        if isinstance(self.number, str):
            # bullet points
            return self
        else:
            # Numbered list
            return type(self)(self.section_number_type, self.number + 1)

    @staticmethod
    def is_any_next_of(following_section_numbers: Set['BaseSectionNumber'],
                       section_numbers: Set['BaseSectionNumber']) -> bool:
        for s1 in section_numbers:
            for s2 in following_section_numbers:
                if s2.is_next_of(s1):
                    return True
        return False


@register_section_pattern('arabic', _PATS_NUM, int)
@register_section_pattern('roman_upper', _PATS_ROMAN_UPPER, roman_to_int)
@register_section_pattern('roman_lower', _PATS_ROMAN_LOWER, roman_to_int)
@register_section_pattern('alph_upper', _PATS_ALPH_UPPER, alphabet_to_int)
@register_section_pattern('alph_lower', _PATS_ALPH_LOWER, alphabet_to_int)
@register_section_pattern('arabic_multilevel', _PATS_NUM_MULTILEVEL, int)
class SectionNumber(BaseSectionNumber):

    @section_pattern()
    def bullet_point(text: str):
        m = _PAT_BULLET_POINTS.match(text)
        return None if m is None else m.group(0)


class NumberedListState(Enum):
    NO_NUM = 0
    CONSECUTIVE = 1
    DOWN = 2
    UP = 3
    UNKNOWN = 4


class MultiLevelNumberedList(object):
    def __init__(self):
        self._numbered_list: List[BaseSectionNumber] = []

    def try_append(self, candidates: Set[BaseSectionNumber]) -> NumberedListState:
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
            if isinstance(section_number.number, str) or section_number.number <= 1:
                self._numbered_list.append(section_number)
                return NumberedListState.DOWN
        return NumberedListState.UNKNOWN
