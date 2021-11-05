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

from enum import Enum
from inspect import signature
from typing import List, Set, Optional, Union, Pattern, Type, Callable


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

    def __repr__(self):
        return f'{str(type(self))[:-1]} type={self.section_number_type}, number={self.number}>'


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
