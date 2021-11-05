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

import regex as re

from pdf_struct.features.listing.base import BaseSectionNumber, section_pattern, \
    register_section_pattern


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

PATS_ALPH_LOWER = [
    re.compile(tmpl.format(num='(?P<num>[a-z])')) for tmpl in _TMPLS_RE_ENUM]
PATS_ALPH_UPPER = [
    re.compile(tmpl.format(num='(?P<num>[A-Z])')) for tmpl in _TMPLS_RE_ENUM]
PATS_NUM = [
    re.compile(tmpl.format(num='(?P<num>[1-9][0-9]*)')) for tmpl in _TMPLS_RE_ENUM]
PATS_ROMAN_LOWER = [
    re.compile(tmpl.format(num=f'(?P<num>{_RE_ROMAN.lower()})')) for tmpl in _TMPLS_RE_ENUM]
PATS_ROMAN_UPPER = [
    re.compile(tmpl.format(num=f'(?P<num>{_RE_ROMAN})')) for tmpl in _TMPLS_RE_ENUM]
_RE_NUM_MULTILEVEL = '[1-9][0-9]*(?:[\.-][1-9][0-9]*)*[\.-](?P<num>[1-9][0-9]*)'
PATS_NUM_MULTILEVEL = [
    re.compile(tmpl.format(num=_RE_NUM_MULTILEVEL)) for tmpl in _TMPLS_RE_ENUM]
_RE_BULLET_POINTS = '[・\-*+•‣⁃○∙◦⦾⦿\uE000-\uF8FF]'
PAT_BULLET_POINTS = re.compile(_RE_BULLET_POINTS)
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


@register_section_pattern('arabic', PATS_NUM, int)
@register_section_pattern('roman_upper', PATS_ROMAN_UPPER, roman_to_int)
@register_section_pattern('roman_lower', PATS_ROMAN_LOWER, roman_to_int)
@register_section_pattern('alph_upper', PATS_ALPH_UPPER, alphabet_to_int)
@register_section_pattern('alph_lower', PATS_ALPH_LOWER, alphabet_to_int)
@register_section_pattern('arabic_multilevel', PATS_NUM_MULTILEVEL, int)
class SectionNumber(BaseSectionNumber):

    @section_pattern()
    def bullet_point(text: str):
        m = PAT_BULLET_POINTS.match(text)
        return None if m is None else m.group(0)
