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

from pdf_struct.features.listing import en
from pdf_struct.features.listing.base import BaseSectionNumber, section_pattern, \
    register_section_pattern


# FIXME: Add Japanese specific patterns

def get_text_body_indent_ja(text: str):
    m = en._PAT_ALL_LISTING.match(text)
    return len(m.group(0))


# We generally should not care about zenkaku because it is normalized in
# pdf_struct.core.preprocessing (except for Kakoi-moji)
KAKOIMOJI = [
    '①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿',
    '㋐㋑㋒㋓㋔㋕㋖㋗㋘㋙㋚㋛㋜㋝㋞㋟㋠㋡㋢㋣㋤㋥㋦㋧㋨㋩㋪㋫㋬㋭㋮㋯㋰㋱㋲㋳㋴㋵㋶㋷㋸㋹㋺㋻㋼㋽',
    '㊀㊁㊂㊃㊄㊅㊆㊇㊈',
    'ⒶⒷⒸⒹⒺⒻⒼⒽⒾⒿⓀⓁⓂⓃⓄⓅⓆⓇⓈⓉⓊⓋⓌⓍⓎⓏ',
    'ⓐⓑⓒⓓⓔⓕⓖⓗⓘⓙⓚⓛⓜⓝⓞⓟⓠⓡⓢⓣⓤⓥⓦⓧⓨⓩ'
]
kakoimoji_to_int_dict = {c: i for _kakoimoji in KAKOIMOJI for i, c in enumerate(_kakoimoji)}


def kakoimoji_to_int(text: str) -> int:
    return kakoimoji_to_int_dict[text]


HIRAGANA = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
KATAKANA = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
HIRAGANA_IROHA = 'いろはにほへとちりぬるをわかよたれそつねならむうゐのおくやまけふこえてあさきゆめみしゑひもせすん'
KATAKANA_IROHA = 'イロハニホヘトチリヌルヲワカヨタレソツネナラムウヰノオクヤマケフコエテアサキユメミシヱヒモセスン'

hiragana_to_int_dict = {c: i for i, c in enumerate(HIRAGANA)}
katakana_to_int_dict = {c: i for i, c in enumerate(KATAKANA)}
hiragana_iroha_to_int_dict = {c: i for i, c in enumerate(HIRAGANA_IROHA)}
katakana_iroha_to_int_dict = {c: i for i, c in enumerate(KATAKANA_IROHA)}


def hiragana_to_int(text: str) -> int:
    return hiragana_to_int_dict[text]


def katakana_to_int(text: str) -> int:
    return katakana_to_int_dict[text]


def hiragana_iroha_to_int(text: str) -> int:
    return hiragana_iroha_to_int_dict[text]


def katakana_iroha_to_int(text: str) -> int:
    return katakana_iroha_to_int_dict[text]


kansuji_to_int_dict = {
    '一': 1,
    '壱': 1,
    '壹': 1,
    '二': 2,
    '弐': 2,
    '貳': 2,
    '三': 3,
    '参': 3,
    '參': 3,
    '四': 4,
    '肆': 4,
    '五': 5,
    '伍': 5,
    '六': 6,
    '陸': 6,
    '七': 7,
    '漆': 7,
    '質': 7,
    '八': 8,
    '捌': 8,
    '九': 9,
    '玖': 9,
    '〇': 0,
    '十': 10,
    '拾': 10,
    '什': 10
}

def kansuji_to_int(text: str) -> Optional[int]:
    assert 1 <= len(text) <= 3
    # we ONLY care about numbers below 100
    # Thus, there are only five possible patterns:
    # 1. A single character number (e.g., 一, 十)
    # 2. A two digits number with a "10" character & numbers for both digits (e.g., 弐拾五)
    # 3. A two digits number with a "10" character and the last digit (e.g., 拾五)
    # 4. A two digits number without the last digit (e.g., 弐拾)
    # 5. A two digits number without "10" character (e.g., 二〇, 二五)
    if len(text) == 1:
        num = kansuji_to_int_dict[text]
        if num == 0:
            # invalid number
            return None
    elif len(text) == 2:
        c0 = kansuji_to_int_dict[text[0]]
        c1 = kansuji_to_int_dict[text[1]]
        if c0 == 10:
            if c1 == 10 or c1 == 0:
                return None
            num = 10 + c1
        elif c1 == 10:
            if c0 == 0:
                return None
            num = c0 * 10
        else:
            num = c0 * 10 + c1
    else:
        c0 = kansuji_to_int_dict[text[0]]
        c1 = kansuji_to_int_dict[text[1]]
        c2 = kansuji_to_int_dict[text[2]]
        if c1 != 10:
            return None
        if c2 == 10 or c2 == 0 or c0 == 10 or c0 == 0:
            return None
        num = kansuji_to_int_dict[text[0]] * 10 + kansuji_to_int_dict[text[2]]
    assert 1 <= num < 100
    return num


_TMPLS_RE_ENUM = [
    '\({num}\)', '{num}\. ', '{num}\)', '{num}  ', '§{num}', '{num}:', '【{num}】']
_TMPLS_RE_SECTION = [
    '第? *{num} *章', '第? *{num} *節', '第? *{num} *項', '第? *{num} *条']

PATS_ALPH_LOWER = [
    re.compile(tmpl.format(num='(?P<num>[a-z])')) for tmpl in _TMPLS_RE_ENUM]
PATS_ALPH_UPPER = [
    re.compile(tmpl.format(num='(?P<num>[A-Z])')) for tmpl in _TMPLS_RE_ENUM]
PATS_NUM = [
    re.compile(tmpl.format(num='(?P<num>[1-9][0-9]*)')) for tmpl in _TMPLS_RE_ENUM + _TMPLS_RE_SECTION]

PATS_HIRAGANA = [
    re.compile(tmpl.format(num=f'(?P<num>[{HIRAGANA}])')) for tmpl in _TMPLS_RE_ENUM]
PATS_KATAKANA = [
    re.compile(tmpl.format(num=f'(?P<num>[{KATAKANA}])')) for tmpl in _TMPLS_RE_ENUM]
PATS_HIRAGANA_IROHA = [
    re.compile(tmpl.format(num=f'(?P<num>[{HIRAGANA_IROHA}])')) for tmpl in _TMPLS_RE_ENUM]
PATS_KATAKANA_IROHA = [
    re.compile(tmpl.format(num=f'(?P<num>[{KATAKANA_IROHA}])')) for tmpl in _TMPLS_RE_ENUM]

# Kansuji and Daiji. Forget about numbers above 99.
PATS_KANSUJI = [
    re.compile(tmpl.format(num=f'(?P<num>[{"".join(kansuji_to_int_dict.keys())}]{{1,3}})'))
    for tmpl in _TMPLS_RE_ENUM + _TMPLS_RE_SECTION]
PATS_ROMAN_LOWER = [
    re.compile(tmpl.format(num=f'(?P<num>{en._RE_ROMAN.lower()})')) for tmpl in _TMPLS_RE_ENUM + _TMPLS_RE_SECTION]
PATS_ROMAN_UPPER = [
    re.compile(tmpl.format(num=f'(?P<num>{en._RE_ROMAN})')) for tmpl in _TMPLS_RE_ENUM + _TMPLS_RE_SECTION]
PATS_NUM_MULTILEVEL = [
    re.compile(tmpl.format(num=en._RE_NUM_MULTILEVEL)) for tmpl in _TMPLS_RE_ENUM + _TMPLS_RE_SECTION]

PATS_KAKOIMOJI_SUJI = [
    re.compile(f'(?P<num>[{kakoimoji_set}])') for kakoimoji_set in KAKOIMOJI
]

@register_section_pattern('arabic', PATS_NUM, int)
@register_section_pattern('roman_upper', PATS_ROMAN_UPPER, en.roman_to_int)
@register_section_pattern('roman_lower', PATS_ROMAN_LOWER, en.roman_to_int)
@register_section_pattern('alph_upper', PATS_ALPH_UPPER, en.alphabet_to_int)
@register_section_pattern('alph_lower', PATS_ALPH_LOWER, en.alphabet_to_int)
@register_section_pattern('arabic_multilevel', PATS_NUM_MULTILEVEL, int)
@register_section_pattern('hiragana', PATS_HIRAGANA, hiragana_to_int)
@register_section_pattern('katakana', PATS_KATAKANA, katakana_to_int)
@register_section_pattern('hiragana_iroha', PATS_HIRAGANA_IROHA, hiragana_iroha_to_int)
@register_section_pattern('katakana_iroha', PATS_KATAKANA_IROHA, katakana_iroha_to_int)
@register_section_pattern('kansuji', PATS_KANSUJI, kansuji_to_int)
@register_section_pattern('kakoimoji', PATS_KAKOIMOJI_SUJI, kakoimoji_to_int)
class SectionNumberJa(BaseSectionNumber):

    @section_pattern()
    def bullet_point(text: str):
        m = en.PAT_BULLET_POINTS.match(text)
        return None if m is None else m.group(0)
