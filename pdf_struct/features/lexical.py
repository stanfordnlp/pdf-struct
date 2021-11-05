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

# Features that are universal to text and PDF
from typing import Optional

import regex as re


def whereas(text: Optional[str]):
    if text is None:
        return False
    return text.strip().lower().startswith('whereas')


def colon_ish(text: Optional[str]):
    if text is None:
        return False
    return text.strip()[-1] in {'-', ';', ':', ','}


def punctuated(text: Optional[str]):
    if text is None:
        return True
    return text.strip().endswith('.')


_PAT_LIST_ISH = re.compile('(?<!^)(?:,|;|and|or) *$', flags=re.IGNORECASE)


def list_ish(text: Optional[str]):
    if text is None:
        return False
    return _PAT_LIST_ISH.search(text) is not None


_PAT_THEREFORE = re.compile('now[ ,]+ therefore', flags=re.IGNORECASE)


def therefore(text: Optional[str]):
    if text is None:
        return False
    m = _PAT_THEREFORE.match(text.strip())
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


def longest_common_substring(s1: str, s2: str):
    # adopted from https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
       for y in range(1, 1 + len(s2)):
           if s1[x - 1] == s2[y - 1]:
               m[x][y] = m[x - 1][y - 1] + 1
               if m[x][y] > longest:
                   longest = m[x][y]
                   x_longest = x
           else:
               m[x][y] = 0
    return s1[x_longest - longest: x_longest]
