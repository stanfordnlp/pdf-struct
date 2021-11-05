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

import itertools
import os
from typing import TypeVar, Iterable, Tuple, Iterator, Union


_UT0 = TypeVar('_UT0')
_UT1 = TypeVar('_UT1')
_UT2 = TypeVar('_UT2')


def groupwise(
        iterable: Iterable[_UT0],
        n: int,
        fill: bool = True,
        fillvalue: _UT1 = None) -> Iterator[Tuple[Union[_UT0, _UT1], ...]]:
    "groupwise(s, 2) -> (None, s0), (s0,s1), (s1,s2), (s2, None)"
    iterable_copies = []
    for ni, it in enumerate(itertools.tee(iterable, n)):
        if not fill:
            for _ in range(ni):
                next(it, None)
        if fill and ni < (n - 1):
            it = itertools.chain([fillvalue] * (n - ni - 1), it)
        iterable_copies.append(it)
    if fill:
        return itertools.zip_longest(
            *iterable_copies, fillvalue=fillvalue)
    else:
        return zip(*iterable_copies)


def pairwise(iterable: Iterable[_UT2]) -> Iterator[Tuple[_UT2, _UT2]]:
    return groupwise(iterable, 2, fill=False)


def get_filename(path):
    return os.path.splitext(os.path.basename(path))[0]
