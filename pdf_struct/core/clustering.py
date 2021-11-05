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

from typing import List


class Cluster(object):
    def __init__(self, values, threshold: float):
        self._values = sorted(values)
        self._mean = sum(values) / len(values)
        self._threshold = threshold

    def __len__(self):
        return len(self._values)

    @property
    def mean(self):
        return self._mean

    @property
    def min(self):
        return self._values[0]

    @property
    def max(self):
        return self._values[-1]

    def __contains__(self, item):
        return (self.max - self._threshold) < item < (
                    self.min + self._threshold)

    def add(self, v: float):
        assert len(self._values) == 0 or v >= self._values[-1]
        n = len(self._values)
        self._mean = (self._mean * n + v) / (n + 1)
        self._values.append(v)

    def pop(self):
        v = self._values[0]
        n = len(self._values)
        self._mean = (self._mean * n - v) / (n - 1)
        # FIXME: Implement with more efficient queue
        self._values = self._values[1:]

    def assess(self, v: float):
        return (v - self._values[0]) <= self._threshold


def cluster_positions(positions, thresh: float):
    # greedily cluster positions
    # cluster_positions([1, 4, 6], 4) will return two clusters
    # with [1, 4], and [6] due its greedy nature
    positions = sorted(positions)
    clusters = [Cluster([positions[0]], thresh)]
    mappings = {positions[0]: 0}
    for p in positions[1:]:
        if clusters[-1].assess(p):
            clusters[-1].add(p)
        else:
            clusters.append(Cluster([p], thresh))
        mappings[p] = len(clusters) - 1
    return clusters, mappings


def get_margins(clusters: List[Cluster], min_occurances: int):
    for c in clusters:
        if len(c) >= min_occurances:
            return c
    return clusters[0]
