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

from itertools import chain
from typing import List

import numpy as np

from pdf_struct.core.document import Document
from pdf_struct.core.structure_evaluation import create_hierarchy_matrix
from pdf_struct.core.transition_labels import ListAction


def to_paragraphs(document: Document, insert_space=True):
    paragraphs = []
    paragraph_levels = []
    last_l, last_p = None, -1
    level = 0
    levels = []
    for l, t, p in zip(document.labels, document.texts, document.pointers):
        if l == ListAction.ELIMINATE:
            levels.append(None)  # You shouldn't be able to point at ELIMINATE
            continue
        if last_l is None or last_l == ListAction.SAME_LEVEL:
            assert last_p == -1
            paragraphs.append(t)
            paragraph_levels.append(level)
        elif last_l == ListAction.DOWN:
            assert last_p == -1
            level += 1
            paragraphs.append(t)
            paragraph_levels.append(level)
        elif last_l == ListAction.UP:
            assert last_p != -1
            level = levels[last_p]
            paragraphs.append(t)
            paragraph_levels.append(level)
        elif last_l == ListAction.CONTINUOUS:
            if insert_space:
                paragraphs[-1] = paragraphs[-1].rstrip(' ') + ' '
            paragraphs[-1] += t
        else:
            assert not 'Should not reach here'
        last_l, last_p = l, p
        levels.append(level)
    assert len(levels) == len(document.labels)
    assert len(paragraphs) == len(paragraph_levels)
    return list(zip(paragraphs, paragraph_levels))


def get_birelationship(hierarchy_matrix, target: int, relation: int) -> List[int]:
    assert relation in [0, 1]
    return sorted(set(np.where(hierarchy_matrix[target, :] == relation)[0]) |
                  set(np.where(hierarchy_matrix[:, target] == relation)[0]))


def to_tree(document: Document, insert_space=True):
    sep = ' ' if insert_space else ''
    m = create_hierarchy_matrix(document)
    assert len(m) == len(document.text_blocks)
    # since m is an upper triangle matrix, incorporate flipped matrix to
    # incorporate bidirectionality to SAME_LEVEL
    m[m.T == ListAction.SAME_LEVEL.value] = m.T[m.T  == ListAction.SAME_LEVEL.value]
    text_boxes = []
    already_added = [False] * len(m)
    id2idx = []
    for i in range(len(document.text_blocks)):
        if document.labels[i] == ListAction.ELIMINATE:
            assert not already_added[i]
            text_boxes.append({
                'text': document.text_blocks[i].text,
                'siblings': [],
                'ancestors': [],
                'parent': None,
                'descendents': [],
                'children': [],
                'eliminated': True
            })
        elif not already_added[i]:
            continual = get_birelationship(m, i, 0)
            siblings = get_birelationship(m, i, 1)
            ancestors = sorted(np.where(m[:, i] == 2)[0])
            parent = ancestors[-1] if len(ancestors) > 0 else None
            descendents = sorted(np.where(m[i, :] == 2)[0])
            if len(descendents) > 0:
                children = get_birelationship(m, descendents[0], 1)
            else:
                children = []
            assert set(descendents).issuperset(set(children))

            text_boxes.append({
                'text': sep.join((document.text_blocks[j].text for j in chain([i], continual))),
                'siblings': siblings,
                'ancestors': ancestors,
                'parent': parent,
                'descendents': descendents,
                'children': children,
                'eliminated': False
            })
            for j in continual:
                assert j > i
                already_added[j] = True
        id2idx.append(len(text_boxes) - 1)
    # now convert all ids to current indices
    for i in range(len(text_boxes)):
        for key in ('siblings', 'ancestors', 'descendents', 'children'):
            text_boxes[i][key] = sorted({id2idx[id_] for id_ in text_boxes[i][key]} - {i})
        text_boxes[i]['parent'] = None if text_boxes[i]['parent'] is None else id2idx[text_boxes[i]['parent']]
    return text_boxes
