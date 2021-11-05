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

import json
import os
from collections import defaultdict
from typing import List

import numpy as np

from pdf_struct.core.document import Document
from pdf_struct.core.export import get_birelationship
from pdf_struct.core.structure_evaluation import create_hierarchy_matrix


def to_ids(cell: Document, indices: List[int]) -> List[str]:
    return [b for i in indices for b in cell.text_blocks[i].blocks]


def export_result(cells: List[Document], out_dir: str):
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    documents = defaultdict(list)
    for cell in cells:
        documents[cell.path].append(cell)
    for path, document_cells in documents.items():
        spans = dict()
        for cell in document_cells:
            # (i, j) shows the relationship (-1: no relationship, 0: same paragraph,
            # 1: same level, 2: line j is below line i)
            m = create_hierarchy_matrix(cell)
            assert len(m) == len(cell.text_blocks)
            for i in range(len(cell.text_blocks)):
                continual = to_ids(cell, get_birelationship(m, i, 0))
                siblings = to_ids(cell, get_birelationship(m, i, 1))
                ancestors = sorted(np.where(m[:, i] == 2)[0])
                parent = to_ids(cell, [ancestors[-1]]) if len(ancestors) > 0 else []
                ancestors = to_ids(cell, ancestors)
                descendents = sorted(np.where(m[i, :] == 2)[0])
                if len(descendents) > 0:
                    child = descendents[0]
                    children = to_ids(cell, get_birelationship(m, child, 1))
                else:
                    children = []
                descendents = to_ids(cell, descendents)
                assert set(descendents).issuperset(set(children))

                for b in cell.text_blocks[i].blocks:
                    continual_ = sorted((set(continual) | cell.text_blocks[i].blocks) - {b})
                    spans[b] = {
                        'text': cell.text_blocks[i].text,
                        'continual': continual_,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'parent': parent,
                        'descendents': descendents,
                        'children': children
                    }
        out_path = os.path.join(
            out_dir, os.path.splitext(os.path.basename(path))[0] + '.json')
        with open(out_path, 'w') as fout:
            json.dump(spans, fout, indent=2)
