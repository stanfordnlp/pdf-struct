import json
import os
from collections import defaultdict
from typing import List

import numpy as np

from pdf_struct.hocr.transition_predictor import HOCRDocumentWithFeatures
from pdf_struct.structure_evaluation import create_hierarchy_matrix


def get_birelationship(hierarchy_matrix, target: int, relation: int) -> List[int]:
    assert relation in [0, 1]
    return sorted(set(np.where(hierarchy_matrix[target, :] == relation)[0]) |
                  set(np.where(hierarchy_matrix[:, target] == relation)[0]))


def to_ids(cell: HOCRDocumentWithFeatures, indices: List[int]) -> List[str]:
    return [b for i in indices for b in cell.text_blocks[i].blocks]


def export_result(cells: List[HOCRDocumentWithFeatures], out_dir: str):
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