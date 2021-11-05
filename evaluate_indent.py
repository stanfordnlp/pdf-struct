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

import copy
import json
import os

import click

from pdf_struct.core import transition_labels
from pdf_struct.core.clustering import cluster_positions
from pdf_struct.core.predictor import ListAction
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels
from pdf_struct.core.utils import pairwise
from pdf_struct import loader


@click.command()
@click.option('--metrics', type=click.Path(exists=False), default=None,
              help='Dump metrics as a JSON file.')
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def main(metrics, file_type: str, raw_dir: str, anno_dir: str):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = loader.pdf.load_from_directory(raw_dir, annos)
        documents_pred = []
        for document in documents:
            horizontal_thresh = 10  # 10 points = 1em
            line_spacing_thresh = 2  # 2 points = 1ex / 2

            clusters_l, mappings_l = cluster_positions(
                [b.bbox[0] for b in document.text_blocks], horizontal_thresh)
            clusters_s, mappings_s = cluster_positions(
                [b1.bbox[1] - b2.bbox[1]
                 for b1, b2 in pairwise(sorted(document.text_blocks, key=lambda b: (
                    b.page, -b.bbox[1], b.bbox[0])))
                 if b1.page == b2.page],
                line_spacing_thresh
            )
            line_spacing = max(clusters_s, key=lambda c: len(c))

            labels = []
            pointers = []
            clusters = [clusters_l[mappings_l[document.text_blocks[0].bbox[0]]]]
            for i in range(1, len(document.text_blocks)):
                c_i = clusters_l[mappings_l[document.text_blocks[i].bbox[0]]]
                if clusters[-1] == c_i:
                    ls = document.text_blocks[i-1].bbox[1] - document.text_blocks[i].bbox[1]
                    if document.text_blocks[i-1].page == document.text_blocks[i].page and ls in line_spacing:
                        # normal line spacing
                        labels.append(ListAction.CONTINUOUS)
                    else:
                        labels.append(ListAction.SAME_LEVEL)
                    pointers.append(None)
                elif clusters[-1].mean < c_i.mean:
                    labels.append(ListAction.DOWN)
                    pointers.append(None)
                elif clusters[-1].mean > c_i.mean:
                    labels.append(ListAction.UP)
                    for j in range(i - 1, -1, -1):
                        if clusters[j] is not None and clusters[j] == c_i:
                            pointers.append(j)
                            break
                        # Disable non-matching cluster to avoid matching to counsins
                        clusters[j] = None
                    else:
                        pointers.append(-1)
                clusters.append(c_i)
            labels.append(ListAction.UP)
            pointers.append(-1)
            d = copy.deepcopy(document)
            d.labels = labels
            d.pointers = pointers
            documents_pred.append(d)

    else:
        documents = loader.text.load_from_directory(raw_dir, annos)
        documents_pred = []
        for document in documents:
            labels = []
            pointers = []
            indent_history = [document.text_blocks[0].indent]
            for i in range(1, len(document.text_blocks)):
                indent = document.text_blocks[i].indent
                if indent_history[-1] == indent:
                    if document.text_blocks[i].top_spacing:
                        labels.append(ListAction.SAME_LEVEL)
                    else:
                        labels.append(ListAction.CONTINUOUS)
                    pointers.append(None)
                elif indent_history[-1] < indent:
                    labels.append(ListAction.DOWN)
                    pointers.append(None)
                elif indent_history[-1] > indent:
                    labels.append(ListAction.UP)
                    for j in range(i - 1, -1, -1):
                        if indent_history[j] is not None and indent_history[j] == indent:
                            pointers.append(j)
                            break
                        # Disable non-matching cluster to avoid matching to counsins
                        indent_history[j] = None
                    else:
                        pointers.append(-1)
                indent_history.append(indent)
            labels.append(ListAction.UP)
            pointers.append(-1)
            d = copy.deepcopy(document)
            d.labels = labels
            d.pointers = pointers
            documents_pred.append(d)

    if metrics is None:
        print(json.dumps(evaluate_structure(documents, documents_pred), indent=2))
        print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))
    else:
        _metrics = {
            'structure': evaluate_structure(documents, documents_pred),
            'labels': evaluate_labels(documents, documents_pred)
        }
        with open(metrics, 'w') as fout:
            json.dump(_metrics, fout, indent=2)


if __name__ == '__main__':
    main()
