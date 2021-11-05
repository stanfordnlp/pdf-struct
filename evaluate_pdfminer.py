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
from pdf_struct.loader.pdf import load_from_directory
from pdf_struct.core.structure_evaluation import evaluate_labels
from pdf_struct.core.predictor import ListAction
from pdf_struct.core.utils import pairwise


@click.command()
@click.option('--metrics', type=click.Path(exists=False), default=None,
              help='Dump metrics as a JSON file.')
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def main(metrics, raw_dir: str, anno_dir: str):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    documents = load_from_directory(raw_dir, annos)
    documents_pred = []
    for document in documents:
        d = copy.deepcopy(document)
        labels = []
        for tb1, tb2 in pairwise(d.text_blocks):
            labels.append(
                ListAction.CONTINUOUS if len(tb1.blocks & tb2.blocks) > 0 else ListAction.SAME_LEVEL
            )
        pointers = [None] * len(labels)
        labels.append(ListAction.UP)
        pointers.append(-1)
        d.labels = labels
        d.pointers = pointers
        documents_pred.append(d)

    if metrics is None:
        print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))
    else:
        _metrics = {
            'labels': evaluate_labels(documents, documents_pred)
        }
        with open(metrics, 'w') as fout:
            json.dump(_metrics, fout, indent=2)


if __name__ == '__main__':
    main()
