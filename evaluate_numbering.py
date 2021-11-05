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
from typing import List

import click

from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct.core.document import Document
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels
from pdf_struct.core.transition_labels import ListAction
from pdf_struct.features.listing import SectionNumber, SectionNumberJa


section_number_cls_dict = {
    'SectionNumber': SectionNumber,
    'SectionNumberJa': SectionNumberJa
}


def predict_transitions_numbering(section_number_cls, document: Document) -> Document:
    numbered_list = []
    anchors: List[int] = []

    labels = []
    pointers = []
    for i in range(document.n_blocks):
        candidates = section_number_cls.extract_section_number(document.texts[i])
        if len(candidates) == 0:
            labels.append(ListAction.CONTINUOUS)
            pointers.append(None)
            continue
        for j in range(len(numbered_list) - 1, -1, -1):
            for section_number in candidates:
                if section_number.is_next_of(numbered_list[j]):
                    if j == len(numbered_list) - 1:
                        labels.append(ListAction.SAME_LEVEL)
                        pointers.append(None)
                    else:
                        labels.append(ListAction.UP)
                        pointers.append(anchors[j])
                    numbered_list = numbered_list[:j]
                    numbered_list.append(section_number)
                    anchors = anchors[:j]
                    anchors.append(i)
                    break
            else:
                continue
            break
        else:
            # No valid continuation found... check if it is a new level
            for section_number in candidates:
                if isinstance(section_number.number, str) or section_number.number <= 1:
                    numbered_list.append(section_number)
                    anchors.append(i)
                    labels.append(ListAction.DOWN)
                    pointers.append(None)
                    break
            else:
                # section number does not match anything, but it is still probably a new paragraph
                labels.append(ListAction.SAME_LEVEL)
                pointers.append(None)

    # append final label --- which would always be ignored
    labels.append(ListAction.UP)
    pointers.append(-1)
    labels = labels[1:]
    pointers = pointers[1:]
    assert len(labels) == len(pointers) == len(document.labels)

    document = copy.deepcopy(document)
    document.pointers = pointers
    document.labels = labels

    return document



@click.command()
@click.option('--metrics', type=click.Path(exists=False), default=None,
              help='Dump metrics as a JSON file.')
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
@click.argument('section-number', type=click.Choice(tuple(section_number_cls_dict.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def main(metrics, file_type: str, section_number: str, raw_dir: str, anno_dir: str):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = loader.pdf.load_from_directory(raw_dir, annos)
    else:
        documents = loader.text.load_from_directory(raw_dir, annos)

    section_number_cls = section_number_cls_dict[section_number]
    documents_pred = [predict_transitions_numbering(section_number_cls, document)
                      for document in documents]

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
