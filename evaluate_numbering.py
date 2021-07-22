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
from pdf_struct.features.listing import SectionNumber


def predict_transitions_numbering(document: Document) -> Document:
    numbered_list: List[SectionNumber] = []
    anchors: List[int] = []

    labels = []
    pointers = []
    for i in range(document.n_blocks):
        candidates = SectionNumber.extract_section_number(document.texts[i])
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
                if section_number.number is None or section_number.number <= 1:
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
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = loader.pdf.load_from_directory(os.path.join('data', 'raw'), annos)
    else:
        documents = loader.pdf.load_from_directory(os.path.join('data', 'raw'), annos)

    documents_pred = [predict_transitions_numbering(document)
                      for document in documents]

    print(json.dumps(evaluate_structure(documents, documents_pred), indent=2))
    print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))


if __name__ == '__main__':
    main()
