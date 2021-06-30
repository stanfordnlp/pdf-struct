import copy
from typing import List

from pdf_struct.listing import SectionNumber
from pdf_struct.transition_labels import ListAction, DocumentWithFeatures


def predict_transitions_numbering(document: DocumentWithFeatures) -> DocumentWithFeatures:
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
