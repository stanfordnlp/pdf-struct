from pdf_struct.core.document import Document
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
