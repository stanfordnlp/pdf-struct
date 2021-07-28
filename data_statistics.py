import json
import os

import click
import numpy as np

from pdf_struct.core.transition_labels import load_annos, ListAction
from pdf_struct import loader


def get_max_depth(document):
    levels = [0]
    for l, p in zip(document.labels[:-1], document.pointers[:-1]):
        if l == ListAction.DOWN:
            levels.append(levels[-1] + 1)
        elif l == ListAction.UP:
            assert p is not None and p >= 0
            levels.append(levels[p])
        else:
            levels.append(levels[-1])
    return max(levels) + 1


def get_measures(values: list):
    return {
        'median': float(np.median(values)),
        'mean': float(np.mean(values)),
        'max': type(values[0])(np.max(values)),
        'min': type(values[0])(np.min(values))
    }


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = loader.pdf.load_from_directory(os.path.join('data', 'raw'), annos)
    else:
        documents = loader.text.load_from_directory(os.path.join('data', 'raw'), annos)

    max_depths = [get_max_depth(d) for d in documents]
    print(json.dumps({
        'n_text_blocks': get_measures([len(d.text_blocks) for d in documents]),
        'max_depth': get_measures(max_depths),
        'label_counts': {
            'continuous': get_measures(
                [len([l for l in d.labels if l == ListAction.CONTINUOUS])
                 for d in documents]),
            'same_level': get_measures(
                [len([l for l in d.labels if l == ListAction.SAME_LEVEL])
                 for d in documents]),
            'down': get_measures(
                [len([l for l in d.labels if l == ListAction.DOWN])
                 for d in documents]),
            'up': get_measures(
                [len([l for l in d.labels if l == ListAction.UP])
                 for d in documents]),
            'eliminated': get_measures(
                [len([l for l in d.labels if l == ListAction.ELIMINATE])
                 for d in documents])
        },
        'label_ratio': {
            'continuous': get_measures(
                [len([l for l in d.labels if l == ListAction.CONTINUOUS]) / len(d.labels)
                 for d in documents]),
            'same_level': get_measures(
                [len([l for l in d.labels if l == ListAction.SAME_LEVEL]) / len(d.labels)
                 for d in documents]),
            'down': get_measures(
                [len([l for l in d.labels if l == ListAction.DOWN]) / len(d.labels)
                 for d in documents]),
            'up': get_measures(
                [len([l for l in d.labels if l == ListAction.UP]) / len(d.labels)
                 for d in documents]),
            'eliminate': get_measures(
                [len([l for l in d.labels if l == ListAction.ELIMINATE]) / len(d.labels)
                 for d in documents])
        }
    }, indent=2))


if __name__ == '__main__':
    main()
