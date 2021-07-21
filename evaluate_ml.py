import json
import os
from collections import Counter

import click
import numpy as np
from sklearn.metrics import accuracy_score

from pdf_struct import transition_labels, transition_predictor
from pdf_struct.pdf import load_pdfs
from pdf_struct.structure_evaluation import evaluate_structure, evaluate_labels
from pdf_struct.text import load_texts
from pdf_struct.hocr import load_hocr, export_result


@click.command()
@click.option('-k', '--k-folds', type=int, default=5)
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
def main(k_folds: int, file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    if file_type == 'hocr':
        annos = transition_labels.load_hocr_annos(anno_dir)
    else:
        annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'hocr':
        documents = load_hocr(os.path.join('data', 'raw'), annos)
    elif file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'), annos)
    else:
        documents = load_texts(os.path.join('data', 'raw'), annos)

    print(f'Extracted {sum(map(lambda d: d.n_blocks, documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    print(f'Extracted {documents[0].n_features}-dim features and '
          f'{documents[0].n_pointer_features}-dim pointer features.')
    documents_pred = transition_predictor.k_fold_train_predict(
        documents, n_splits=k_folds)

    with open(os.path.join('data', f'results_{file_type}.jsonl'), 'w') as fout:
        for d, d_p in zip(documents, documents_pred):
            assert d.path == d_p.path
            transition_prediction_accuracy = accuracy_score(
                np.array([l.value for l in d.labels]),
                np.array([l.value for l in d_p.labels])
            )
            fout.write(json.dumps({
                'path': d.path,
                'texts': d.texts,
                'features': d.feature_array,
                'transition_prediction_accuracy': transition_prediction_accuracy,
                'ground_truth': {
                    'labels': [l.name for l in d.labels],
                    'pointers': d.pointers
                },
                'prediction': {
                    'labels': [l.name for l in d_p.labels],
                    'pointers': d_p.pointers
                }
            }))
            fout.write('\n')
    print(json.dumps(evaluate_structure(documents, documents_pred), indent=2))
    print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))
    if file_type == 'hocr':
        export_result(documents_pred, os.path.join('data', f'export_hocr'))


if __name__ == '__main__':
    main()
