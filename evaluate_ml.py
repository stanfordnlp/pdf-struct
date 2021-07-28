import json
import os
from collections import Counter

import click
import numpy as np
import tqdm
from sklearn.metrics import accuracy_score

from pdf_struct import loader
from pdf_struct.core import predictor, transition_labels
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels
from pdf_struct.feature_extractor.pdf_contract import PDFContractEnFeatureExtractor
from pdf_struct.feature_extractor.text_contract import PlainTextFeatureExtractor


@click.command()
@click.option('-k', '--k-folds', type=int, default=5)
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
@click.argument('out-path', type=click.Path(exists=False))
def main(k_folds: int, file_type: str, raw_dir, anno_dir, out_path):
    print(f'Loading annotations from {anno_dir}')
    if file_type == 'hocr':
        annos = transition_labels.load_hocr_annos(anno_dir)
    else:
        annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'hocr':
        documents = loader.hocr.load_from_directory(raw_dir, annos)
    elif file_type == 'pdf':
        documents = loader.pdf.load_from_directory(raw_dir, annos)
        documents = [PDFContractEnFeatureExtractor.append_features_to_document(document)
                     for document in tqdm.tqdm(documents)]
    else:
        documents = loader.text.load_from_directory(raw_dir, annos)
        documents = [PlainTextFeatureExtractor.append_features_to_document(document)
                     for document in tqdm.tqdm(documents)]

    print(f'Extracted {sum(map(lambda d: d.n_blocks, documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    print(f'Extracted {documents[0].n_features}-dim features and '
          f'{documents[0].n_pointer_features}-dim pointer features.')
    documents_pred = predictor.k_fold_train_predict(
        documents, n_splits=k_folds)

    with open(os.path.join('data', out_path), 'w') as fout:
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


if __name__ == '__main__':
    main()
