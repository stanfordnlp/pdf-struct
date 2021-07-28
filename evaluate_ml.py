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
from pdf_struct import feature_extractor


feature_extractor_dict = {
    'HOCRFeatureExtractor': feature_extractor.HOCRFeatureExtractor,
    'PDFContractEnFeatureExtractor': feature_extractor.PDFContractEnFeatureExtractor,
    'PDFContractJaFeatureExtractor': feature_extractor.PDFContractJaFeatureExtractor,
    'TextContractFeatureExtractor': feature_extractor.TextContractFeatureExtractor
}


def dump_predictions(out_path, documents, documents_pred):
    with open(out_path, 'w') as fout:
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


@click.command()
@click.option('-k', '--k-folds', type=int, default=5)
@click.option('--prediction', type=click.Path(exists=False), default=None,
              help='Dump prediction as a JSONL file.')
@click.option('--metrics', type=click.Path(exists=False), default=None,
              help='Dump metrics as a JSON file.')
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
@click.argument('feature', type=click.Choice(tuple(feature_extractor_dict.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def main(k_folds: int, prediction, metrics, file_type: str, feature: str, raw_dir, anno_dir):
    print(f'Loading annotations from {anno_dir}')
    if file_type == 'hocr':
        annos = transition_labels.load_hocr_annos(anno_dir)
    else:
        annos = transition_labels.load_annos(anno_dir)

    print('Loading raw files')
    if file_type == 'hocr':
        documents = loader.hocr.load_from_directory(raw_dir, annos)
    elif file_type == 'pdf':
        documents = loader.pdf.load_from_directory(raw_dir, annos)
    else:
        documents = loader.text.load_from_directory(raw_dir, annos)

    print('Extracting features from documents')
    feature_extractor_cls = feature_extractor_dict[feature]
    documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

    print(f'Extracted {sum(map(lambda d: d.n_blocks, documents))} lines from '
          f'{len(documents)} documents with label distribution: '
          f'{Counter(sum(map(lambda d: d.labels, documents), []))} for evaluation.')
    print(f'Extracted {documents[0].n_features}-dim features and '
          f'{documents[0].n_pointer_features}-dim pointer features.')
    documents_pred = predictor.k_fold_train_predict(
        documents, n_splits=k_folds)

    if prediction is not None:
        dump_predictions(prediction, documents, documents_pred)

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
