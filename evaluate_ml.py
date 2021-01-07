import json
import os

import click

from pdf_struct import transition_labels, transition_predictor
from pdf_struct.pdf import load_pdfs
from pdf_struct.text import load_texts
from pdf_struct.structure_evaluation import evaluate_structure, evaluate_labels


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'), annos)
    else:
        documents = load_texts(os.path.join('data', 'raw'), annos)
    documents_pred = transition_predictor.k_fold_train_predict(documents)

    with open(os.path.join('data', f'results_{file_type}.jsonl'), 'w') as fout:
        for d, d_p in zip(documents, documents_pred):
            assert d.path == d_p.path
            fout.write(json.dumps({
                'path': d.path,
                'texts': d.texts,
                'features': [list(map(float, f)) for f in d.feats],
                'ground_truth': {
                    'labels': [l.name for l in d.labels],
                    'pointers': d.pointers
                },
                'prediction': {
                    'labels': [l.name for l in d_p.labels],
                    'pointers': d_p.pointers
                }
            }))
    print(json.dumps(evaluate_structure(documents, documents_pred), indent=2))
    print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))


if __name__ == '__main__':
    main()
