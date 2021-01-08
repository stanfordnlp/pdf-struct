import json
import os

import click

from pdf_struct import transition_labels
from pdf_struct.number_based_predictor import predict_transitions_numbering
from pdf_struct.pdf import load_pdfs
from pdf_struct.structure_evaluation import evaluate_structure, evaluate_labels
from pdf_struct.text import load_texts


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'), annos, dummy_feats=True)
    else:
        documents = load_texts(os.path.join('data', 'raw'), annos)

    documents_pred = [predict_transitions_numbering(document)
                      for document in documents]

    print(json.dumps(evaluate_structure(documents, documents_pred), indent=2))
    print(json.dumps(evaluate_labels(documents, documents_pred), indent=2))


if __name__ == '__main__':
    main()
