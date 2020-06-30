import os

import tqdm
import click
import joblib

from pdf_struct import transition_predictor
from pdf_struct.pdf import load_pdfs
from pdf_struct.text import load_texts


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
def main(file_type: str):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_predictor.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'))
    else:
        documents = load_texts(os.path.join('data', 'raw'))

    clf = transition_predictor.train_model(documents, annos)
    out_path = os.path.join('data', f'transition_parser_{file_type}.joblib')
    joblib.dump(clf, out_path)
    print(f'Dumped classifier to "{out_path}"')
    out_dir = os.path.join('data', f'text_{file_type}')
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    for document in tqdm.tqdm(documents):
        actions = transition_predictor.predict_labels(clf, document, annos)
        paragraphs = transition_predictor.construct_hierarchy(document, actions)
        out_path = os.path.join(
            out_dir, os.path.splitext(os.path.basename(document.path))[0] + '.tsv')
        transition_predictor.write_text(out_path, paragraphs)


if __name__ == '__main__':
    main()
