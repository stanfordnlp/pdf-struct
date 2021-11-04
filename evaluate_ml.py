import json

import click

from pdf_struct import feature_extractor
from pdf_struct import loader
from pdf_struct.core import transition_labels
from pdf_struct.core.evaluation import evaluate

feature_extractor_dict = {
    'HOCRFeatureExtractor': feature_extractor.HOCRFeatureExtractor,
    'PDFContractEnFeatureExtractor': feature_extractor.PDFContractEnFeatureExtractor,
    'PDFContractJaFeatureExtractor': feature_extractor.PDFContractJaFeatureExtractor,
    'TextContractFeatureExtractor': feature_extractor.TextContractFeatureExtractor
}


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

    feature_extractor_cls = feature_extractor_dict[feature]

    if prediction is not None:
        metrics_, prediction = evaluate(
            documents, feature_extractor_cls, k_folds, True)
        with open(prediction, 'w') as fout:
            for p in prediction:
                fout.write(json.dumps(p))
                fout.write('\n')
    else:
        metrics_ = evaluate(documents, feature_extractor_cls, k_folds, False)

    if metrics is None:
        print(json.dumps(metrics_, indent=2))
    else:
        with open(metrics, 'w') as fout:
            json.dump(metrics, fout, indent=2)


if __name__ == '__main__':
    main()
