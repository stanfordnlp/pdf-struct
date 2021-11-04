import glob
import json
import os

import click
import tqdm

from pdf_struct import loader, feature_extractor
from pdf_struct.core import transition_labels
from pdf_struct.core.evaluation import evaluate
from pdf_struct.core.data_statistics import get_documents_statistics


@click.group()
def cli():
    pass


@cli.command('init-dataset')
@click.argument('file-type', type=click.Choice(tuple(loader.modules.keys())))
@click.argument('indir', type=click.Path(exists=True))
@click.argument('outdir', type=click.Path(exists=False))
def init_dataset(file_type, indir, outdir):
    paths = glob.glob(os.path.join(indir, f'*.{file_type}'))
    os.makedirs(outdir)
    for path in tqdm.tqdm(paths):
        out_filename = os.path.splitext(os.path.basename(path))[0] + '.tsv'
        loader.modules[file_type].create_training_data(path, os.path.join(outdir, out_filename))


@cli.command('evaluate')
@click.option('-k', '--k-folds', type=int, default=5)
@click.option('--prediction', type=click.Path(exists=False), default=None,
              help='Dump prediction as a JSONL file.')
@click.option('--metrics', type=click.Path(exists=False), default=None,
              help='Dump metrics as a JSON file.')
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
@click.argument('feature',
                type=click.Choice(tuple(feature_extractor.feature_extractors.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def _evaluate(k_folds: int, prediction, metrics, file_type: str, feature: str, raw_dir, anno_dir):
    print(f'Loading annotations from {anno_dir}')
    if file_type == 'hocr':
        annos = transition_labels.load_hocr_annos(anno_dir)
    else:
        annos = transition_labels.load_annos(anno_dir)

    print('Loading raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)

    feature_extractor_cls = feature_extractor.feature_extractors[feature]

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


@cli.command('data-stats')
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def dat_stats(file_type: str, raw_dir: str, anno_dir: str):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)

    stats = get_documents_statistics(documents)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    cli()
