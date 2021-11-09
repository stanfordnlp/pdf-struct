# Copyright (c) 2021, Hitachi America Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import sys
from typing import Optional

import click
import joblib
import tqdm

from pdf_struct import loader, feature_extractor
from pdf_struct.core import transition_labels
from pdf_struct.core.data_statistics import get_documents_statistics
from pdf_struct.core.evaluation import evaluate
from pdf_struct.core.export import to_paragraphs, to_tree
from pdf_struct.core.predictor import train_classifiers, \
    predict_with_classifiers
from pdf_struct.core.download import cached_model_download


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
@click.argument('file-type', type=click.Choice(tuple(loader.modules.keys())))
@click.argument('feature',
                type=click.Choice(tuple(feature_extractor.feature_extractors.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def _evaluate(k_folds: int, prediction, metrics, file_type: str, feature: str, raw_dir, anno_dir):
    """ Evaluate pdf-struct with k-Fold cross validation.
    """
    print(f'Loading annotations from {anno_dir}')
    if file_type == 'hocr':
        annos = transition_labels.load_hocr_annos(anno_dir)
    else:
        annos = transition_labels.load_annos(anno_dir)

    print('Loading raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)
    if len(documents) == 0:
        click.echo(
            f'No matching documents found in {raw_dir}. Please check directory '
            'path and file extention.', err=True)
        exit(1)

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


@cli.command()
@click.argument('file-type', type=click.Choice(tuple(loader.modules.keys())))
@click.argument('feature',
                type=click.Choice(tuple(feature_extractor.feature_extractors.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
@click.argument('out-path', type=click.Path(exists=False))
def train(file_type: str, feature: str, raw_dir: str, anno_dir: str, out_path: str):
    """ Train a pdf-struct model with raw-dir and anno-dir as inputs and dump
    the model to out-path.
    """
    if file_type == 'hocr':
        raise NotImplementedError('data-stats does not currently support hocr')

    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)
    if len(documents) == 0:
        click.echo(
            f'No matching documents found in {raw_dir}. Please check directory '
            'path and file extention.', err=True)
        exit(1)
    feature_extractor_cls = feature_extractor.feature_extractors[feature]
    print('Extracting features from documents')
    documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

    print('Training a model')
    clf, clf_ptr = train_classifiers(documents)
    joblib.dump((clf, clf_ptr, file_type, feature), out_path)
    print(f'Model successfully dumped at {out_path}')


@cli.command()
@click.option('-o', '--out', type=click.Path(exists=False), default=None)
@click.option(
    '-f', '--format', type=click.Choice(('paragraphs', 'tabbed', 'tree')),
    default='paragraphs', help='Output format.')
@click.option(
    '-m', '--model', type=str, default=None,
    help='Specify a pretrained model from https://github.com/stanfordnlp/pdf-struct-models')
@click.option(
    '-p', '--path', type=click.Path(exists=False), default=None,
    help='Specify a local pretrained model.')
@click.argument('in-path', type=click.Path(exists=True))
def predict(out: Optional[str], format: str, model: Optional[str],
            path: Optional[str], in_path: str):
    """ Predict in-path's logical structures using the specified model. You need
    to specify either one of --model and --path.
    """
    if (model is None) == (path is None):
        raise click.UsageError('One and only one of --model and --path must be specified.')
    if model is not None:
        path = cached_model_download(model)
        if path is None:
            click.echo(f'Model "{model}" cannot be found. Aborting...', err=True)
            exit(1)
    clf, clf_ptr, file_type, feature = joblib.load(path)
    document = loader.modules[file_type].load_document(in_path, None, None)
    feature_extractor_cls = feature_extractor.feature_extractors[feature]
    document = feature_extractor_cls.append_features_to_document(document)

    if out is None:
        out_ = sys.stdout
    else:
        out_ = open(out, 'w')
    pred = predict_with_classifiers(clf, clf_ptr, [document])[0]
    if format in ('paragraphs', 'tabbed'):
        paragraphs = to_paragraphs(pred)
        if format == 'paragraphs':
            for paragraph, _ in paragraphs:
                out_.write(paragraph + '\n')
        else:
            for paragraph, level in paragraphs:
                out_.write('\t' * level + paragraph + '\n')
    elif format == 'tree':
        pred = to_tree(pred)
        out_.write(json.dumps(pred, indent=2))
    else:
        assert not 'Should not get here'
    if out is not None:
        out_.close()


@cli.command('data-stats')
@click.argument('file-type', type=click.Choice(tuple(loader.modules.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
def data_stats(file_type: str, raw_dir: str, anno_dir: str):
    """ Calculate dataset statistics.
    """
    if file_type == 'hocr':
        raise NotImplementedError('data-stats does not currently support hocr')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)

    stats = get_documents_statistics(documents)
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    cli()
