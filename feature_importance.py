import json

import click
import tqdm
from joblib import Parallel, delayed

from pdf_struct import feature_extractor
from pdf_struct import loader
from pdf_struct.core import predictor, transition_labels
from pdf_struct.core.structure_evaluation import evaluate_structure, \
    evaluate_labels


def single_run(documents, feature_indices, i):
    documents_pred = predictor.k_fold_train_predict(
        documents, used_features=feature_indices)
    structure_metrics = evaluate_structure(documents, documents_pred)
    label_metrics = evaluate_labels(documents, documents_pred,
                                    confusion_matrix=False)
    feature_names = documents[0].get_feature_names()
    return {
        'target_feature': feature_names[i],
        'target_feature_index': i,
        'used_features': [feature_names[j] for j in feature_indices],
        'structure_metrics': structure_metrics,
        'label_metrics': label_metrics
    }


@click.command()
@click.option('--search-method',
              type=click.Choice(('incr-important', 'decr-important', 'decr-unimportant')),
              default='incr-important')
@click.option('--n-rounds', type=int, default=0)
@click.option('--n-jobs', type=int, default=1)
@click.argument('file-type', type=click.Choice(('hocr', 'txt', 'pdf')))
@click.argument('feature', type=click.Choice(tuple(feature_extractor.feature_extractors.keys())))
@click.argument('raw-dir', type=click.Path(exists=True))
@click.argument('anno-dir', type=click.Path(exists=True))
@click.argument('out-path', type=click.Path(exists=False))
def main(search_method: str, n_rounds: int, n_jobs: int, file_type: str,
         feature: str, raw_dir: str, anno_dir: str, out_path: str):
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading raw files')
    documents = loader.modules[file_type].load_from_directory(raw_dir, annos)

    print('Extracting features from documents')
    feature_extractor_cls = feature_extractor.feature_extractors[feature]
    documents = [feature_extractor_cls.append_features_to_document(document)
                 for document in tqdm.tqdm(documents)]

    if n_rounds <= 0:
        if search_method[:4] == 'incr':
            n_rounds = documents[0].n_features
        else:
            n_rounds = documents[0].n_features - 1
    all_feature_indices = set(range(documents[0].n_features))
    print(f'Search amongst {len(all_feature_indices)} features with '
          f'"{search_method}" appoach.')
    results = []
    cur_feature_indices = set()
    for round in range(n_rounds):
        if search_method == 'incr-important':
            results_round = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(single_run)(documents, sorted(cur_feature_indices | {i}), i)
                for i in all_feature_indices - cur_feature_indices)
        else:
            results_round = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(single_run)(documents, sorted(all_feature_indices - (cur_feature_indices | {i})), i)
                for i in all_feature_indices - cur_feature_indices)

        if search_method == 'incr-important' or search_method == 'decr-unimportant':
            result_chosen = max(
                results_round,
                key=lambda r: r['label_metrics']['accuracy']['micro'])
        else:
            result_chosen = min(
                results_round,
                key=lambda r: r['label_metrics']['accuracy']['micro'])

        print(f'Chosen feature "{result_chosen["target_feature"]}" with accuracy'
              f' {result_chosen["label_metrics"]["accuracy"]["micro"]}')
        cur_feature_indices.add(result_chosen["target_feature_index"])
        results.append({
            'round': round,
            'chosen_feature': result_chosen["target_feature"],
            'results': results_round
        })
        # save every round in case something goes wrong
        with open(out_path, 'w') as fout:
            json.dump(results, fout, indent=2)


if __name__ == '__main__':
    main()
