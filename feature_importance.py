import json
import os

import click
from joblib import Parallel, delayed


from pdf_struct import transition_labels, transition_predictor
from pdf_struct.pdf import load_pdfs
from pdf_struct.structure_evaluation import evaluate_structure, evaluate_labels
from pdf_struct.text import load_texts


def single_run(documents, feature_indices, i):
    documents_pred = transition_predictor.k_fold_train_predict(
        documents, used_features=feature_indices)
    structure_metrics = evaluate_structure(documents, documents_pred)
    label_metrics = evaluate_labels(documents, documents_pred,
                                    confusion_matrix=False)
    return {
        'target_feature': i,
        'used_features': feature_indices,
        'structure_metrics': structure_metrics,
        'label_metrics': label_metrics
    }


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
@click.argument('search-method', type=click.Choice(('incr-important', 'decr-important', 'decr-unimportant')))
@click.argument('n-rounds', type=int)
@click.option('--n-jobs', type=int, default=1)
def main(file_type: str, search_method: str, n_rounds: int, n_jobs: int):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'), annos)
    else:
        documents = load_texts(os.path.join('data', 'raw'), annos)

    n_features = len(documents[0].feats[0])
    if n_rounds <= 0:
        n_rounds = n_features
    all_feature_indices = set(range(n_features))
    results = []
    cur_feature_indices = set()
    for round in range(n_rounds):
        if search_method == 'incr-important':
            results_round = Parallel(n_jobs=n_jobs)(
                delayed(single_run)(documents, sorted(cur_feature_indices | {i}), i)
                for i in all_feature_indices - cur_feature_indices)
        else:
            results_round = Parallel(n_jobs=n_jobs)(
                delayed(single_run)(documents, all_feature_indices - (cur_feature_indices | {i}), i)
                for i in all_feature_indices - cur_feature_indices)

        if search_method == 'incr-important' or search_method == 'decr-unimportant':
            result_chosen = max(
                results_round,
                key=lambda r: r['label_metrics']['accuracy']['micro'])
        else:
            result_chosen = min(
                results_round,
                key=lambda r: r['label_metrics']['accuracy']['micro'])

        print(f'Chosen feature #{result_chosen["target_feature"]} with accuracy '
              f'{result_chosen["label_metrics"]["accuracy"]["micro"]}')
        cur_feature_indices.add(result_chosen["target_feature"])
        results.append({
            'round': round,
            'chosen_feature': result_chosen["target_feature"],
            'results': results_round
        })
    with open(os.path.join('data', f'results_importance_{file_type}_{search_method}.json'), 'w') as fout:
        json.dump(results, fout, indent=2)


if __name__ == '__main__':
    main()
