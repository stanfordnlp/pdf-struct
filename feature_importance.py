import json
import os

import click

from pdf_struct import transition_labels, transition_predictor
from pdf_struct.pdf import load_pdfs
from pdf_struct.structure_evaluation import evaluate_structure, evaluate_labels
from pdf_struct.text import load_texts


@click.command()
@click.argument('file-type', type=click.Choice(('txt', 'pdf')))
@click.argument('search-method', type=click.Choice(('incr-important', 'decr-important', 'decr-unimportant')))
@click.argument('n-rounds', type=int)
def main(file_type: str, search_method: str, n_rounds: int):
    anno_dir = os.path.join('data', f'anno_{file_type}')
    print(f'Loading annotations from {anno_dir}')
    annos = transition_labels.load_annos(anno_dir)

    print('Loading and extracting features from raw files')
    if file_type == 'pdf':
        documents = load_pdfs(os.path.join('data', 'raw'), annos)
    else:
        documents = load_texts(os.path.join('data', 'raw'), annos)

    import pdb; pdb.set_trace()
    n_features = len(documents[0].feats[0])
    if n_rounds <= 0:
        n_rounds = n_features
    all_feature_indices = set(range(n_features))
    results = []
    cur_feature_indices = set()
    for round in range(n_rounds):
        metrics = []
        results_round = []
        for i in all_feature_indices - cur_feature_indices:
            if search_method == 'incr-important':
                feature_indices = sorted(cur_feature_indices | {i})
            else:
                feature_indices = sorted(
                    all_feature_indices - (cur_feature_indices | {i}))
            documents_pred = transition_predictor.k_fold_train_predict(documents, used_features=feature_indices)
            structure_metrics = evaluate_structure(documents, documents_pred)
            label_metrics = evaluate_labels(documents, documents_pred, confusion_matrix=False)
            results_round.append({
                'target_feature': i,
                'used_features': feature_indices,
                'structure_metrics': structure_metrics,
                'label_metrics': label_metrics
            })
            metrics.append((i, label_metrics['accuracy']['micro']))
        if search_method == 'incr-important' or search_method == 'decr-unimportant':
            chosen_ind, acc = max(metrics, key=lambda i_m: i_m[1])
        else:
            chosen_ind, acc = min(metrics, key=lambda i_m: i_m[1])

        print(f'Chosen feature #{chosen_ind} with accuracy {acc}')
        cur_feature_indices.add(chosen_ind)
        results.append({
            'round': round,
            'chosen_feature': chosen_ind,
            'results': results_round
        })
    with open(os.path.join('data', f'results_importance_{file_type}_{search_method}.json'), 'w') as fout:
        json.dump(results, fout)


if __name__ == '__main__':
    main()
