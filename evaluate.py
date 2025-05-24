from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np


def evaluate_model_comprehensive(model, test_X, test_y, train_X, train_y, model_name, param_info):
    """
    Perform a comprehensive evaluation of the model.
    """
    # Perform predictions
    y_pred = model.predict(test_X)
    y_pred_proba = None

    try:
        y_pred_proba = model.predict_proba(
            test_X)[:, 1]  # positive class probability
    except:
        pass  # Some models may not support predict_proba

    # Calculate basic metrics
    accuracy = accuracy_score(test_y, y_pred)
    precision = precision_score(test_y, y_pred, average='binary')
    recall = recall_score(test_y, y_pred, average='binary')
    f1 = f1_score(test_y, y_pred, average='binary')

    # ROC-AUC (if probability prediction is possible)
    roc_auc = None
    if y_pred_proba is not None:
        try:
            roc_auc = roc_auc_score(test_y, y_pred_proba)
        except:
            pass

    # Confusion Matrix
    cm = confusion_matrix(test_y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Save results
    result = {
        "model_name": model_name,
        "parameters": param_info,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn
    }

    return result


def print_evaluation_summary(all_results, preprocessing_info=None):
    """
    Print comprehensive evaluation summary including preprocessing information.
    """
    # Summary of all results
    print("\n" + "=" * 80)
    print("Overall Results Summary (Top 5 F1-Score)")
    print("=" * 80)

    # Sort by F1-Score
    sorted_results = sorted(
        all_results, key=lambda x: x['f1_score'], reverse=True)

    for i, result in enumerate(sorted_results[:5], 1):
        print(f"{i}. {result['model_name']}")
        print(f"   Parameters: {result['parameters']}")
        print(f"   Accuracy: {result['accuracy']:.4f}, "
              f"Precision: {result['precision']:.4f}, "
              f"Recall: {result['recall']:.4f}, "
              f"F1: {result['f1_score']:.4f}")
        if result['roc_auc'] is not None:
            print(f"   ROC-AUC: {result['roc_auc']:.4f}")

        # If optimization results, display additional information
        if 'optuna_cv_score' in result:
            print(
                f"   Optuna CV F1: {result['optuna_cv_score']:.4f} ({result['optuna_n_trials']} trials)")
        elif 'grid_search_cv_score' in result:
            print(
                f"   GridSearch CV F1: {result['grid_search_cv_score']:.4f} ({result['grid_search_type']} grid)")
        print()

    # Evaluation method-based highest performance model
    print("=" * 80)
    print("Best Performance by Metric")
    print("=" * 80)

    metrics = [
        ('accuracy', 'ACCURACY'),
        ('precision', 'PRECISION'),
        ('recall', 'RECALL'),
        ('f1_score', 'F1-SCORE')
    ]

    if any(r['roc_auc'] is not None for r in all_results):
        metrics.append(('roc_auc', 'ROC-AUC'))

    for metric, display_name in metrics:
        if metric == 'roc_auc':
            valid_results = [r for r in all_results if r[metric] is not None]
            if not valid_results:
                continue
            best = max(valid_results, key=lambda x: x[metric])
        else:
            best = max(all_results, key=lambda x: x[metric])

        print(f"{display_name}: {best[metric]:.4f} - {best['model_name']}")

    # Print preprocessing information if provided
    if preprocessing_info:
        print("\n" + "=" * 80)
        print("Preprocessing Information")
        print("=" * 80)
        print(f"Scaler: {preprocessing_info.get('scaler', 'N/A')}")
        print(f"Encoder: {preprocessing_info.get('encoder', 'N/A')}")
        if 'train_shape' in preprocessing_info:
            print(f"Train Data Shape: {preprocessing_info['train_shape']}")
        if 'test_shape' in preprocessing_info:
            print(f"Test Data Shape: {preprocessing_info['test_shape']}")
        if 'features' in preprocessing_info:
            print(f"Number of Features: {preprocessing_info['features']}")
        print("=" * 80)


def compare_preprocessing_results(preprocessing_results):
    """
    Compare results from different preprocessing techniques.
    """
    print("\n" + "=" * 80)
    print("PREPROCESSING TECHNIQUE COMPARISON")
    print("=" * 80)

    # Group results by preprocessing technique
    grouped_results = {}
    for result in preprocessing_results:
        prep_key = f"{result['preprocessing_info']['scaler']}_{result['preprocessing_info']['encoder']}"
        if prep_key not in grouped_results:
            grouped_results[prep_key] = {
                'preprocessing_info': result['preprocessing_info'],
                'results': []
            }
        grouped_results[prep_key]['results'].extend(result['model_results'])

    # Calculate average metrics for each preprocessing technique
    prep_performance = []
    for prep_key, data in grouped_results.items():
        results = data['results']
        prep_info = data['preprocessing_info']

        avg_metrics = {
            'preprocessing': prep_key,
            'scaler': prep_info['scaler'],
            'encoder': prep_info['encoder'],
            'avg_accuracy': np.mean([r['accuracy'] for r in results]),
            'avg_precision': np.mean([r['precision'] for r in results]),
            'avg_recall': np.mean([r['recall'] for r in results]),
            'avg_f1': np.mean([r['f1_score'] for r in results]),
            'best_f1': max([r['f1_score'] for r in results]),
            'best_accuracy': max([r['accuracy'] for r in results]),
            'best_precision': max([r['precision'] for r in results]),
            'best_recall': max([r['recall'] for r in results]),
            'best_model': max(results, key=lambda x: x['f1_score'])['model_name'],
            'num_models': len(results)
        }

        # Calculate average ROC-AUC if available
        roc_scores = [r['roc_auc']
                      for r in results if r['roc_auc'] is not None]
        if roc_scores:
            avg_metrics['avg_roc_auc'] = np.mean(roc_scores)
            avg_metrics['best_roc_auc'] = max(roc_scores)
        else:
            avg_metrics['avg_roc_auc'] = None
            avg_metrics['best_roc_auc'] = None

        prep_performance.append(avg_metrics)

    # Sort by best F1-score instead of average
    prep_performance.sort(key=lambda x: x['best_f1'], reverse=True)

    print("Performance Ranking by Best F1-Score:")
    print("-" * 80)
    for i, prep in enumerate(prep_performance, 1):
        print(
            f"{i}. Scaler: {prep['scaler'].upper()}, Encoder: {prep['encoder'].upper()}")
        print(
            f"   Best F1-Score: {prep['best_f1']:.4f} ({prep['best_model']})")
        print(f"   Best Accuracy: {prep['best_accuracy']:.4f}")
        print(f"   Best Precision: {prep['best_precision']:.4f}")
        print(f"   Best Recall: {prep['best_recall']:.4f}")
        if prep['best_roc_auc'] is not None:
            print(f"   Best ROC-AUC: {prep['best_roc_auc']:.4f}")
        print(f"   Average F1-Score: {prep['avg_f1']:.4f}")
        print(f"   Models Tested: {prep['num_models']}")
        print()

    # Find best preprocessing technique for each metric
    print("Best Preprocessing Technique by Metric:")
    print("-" * 80)

    metrics_to_check = [
        ('best_f1', 'Best F1-Score'),
        ('best_accuracy', 'Best Accuracy'),
        ('best_precision', 'Best Precision'),
        ('best_recall', 'Best Recall'),
        ('avg_f1', 'Average F1-Score'),
        ('avg_accuracy', 'Average Accuracy'),
        ('avg_precision', 'Average Precision'),
        ('avg_recall', 'Average Recall')
    ]

    if any(prep['avg_roc_auc'] is not None for prep in prep_performance):
        metrics_to_check.append(('avg_roc_auc', 'Average ROC-AUC'))
        metrics_to_check.append(('best_roc_auc', 'Best ROC-AUC'))

    for metric_key, metric_name in metrics_to_check:
        if metric_key.startswith('avg_roc') or metric_key.startswith('best_roc'):
            valid_preps = [
                p for p in prep_performance if p[metric_key] is not None]
            if not valid_preps:
                continue
            best_prep = max(valid_preps, key=lambda x: x[metric_key])
        else:
            best_prep = max(prep_performance, key=lambda x: x[metric_key])

        print(f"{metric_name}: {best_prep[metric_key]:.4f} - "
              f"Scaler: {best_prep['scaler'].upper()}, Encoder: {best_prep['encoder'].upper()}")

    print("=" * 80)

    return prep_performance
