from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, confusion_matrix, classification_report
)
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import print as rprint

console = Console()


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
    console.print()
    panel = Panel.fit(
        "[bold white]Overall Results Summary (Top 5 F1-Score)[/bold white]",
        style="bold blue"
    )
    console.print(panel)

    # Sort by F1-Score
    sorted_results = sorted(
        all_results, key=lambda x: x['f1_score'], reverse=True)

    # Create table for top 5 results
    results_table = Table(show_header=True, header_style="bold magenta")
    results_table.add_column("Rank", style="cyan", width=4)
    results_table.add_column("Model", style="yellow")
    results_table.add_column("Parameters", style="white")
    results_table.add_column("Accuracy", style="green", justify="right")
    results_table.add_column("Precision", style="green", justify="right")
    results_table.add_column("Recall", style="green", justify="right")
    results_table.add_column("F1", style="bold green", justify="right")
    results_table.add_column("ROC-AUC", style="green", justify="right")
    results_table.add_column("Additional Info", style="dim")

    for i, result in enumerate(sorted_results[:5], 1):
        # Format parameters
        params_str = str(result['parameters'])
        if len(params_str) > 50:
            params_str = params_str[:47] + "..."

        # ROC-AUC handling
        roc_auc_str = f"{result['roc_auc']:.4f}" if result['roc_auc'] is not None else "N/A"

        # Additional info
        additional_info = ""
        if 'optuna_cv_score' in result:
            additional_info = f"Optuna CV: {result['optuna_cv_score']:.4f} ({result['optuna_n_trials']} trials)"
        elif 'grid_search_cv_score' in result:
            additional_info = f"GridSearch CV: {result['grid_search_cv_score']:.4f} ({result['grid_search_type']} grid)"

        results_table.add_row(
            str(i),
            result['model_name'],
            params_str,
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1_score']:.4f}",
            roc_auc_str,
            additional_info
        )

    console.print(results_table)

    # Evaluation method-based highest performance model
    console.print()
    panel = Panel.fit(
        "[bold white]Best Performance by Metric[/bold white]",
        style="bold blue"
    )
    console.print(panel)

    metrics = [
        ('accuracy', 'ACCURACY'),
        ('precision', 'PRECISION'),
        ('recall', 'RECALL'),
        ('f1_score', 'F1-SCORE')
    ]

    if any(r['roc_auc'] is not None for r in all_results):
        metrics.append(('roc_auc', 'ROC-AUC'))

    # Create table for best metrics
    metrics_table = Table(show_header=True, header_style="bold magenta")
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Score", style="bold green", justify="right")
    metrics_table.add_column("Model", style="yellow")

    for metric, display_name in metrics:
        if metric == 'roc_auc':
            valid_results = [r for r in all_results if r[metric] is not None]
            if not valid_results:
                continue
            best = max(valid_results, key=lambda x: x[metric])
        else:
            best = max(all_results, key=lambda x: x[metric])

        metrics_table.add_row(
            display_name,
            f"{best[metric]:.4f}",
            best['model_name']
        )

    console.print(metrics_table)

    # Print preprocessing information if provided
    if preprocessing_info:
        console.print()
        panel = Panel.fit(
            "[bold white]Preprocessing Information[/bold white]",
            style="bold blue"
        )
        console.print(panel)

        prep_table = Table(show_header=False, box=None)
        prep_table.add_column("Property", style="cyan")
        prep_table.add_column("Value", style="white")

        prep_table.add_row("Scaler", preprocessing_info.get('scaler', 'N/A'))
        prep_table.add_row("Encoder", preprocessing_info.get('encoder', 'N/A'))
        if 'train_shape' in preprocessing_info:
            prep_table.add_row("Train Data Shape", str(
                preprocessing_info['train_shape']))
        if 'test_shape' in preprocessing_info:
            prep_table.add_row("Test Data Shape", str(
                preprocessing_info['test_shape']))
        if 'features' in preprocessing_info:
            prep_table.add_row("Number of Features", str(
                preprocessing_info['features']))

        console.print(prep_table)


def compare_preprocessing_results(preprocessing_results):
    """
    Compare results from different preprocessing techniques.
    """
    console.print()
    panel = Panel.fit(
        "[bold white]PREPROCESSING TECHNIQUE COMPARISON[/bold white]",
        style="bold blue"
    )
    console.print(panel)

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

    console.print(
        "\n[bold yellow]Performance Ranking by Best F1-Score:[/bold yellow]")

    # Create table for preprocessing performance
    prep_table = Table(show_header=True, header_style="bold magenta")
    prep_table.add_column("Rank", style="cyan", width=4)
    prep_table.add_column("Scaler", style="yellow")
    prep_table.add_column("Encoder", style="yellow")
    prep_table.add_column("Best F1", style="bold green", justify="right")
    prep_table.add_column("Best Model", style="white")
    prep_table.add_column("Best Acc", style="green", justify="right")
    prep_table.add_column("Best Prec", style="green", justify="right")
    prep_table.add_column("Best Rec", style="green", justify="right")
    prep_table.add_column("Best ROC", style="green", justify="right")
    prep_table.add_column("Avg F1", style="dim", justify="right")
    prep_table.add_column("Models", style="dim", justify="right")

    for i, prep in enumerate(prep_performance, 1):
        roc_auc_str = f"{prep['best_roc_auc']:.4f}" if prep['best_roc_auc'] is not None else "N/A"

        prep_table.add_row(
            str(i),
            prep['scaler'].upper(),
            prep['encoder'].upper(),
            f"{prep['best_f1']:.4f}",
            prep['best_model'],
            f"{prep['best_accuracy']:.4f}",
            f"{prep['best_precision']:.4f}",
            f"{prep['best_recall']:.4f}",
            roc_auc_str,
            f"{prep['avg_f1']:.4f}",
            str(prep['num_models'])
        )

    console.print(prep_table)

    # Find best preprocessing technique for each metric
    console.print(
        "\n[bold yellow]Best Preprocessing Technique by Metric:[/bold yellow]")

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

    # Create table for best preprocessing by metric
    best_prep_table = Table(show_header=True, header_style="bold magenta")
    best_prep_table.add_column("Metric", style="cyan")
    best_prep_table.add_column("Score", style="bold green", justify="right")
    best_prep_table.add_column("Scaler", style="yellow")
    best_prep_table.add_column("Encoder", style="yellow")

    for metric_key, metric_name in metrics_to_check:
        if metric_key.startswith('avg_roc') or metric_key.startswith('best_roc'):
            valid_preps = [
                p for p in prep_performance if p[metric_key] is not None]
            if not valid_preps:
                continue
            best_prep = max(valid_preps, key=lambda x: x[metric_key])
        else:
            best_prep = max(prep_performance, key=lambda x: x[metric_key])

        best_prep_table.add_row(
            metric_name,
            f"{best_prep[metric_key]:.4f}",
            best_prep['scaler'].upper(),
            best_prep['encoder'].upper()
        )

    console.print(best_prep_table)

    return prep_performance
