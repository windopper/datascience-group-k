import kagglehub
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
import warnings
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

warnings.simplefilter(action='ignore', category=FutureWarning)

scaler_types = ['standard', 'minmax']
encoder_types = ['onehot', 'label']


def preprocessing(scaler='standard', encoder='onehot'):
    """
    Preprocess the data by removing unnecessary columns and renaming them.
    """
    if scaler not in scaler_types:
        raise ValueError(
            f"Invalid scaler: {scaler}, please choose from {scaler_types}")
    if encoder not in encoder_types:
        raise ValueError(
            f"Invalid encoder: {encoder}, please choose from {encoder_types}")

    path = kagglehub.dataset_download(
        "teejmahal20/airline-passenger-satisfaction")
    train_df = pd.read_csv(path + "/train.csv")
    test_df = pd.read_csv(path + "/test.csv")

    # Drop unnecessary columns
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Drop duplicate columns
    train_df = train_df.drop_duplicates()
    test_df = test_df.drop_duplicates()

    # log transformation
    # log
    log_cols = ['Arrival Delay in Minutes', 'Departure Delay in Minutes']
    train_df[log_cols] = np.log1p(train_df[log_cols])
    test_df[log_cols] = np.log1p(test_df[log_cols])

    # scaling
    scaling_cols = ['Age', 'Flight Distance',
                    'Departure Delay in Minutes', 'Arrival Delay in Minutes']

    if scaler == 'standard':
        standard_scaler = StandardScaler()
        train_df[scaling_cols] = standard_scaler.fit_transform(
            train_df[scaling_cols])
        test_df[scaling_cols] = standard_scaler.transform(
            test_df[scaling_cols])
    elif scaler == 'minmax':
        minmax_scaler = MinMaxScaler()
        train_df[scaling_cols] = minmax_scaler.fit_transform(
            train_df[scaling_cols])
        test_df[scaling_cols] = minmax_scaler.transform(test_df[scaling_cols])

    unused_cols = ['Unnamed: 0', 'id']
    target_col = 'satisfaction'
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    # encoding
    if encoder == 'onehot':
        train_df = pd.get_dummies(train_df, columns=categorical_cols)
        test_df = pd.get_dummies(test_df, columns=categorical_cols)
    elif encoder == 'label':
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            train_df[col] = label_encoder.fit_transform(train_df[col])
            test_df[col] = label_encoder.transform(test_df[col])

    train_df[target_col] = train_df[target_col].replace(
        {'neutral or dissatisfied': 0, 'satisfied': 1})
    test_df[target_col] = test_df[target_col].replace(
        {'neutral or dissatisfied': 0, 'satisfied': 1})

    train_df = train_df.astype(
        {col: 'int' for col in train_df.columns if train_df[col].dtype == 'bool'})
    test_df = test_df.astype(
        {col: 'int' for col in test_df.columns if test_df[col].dtype == 'bool'})

    train_X = train_df.drop(unused_cols + [target_col], axis=1)
    train_y = train_df[target_col]
    test_X = test_df.drop(unused_cols + [target_col], axis=1)
    test_y = test_df[target_col]

    return train_X, train_y, test_X, test_y


def run_preprocessing_comparison(args):
    """
    Run experiments with different preprocessing combinations to compare performance.
    """
    from train import train
    from evaluate import compare_preprocessing_results

    preprocessing_combinations = []
    for scaler in scaler_types:
        for encoder in encoder_types:
            preprocessing_combinations.append((scaler, encoder))

    # Create header panel
    panel = Panel.fit(
        "[bold white]PREPROCESSING TECHNIQUE COMPARISON EXPERIMENT[/bold white]",
        style="bold blue"
    )
    console.print(panel)

    console.print(
        f"[bold]Testing {len(preprocessing_combinations)} unique preprocessing combinations:[/bold]")

    # Create table for combinations
    combo_table = Table(show_header=True, header_style="bold magenta")
    combo_table.add_column("No.", style="cyan", width=4)
    combo_table.add_column("Scaler", style="yellow")
    combo_table.add_column("Encoder", style="yellow")

    for i, (scaler, encoder) in enumerate(preprocessing_combinations, 1):
        combo_table.add_row(str(i), scaler.upper(), encoder.upper())

    console.print(combo_table)

    all_preprocessing_results = []

    # Use progress bar for preprocessing combinations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task(
            "Processing combinations...", total=len(preprocessing_combinations))

        for i, (scaler, encoder) in enumerate(preprocessing_combinations, 1):
            # Update progress description with current status
            progress.update(
                task, description=f"[{i}/{len(preprocessing_combinations)}] Testing: Scaler={scaler.upper()}, Encoder={encoder.upper()}")

            # Preprocessing information
            preprocessing_info = {
                "scaler": scaler,
                "encoder": encoder,
                "random_seed": args.seed
            }

            try:
                # Update progress description for preprocessing step
                progress.update(
                    task, description=f"[{i}/{len(preprocessing_combinations)}] Preprocessing data: Scaler={scaler.upper()}, Encoder={encoder.upper()}")

                # Run preprocessing
                train_X, train_y, test_X, test_y = preprocessing(
                    scaler=scaler, encoder=encoder)

                # Update preprocessing info
                preprocessing_info.update({
                    "train_shape": train_X.shape,
                    "test_shape": test_X.shape,
                    "features": train_X.shape[1]
                })

                # Update progress description for training step
                progress.update(
                    task, description=f"[{i}/{len(preprocessing_combinations)}] Training model: Scaler={scaler.upper()}, Encoder={encoder.upper()}")

                # Run training (with quiet mode to avoid interfering with progress bar)
                results = train(train_X, train_y, test_X, test_y,
                                model=args.model, seed=args.seed,
                                use_optuna=args.optuna, use_grid_search=args.grid_search,
                                n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                                optimization_data_ratio=args.optimization_data_ratio,
                                quiet=True)

                preprocessing_result = {
                    "preprocessing_info": preprocessing_info,
                    "model_results": results
                }
                all_preprocessing_results.append(preprocessing_result)

                # Update progress description for completion
                progress.update(
                    task, description=f"[{i}/{len(preprocessing_combinations)}] ✓ Completed: Scaler={scaler.upper()}, Encoder={encoder.upper()}")

            except Exception as e:
                # Update progress description for error
                progress.update(
                    task, description=f"[{i}/{len(preprocessing_combinations)}] ✗ Error: Scaler={scaler.upper()}, Encoder={encoder.upper()}")
                continue

            progress.advance(task)

    # Compare all preprocessing results
    if all_preprocessing_results:
        best_combinations = compare_preprocessing_results(
            all_preprocessing_results)

        # Save comprehensive results
        if args.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = args.output_file or f"preprocessing_comparison_{timestamp}.json"

            comprehensive_output = {
                "experiment_type": "preprocessing_comparison",
                "note": "default scaler = standard, default encoder = onehot",
                "combinations_tested": len(preprocessing_combinations),
                "experiment_settings": {
                    "model": args.model,
                    "seed": args.seed,
                    "optuna": args.optuna,
                    "grid_search": args.grid_search,
                    "n_trials": args.n_trials,
                    "reduced_grid": args.reduced_grid
                },
                "preprocessing_results": all_preprocessing_results,
                "performance_comparison": best_combinations,
                "timestamp": datetime.now().isoformat()
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(comprehensive_output, f,
                          indent=2, ensure_ascii=False)

            console.print(
                f"\n[green]✓[/green] Comprehensive comparison results saved to [bold]{filename}[/bold]")

    return all_preprocessing_results
