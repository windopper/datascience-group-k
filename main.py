import argparse
import numpy as np
import json
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint
from preprocess import preprocessing, run_preprocessing_comparison
from train import train
from evaluate import print_evaluation_summary

console = Console()


def save_results_to_file(results, filename=None, preprocessing_info=None):
    """
    Save model training results to a JSON file.
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_results_{timestamp}.json"

    # Convert numpy arrays to lists
    for result in results:
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, np.float64):
                result[key] = float(value)
            elif isinstance(value, np.int64):
                result[key] = int(value)

    # Include preprocessing information
    output_data = {
        "preprocessing_info": preprocessing_info,
        "model_results": results,
        "timestamp": datetime.now().isoformat()
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    console.print(f"[green]✓[/green] Results saved to [bold]{filename}[/bold]")
    return filename


def print_available_commands():
    """
    Print available command line usage examples.
    """
    # Create a table for commands
    table = Table(title="Available Commands", show_header=True,
                  header_style="bold magenta")
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description", style="white")

    # Basic commands
    table.add_row("python main.py", "Run the entire pipeline")
    table.add_row("python main.py --model 'Random Forest'",
                  "Run a specific model")
    table.add_row("python main.py --save-results",
                  "Save the results to a file")
    table.add_row("python main.py --model 'Random Forest' --save-results",
                  "Run the Random Forest model and save the results")

    console.print(table)

    # Hyperparameter Optimization section
    console.print("\n[bold yellow]Hyperparameter Optimization:[/bold yellow]")
    opt_table = Table(show_header=False, box=None, padding=(0, 2))
    opt_table.add_column("Command", style="cyan")
    opt_table.add_column("Description", style="white")

    opt_table.add_row("python main.py --optuna",
                      "Use Optuna (auto uses 20% data for optimization)")
    opt_table.add_row("python main.py --optuna --n-trials 100",
                      "Use Optuna with 100 trials (auto uses 20% data)")
    opt_table.add_row("python main.py --grid-search",
                      "Use GridSearchCV (auto uses 20% data for optimization)")
    opt_table.add_row("python main.py --grid-search --reduced-grid",
                      "Use GridSearchCV with reduced parameter grid")
    opt_table.add_row("python main.py --model 'Random Forest' --optuna",
                      "Run specific model with Optuna")
    opt_table.add_row("python main.py --model 'Random Forest' --grid-search",
                      "Run specific model with GridSearchCV")

    console.print(opt_table)

    # Optimization Data Ratio section
    console.print(
        "\n[bold yellow]Optimization Data Ratio (for efficiency):[/bold yellow]")
    ratio_table = Table(show_header=False, box=None, padding=(0, 2))
    ratio_table.add_column("Command", style="cyan")
    ratio_table.add_column("Description", style="white")

    ratio_table.add_row("python main.py --optuna --optimization-data-ratio 0.1",
                        "Use only 10% of data for Optuna optimization")
    ratio_table.add_row("python main.py --grid-search --optimization-data-ratio 0.5",
                        "Use 50% of data for GridSearchCV")
    ratio_table.add_row("python main.py --optuna --optimization-data-ratio 1.0",
                        "Use all data for optimization (slower)")
    ratio_table.add_row(
        "python main.py --optuna --n-trials 50 --optimization-data-ratio 0.2", "Combine with other options")

    console.print(ratio_table)
    console.print(
        "[dim]# Note: Optuna/GridSearch automatically use 20% of data by default for faster optimization[/dim]")

    # Data Preprocessing section
    console.print("\n[bold yellow]Data Preprocessing:[/bold yellow]")
    prep_table = Table(show_header=False, box=None, padding=(0, 2))
    prep_table.add_column("Command", style="cyan")
    prep_table.add_column("Description", style="white")

    prep_table.add_row(
        "python main.py --scaler 'standard' --encoder 'onehot'", "Set the scaler and encoder")
    prep_table.add_row("python main.py --compare-preprocessing",
                       "Compare 4 preprocessing combinations (standard/minmax × onehot/label)")
    prep_table.add_row("python main.py --compare-preprocessing --model 'Random Forest'",
                       "Compare preprocessing with specific model")
    prep_table.add_row("python main.py --compare-preprocessing --save-results",
                       "Compare preprocessing and save comprehensive results")

    console.print(prep_table)

    # Help section
    console.print("\n[bold yellow]Help:[/bold yellow]")
    help_table = Table(show_header=False, box=None, padding=(0, 2))
    help_table.add_column("Command", style="cyan")
    help_table.add_column("Description", style="white")

    help_table.add_row("python main.py --help-commands",
                       "Show this help message")

    console.print(help_table)


def main():
    parser = argparse.ArgumentParser(
        description="Data Preprocessing and Model Training Pipeline")
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run only the data preprocessing."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run only the model training. (The preprocessed data is required.)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all",
        help="Select the model to train. Available models: 'Naive Bayes', 'K-Nearest Neighbors', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'all' (default: all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Set the numpy random seed. (default: 42)"
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save the model training results to a JSON file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Specify the file name to save the results. (Use with --save-results)"
    )
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Use Optuna to automatically search for the optimal parameters."
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Use GridSearchCV to automatically search for the optimal parameters."
    )
    parser.add_argument(
        "--reduced-grid",
        action="store_true",
        help="Use reduced parameter grid for faster GridSearchCV computation. (Use with --grid-search)"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Set the number of Optuna trials. (default: 10)"
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        help="Select the scaler to use for data preprocessing. (default: standard)"
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="onehot",
        help="Select the encoder to use for data preprocessing. (default: onehot)"
    )
    parser.add_argument(
        "--compare-preprocessing",
        action="store_true",
        help="Compare all available preprocessing combinations to find the best performing one."
    )
    parser.add_argument(
        "--help-commands",
        action="store_true",
        help="Show available command line usage examples."
    )
    parser.add_argument(
        "--optimization-data-ratio",
        type=float,
        default=1.0,
        help="Ratio of training data to use for hyperparameter optimization (0.0 < ratio <= 1.0). Defaults to 0.2 when using Optuna or GridSearch, 1.0 otherwise. Use smaller values for faster optimization on large datasets."
    )

    args = parser.parse_args()

    # Show available commands if requested
    if args.help_commands:
        print_available_commands()
        return

    # Set default optimization_data_ratio based on optimization method
    # If user explicitly set the value, respect their choice
    # If using optimization methods and user didn't set it, use 0.2
    if (args.optuna or args.grid_search) and args.optimization_data_ratio == 1.0:
        # Check if user explicitly set optimization_data_ratio to 1.0 or it's the default
        import sys
        if '--optimization-data-ratio' not in sys.argv:
            args.optimization_data_ratio = 0.2
            console.print(
                f"[blue]Info:[/blue] Using optimization_data_ratio=0.2 (default for Optuna/GridSearch)")

    # Validation: Optuna and GridSearch cannot be used simultaneously
    if args.optuna and args.grid_search:
        console.print(
            "[red]Error:[/red] --optuna and --grid-search cannot be used simultaneously.")
        console.print(
            "Please choose either Optuna or GridSearchCV for hyperparameter optimization.")
        return

    # Validation: reduced-grid requires grid-search
    if args.reduced_grid and not args.grid_search:
        console.print(
            "[yellow]Warning:[/yellow] --reduced-grid option requires --grid-search. Ignoring --reduced-grid.")
        args.reduced_grid = False

    # Validation: optimization_data_ratio should be between 0 and 1
    if not 0.0 < args.optimization_data_ratio <= 1.0:
        console.print(
            f"[red]Error:[/red] --optimization-data-ratio should be between 0.0 and 1.0, got {args.optimization_data_ratio}")
        return

    # Warning: optimization_data_ratio only affects when using optimization
    if args.optimization_data_ratio < 1.0 and not (args.optuna or args.grid_search):
        console.print(
            f"[yellow]Warning:[/yellow] --optimization-data-ratio={args.optimization_data_ratio} specified but no optimization method selected.")
        console.print(
            "This parameter only affects Optuna or GridSearch optimization. It will be ignored.")

    np.random.seed(args.seed)
    console.print(
        f"[blue]Info:[/blue] Numpy random seed: [bold]{args.seed}[/bold]")

    # Preprocessing comparison mode
    if args.compare_preprocessing:
        console.print(
            "[bold blue]Starting preprocessing technique comparison...[/bold blue]")
        run_preprocessing_comparison(args)
        return

    # Store preprocessing information
    preprocessing_info = {
        "scaler": args.scaler,
        "encoder": args.encoder,
        "random_seed": args.seed
    }

    if args.preprocess and not args.train:
        console.print("[bold cyan]Data preprocessing...[/bold cyan]")
        preprocessing(scaler=args.scaler, encoder=args.encoder)
        console.print("[green]✓[/green] Data preprocessing complete.")
    elif args.train and not args.preprocess:
        console.print("[bold cyan]Model training...[/bold cyan]")
        console.print(
            "[yellow]Warning:[/yellow] If you use only the --train option, the preprocessed data must be prepared.")
        console.print("[bold cyan]Data preprocessing...[/bold cyan]")
        train_X, train_y, test_X, test_y = preprocessing(
            scaler=args.scaler, encoder=args.encoder)

        # Update preprocessing information
        preprocessing_info.update({
            "train_shape": train_X.shape,
            "test_shape": test_X.shape,
            "features": train_X.shape[1]
        })

        console.print("[green]✓[/green] Data preprocessing complete.")
        console.print("[bold cyan]Model training start...[/bold cyan]")
        results = train(train_X, train_y, test_X, test_y,
                        model=args.model, seed=args.seed,
                        use_optuna=args.optuna, use_grid_search=args.grid_search,
                        n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                        optimization_data_ratio=args.optimization_data_ratio)

        # Print evaluation results
        print_evaluation_summary(results, preprocessing_info)

        if args.save_results:
            save_results_to_file(results, args.output_file, preprocessing_info)

        console.print("[green]✓[/green] Model training complete.")
    else:
        console.print(
            "[bold cyan]Data preprocessing and model training start...[/bold cyan]")
        console.print("[bold cyan]Data preprocessing...[/bold cyan]")
        train_X, train_y, test_X, test_y = preprocessing(
            scaler=args.scaler, encoder=args.encoder)

        # Update preprocessing information
        preprocessing_info.update({
            "train_shape": train_X.shape,
            "test_shape": test_X.shape,
            "features": train_X.shape[1]
        })

        console.print("[green]✓[/green] Data preprocessing complete.")
        console.print("[bold cyan]Model training start...[/bold cyan]")
        results = train(train_X, train_y, test_X, test_y,
                        model=args.model, seed=args.seed,
                        use_optuna=args.optuna, use_grid_search=args.grid_search,
                        n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                        optimization_data_ratio=args.optimization_data_ratio)

        # Print evaluation results
        print_evaluation_summary(results, preprocessing_info)

        if args.save_results:
            save_results_to_file(results, args.output_file, preprocessing_info)

        console.print(
            "[green]✓[/green] Data preprocessing and model training complete.")


if __name__ == "__main__":
    main()
