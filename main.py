import argparse
import numpy as np
import json
from datetime import datetime
from preprocess import preprocessing
from train import train
from evaluate import print_evaluation_summary, compare_preprocessing_results


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

    print(f"Results saved to {filename}")
    return filename


def run_preprocessing_comparison(args):
    """
    Run experiments with different preprocessing combinations to compare performance.
    """
    from preprocess import scaler_types, encoder_types

    # Define preprocessing combinations to test, avoiding duplicates
    # since default scaler = standard and default encoder = onehot
    # Exclude 'default' since it's same as 'standard'
    unique_scalers = ['standard', 'minmax']
    # Exclude 'default' since it's same as 'onehot'
    unique_encoders = ['onehot', 'label']

    preprocessing_combinations = []
    for scaler in unique_scalers:
        for encoder in unique_encoders:
            preprocessing_combinations.append((scaler, encoder))

    print("=" * 80)
    print("PREPROCESSING TECHNIQUE COMPARISON EXPERIMENT")
    print("=" * 80)
    print("Note: 'default' scaler is equivalent to 'standard', 'default' encoder is equivalent to 'onehot'")
    print(
        f"Testing {len(preprocessing_combinations)} unique preprocessing combinations:")
    for i, (scaler, encoder) in enumerate(preprocessing_combinations, 1):
        print(f"  {i}. Scaler: {scaler.upper()}, Encoder: {encoder.upper()}")
    print("=" * 80)

    all_preprocessing_results = []

    for i, (scaler, encoder) in enumerate(preprocessing_combinations, 1):
        print(
            f"\n[{i}/{len(preprocessing_combinations)}] Testing: Scaler={scaler.upper()}, Encoder={encoder.upper()}")
        print("-" * 60)

        # Preprocessing information
        preprocessing_info = {
            "scaler": scaler,
            "encoder": encoder,
            "random_seed": args.seed
        }

        try:
            # Run preprocessing
            print("Data preprocessing...")
            train_X, train_y, test_X, test_y = preprocessing(
                scaler=scaler, encoder=encoder)

            # Update preprocessing info
            preprocessing_info.update({
                "train_shape": train_X.shape,
                "test_shape": test_X.shape,
                "features": train_X.shape[1]
            })

            print("Data preprocessing complete.")
            print("Model training start...")

            # Run training
            results = train(train_X, train_y, test_X, test_y,
                            model=args.model, seed=args.seed,
                            use_optuna=args.optuna, use_grid_search=args.grid_search,
                            n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                            optimization_data_ratio=args.optimization_data_ratio)

            # Store results with preprocessing info
            preprocessing_result = {
                "preprocessing_info": preprocessing_info,
                "model_results": results
            }
            all_preprocessing_results.append(preprocessing_result)

            print(
                f"Completed: Scaler={scaler.upper()}, Encoder={encoder.upper()}")

        except Exception as e:
            print(
                f"Error with Scaler={scaler.upper()}, Encoder={encoder.upper()}: {e}")
            continue

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

            print(f"\nComprehensive comparison results saved to {filename}")

    return all_preprocessing_results


def print_available_commands():
    """
    Print available command line usage examples.
    """
    print("\nAvailable Commands:")
    print("  python main.py                                    # Run the entire pipeline")
    print("  python main.py --model 'Random Forest'           # Run a specific model")
    print("  python main.py --save-results                    # Save the results to a file")
    print("  python main.py --model 'Random Forest' --save-results      # Run the Random Forest model and save the results")
    print("")
    print("  # Hyperparameter Optimization:")
    print("  python main.py --optuna                          # Use Optuna (auto uses 20% data for optimization)")
    print("  python main.py --optuna --n-trials 100           # Use Optuna with 100 trials (auto uses 20% data)")
    print("  python main.py --grid-search                     # Use GridSearchCV (auto uses 20% data for optimization)")
    print("  python main.py --grid-search --reduced-grid      # Use GridSearchCV with reduced parameter grid")
    print("  python main.py --model 'Random Forest' --optuna  # Run specific model with Optuna")
    print("  python main.py --model 'Random Forest' --grid-search  # Run specific model with GridSearchCV")
    print("")
    print("  # Optimization Data Ratio (for efficiency):")
    print("  python main.py --optuna --optimization-data-ratio 0.1    # Use only 10% of data for Optuna optimization")
    print("  python main.py --grid-search --optimization-data-ratio 0.5  # Use 50% of data for GridSearchCV")
    print("  python main.py --optuna --optimization-data-ratio 1.0     # Use all data for optimization (slower)")
    print("  # Note: Optuna/GridSearch automatically use 20% of data by default for faster optimization")
    print("  python main.py --optuna --n-trials 50 --optimization-data-ratio 0.2  # Combine with other options")
    print("")
    print("  # Data Preprocessing:")
    print("  python main.py --scaler 'standard' --encoder 'onehot' # Set the scaler and encoder")
    print("  python main.py --compare-preprocessing            # Compare 4 preprocessing combinations (standard/minmax × onehot/label)")
    print("  python main.py --compare-preprocessing --model 'Random Forest' # Compare preprocessing with specific model")
    print("  python main.py --compare-preprocessing --save-results # Compare preprocessing and save comprehensive results")
    print("")
    print("  # Help:")
    print("  python main.py --help-commands                   # Show this help message")


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
            print(
                f"Info: Using optimization_data_ratio=0.2 (default for Optuna/GridSearch)")

    # Validation: Optuna and GridSearch cannot be used simultaneously
    if args.optuna and args.grid_search:
        print("Error: --optuna and --grid-search cannot be used simultaneously.")
        print(
            "Please choose either Optuna or GridSearchCV for hyperparameter optimization.")
        return

    # Validation: reduced-grid requires grid-search
    if args.reduced_grid and not args.grid_search:
        print("Warning: --reduced-grid option requires --grid-search. Ignoring --reduced-grid.")
        args.reduced_grid = False

    # Validation: optimization_data_ratio should be between 0 and 1
    if not 0.0 < args.optimization_data_ratio <= 1.0:
        print(
            f"Error: --optimization-data-ratio should be between 0.0 and 1.0, got {args.optimization_data_ratio}")
        return

    # Warning: optimization_data_ratio only affects when using optimization
    if args.optimization_data_ratio < 1.0 and not (args.optuna or args.grid_search):
        print(
            f"Warning: --optimization-data-ratio={args.optimization_data_ratio} specified but no optimization method selected.")
        print("This parameter only affects Optuna or GridSearch optimization. It will be ignored.")

    np.random.seed(args.seed)
    print(f"Numpy random seed: {args.seed}")

    # Preprocessing comparison mode
    if args.compare_preprocessing:
        print("Starting preprocessing technique comparison...")
        run_preprocessing_comparison(args)
        return

    # Store preprocessing information
    preprocessing_info = {
        "scaler": args.scaler,
        "encoder": args.encoder,
        "random_seed": args.seed
    }

    if args.preprocess and not args.train:
        print("Data preprocessing...")
        preprocessing(scaler=args.scaler, encoder=args.encoder)
        print("Data preprocessing complete.")
    elif args.train and not args.preprocess:
        print("Model training...")
        print("Warning: If you use only the --train option, the preprocessed data must be prepared.")
        print("Data preprocessing...")
        train_X, train_y, test_X, test_y = preprocessing(
            scaler=args.scaler, encoder=args.encoder)

        # Update preprocessing information
        preprocessing_info.update({
            "train_shape": train_X.shape,
            "test_shape": test_X.shape,
            "features": train_X.shape[1]
        })

        print("Data preprocessing complete.")
        print("Model training start...")
        results = train(train_X, train_y, test_X, test_y,
                        model=args.model, seed=args.seed,
                        use_optuna=args.optuna, use_grid_search=args.grid_search,
                        n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                        optimization_data_ratio=args.optimization_data_ratio)

        # Print evaluation results
        print_evaluation_summary(results, preprocessing_info)

        if args.save_results:
            save_results_to_file(results, args.output_file, preprocessing_info)

        print("Model training complete.")
    else:
        print("Data preprocessing and model training start...")
        print("Data preprocessing...")
        train_X, train_y, test_X, test_y = preprocessing(
            scaler=args.scaler, encoder=args.encoder)

        # Update preprocessing information
        preprocessing_info.update({
            "train_shape": train_X.shape,
            "test_shape": test_X.shape,
            "features": train_X.shape[1]
        })

        print("Data preprocessing complete.")
        print("Model training start...")
        results = train(train_X, train_y, test_X, test_y,
                        model=args.model, seed=args.seed,
                        use_optuna=args.optuna, use_grid_search=args.grid_search,
                        n_trials=args.n_trials, reduced_grid=args.reduced_grid,
                        optimization_data_ratio=args.optimization_data_ratio)

        # Print evaluation results
        print_evaluation_summary(results, preprocessing_info)

        if args.save_results:
            save_results_to_file(results, args.output_file, preprocessing_info)

        print("Data preprocessing and model training complete.")


if __name__ == "__main__":
    main()
