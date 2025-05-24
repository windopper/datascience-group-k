from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from optuna_optimization import perform_optuna_optimization
from grid_search import perform_grid_search, get_grid_search_results, print_grid_search_summary
from evaluate import evaluate_model_comprehensive


def get_model_parameter_combinations():
    """
    Define various parameter combinations for each model.
    """
    model_params = {
        "Naive Bayes": {
            "model": GaussianNB,
            "params": {},
            "default_params": {}
        },
        "K-Nearest Neighbors": {
            "model": KNeighborsClassifier,
            "params": {"n_neighbors": 5, "metric": "minkowski", "p": 2},
            "default_params": {}
        },
        "Logistic Regression": {
            "model": LogisticRegression,
            "params": {"max_iter": 2000, "solver": "lbfgs", "C": 1.0},
            "default_params": {}
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier,
            "params": {"max_depth": None, "min_samples_split": 2},
            "default_params": {}
        },
        "Random Forest": {
            "model": RandomForestClassifier,
            "params": {"n_estimators": 100, "max_depth": None},
            "default_params": {}
        },
    }
    return model_params


def train(train_X, train_y, test_X, test_y, model=None, seed=42, use_optuna=False, use_grid_search=False, n_trials=10, reduced_grid=False, optimization_data_ratio=1.0):
    """
    Train with various models and parameter combinations and evaluate comprehensively.

    Args:
        train_X: Training features
        train_y: Training labels
        test_X: Test features
        test_y: Test labels
        model: Specific model to train (None for all models)
        seed: Random seed
        use_optuna: Whether to use Optuna optimization
        use_grid_search: Whether to use Grid Search optimization
        n_trials: Number of trials for Optuna
        reduced_grid: Whether to use reduced grid for Grid Search
        optimization_data_ratio: Ratio of training data to use for optimization (0.0 < ratio <= 1.0)
    """
    np.random.seed(seed)

    model_params = get_model_parameter_combinations()

    # Run only specific models
    if model is not None and model != "all":
        if model not in model_params:
            raise ValueError(
                f"Invalid model: {model}, please choose from {list(model_params.keys())}")
        model_params = {model: model_params[model]}

    all_results = []

    print("=" * 80)
    if use_optuna:
        print(
            f"Model training and evaluation start (Optuna - trials: {n_trials}, optimization data ratio: {optimization_data_ratio:.2f})")
    elif use_grid_search:
        grid_type = "reduced" if reduced_grid else "full"
        print(
            f"Model training and evaluation start (GridSearchCV - {grid_type} grid, optimization data ratio: {optimization_data_ratio:.2f})")
    else:
        print("Model training and evaluation start")
    print("=" * 80)

    # Prepare optimization data if needed
    if (use_optuna or use_grid_search) and optimization_data_ratio < 1.0:
        print(
            f"\nSampling {optimization_data_ratio:.2f} of training data for optimization...")
        opt_train_X, _, opt_train_y, _ = train_test_split(
            train_X, train_y,
            train_size=optimization_data_ratio,
            random_state=seed,
            stratify=train_y
        )
        print(f"Original training data size: {len(train_X)}")
        print(f"Optimization data size: {len(opt_train_X)}")
    else:
        opt_train_X = train_X
        opt_train_y = train_y

    # Train each model
    for model_name, model_config in model_params.items():
        print(f"\n[{model_name}] Model training...")
        model_class = model_config["model"]
        model_default_params = model_config["default_params"]

        if use_optuna:
            # Optuna optimization
            data_info = f" (using {len(opt_train_X)}/{len(train_X)} samples)" if optimization_data_ratio < 1.0 else ""
            print(
                f"  Optuna is searching for optimal parameters{data_info}... ({n_trials} trials)")

            try:
                study = perform_optuna_optimization(
                    model_class, opt_train_X, opt_train_y,
                    n_trials=n_trials, cv=5, random_state=seed, **model_default_params
                )

                best_params = study.best_params
                best_score = study.best_value

                print(f"  Optimal parameters: {best_params}")
                print(f"  Optuna best CV F1-Score: {best_score:.4f}")

                # Create and train the model with the optimal parameters using FULL training data
                if model_name in ["Decision Tree", "Random Forest", "Logistic Regression"]:
                    if 'random_state' not in best_params:
                        best_params['random_state'] = seed

                best_model = model_class(**best_params)
                print(
                    f"  Training final model with optimal parameters on full training data...")
                best_model.fit(train_X, train_y)

                # Evaluate the optimal model
                result = evaluate_model_comprehensive(
                    best_model, test_X, test_y, train_X, train_y,
                    f"{model_name}_Optuna", best_params
                )
                result["optuna_cv_score"] = best_score
                result["optuna_n_trials"] = n_trials
                result["optimization_data_ratio"] = optimization_data_ratio
                all_results.append(result)

                # Print results
                print(f"  Test Set - Accuracy: {result['accuracy']:.4f}, "
                      f"Precision: {result['precision']:.4f}, "
                      f"Recall: {result['recall']:.4f}, "
                      f"F1: {result['f1_score']:.4f}")

                if result['roc_auc'] is not None:
                    print(f"  ROC-AUC: {result['roc_auc']:.4f}")

            except Exception as e:
                print(f"  Error occurred during Optuna optimization: {e}")
                print("  Proceeding with default parameters...")
                # Fallback to default parameters
                default_params = model_config["params"].copy()
                if model_name in ["Decision Tree", "Random Forest", "Logistic Regression"]:
                    default_params["random_state"] = seed

                clf = model_class(**default_params)
                clf.fit(train_X, train_y)

                result = evaluate_model_comprehensive(
                    clf, test_X, test_y, train_X, train_y,
                    f"{model_name}_Default", default_params
                )
                all_results.append(result)

        elif use_grid_search:
            # Grid Search optimization
            grid_type = "reduced" if reduced_grid else "full"
            data_info = f" (using {len(opt_train_X)}/{len(train_X)} samples)" if optimization_data_ratio < 1.0 else ""
            print(
                f"  GridSearchCV is searching for optimal parameters{data_info}... ({grid_type} grid)")

            try:
                grid_search = perform_grid_search(
                    model_class, opt_train_X, opt_train_y,
                    cv=5, scoring='f1', random_state=seed,
                    reduced=reduced_grid, **model_default_params
                )

                # Get results and print summary
                grid_results = get_grid_search_results(grid_search, model_name)
                print_grid_search_summary(grid_results)

                best_params = grid_results['best_params']
                best_score = grid_results['best_score']
                best_estimator = grid_results['best_estimator']

                # Train the best model on FULL training data
                print(
                    f"  Training final model with optimal parameters on full training data...")
                best_estimator.fit(train_X, train_y)

                # Evaluate the optimal model
                result = evaluate_model_comprehensive(
                    best_estimator, test_X, test_y, train_X, train_y,
                    f"{model_name}_GridSearch", best_params
                )
                result["grid_search_cv_score"] = best_score
                result["grid_search_type"] = grid_type
                result["optimization_data_ratio"] = optimization_data_ratio
                all_results.append(result)

                # Print results
                print(f"  Test Set - Accuracy: {result['accuracy']:.4f}, "
                      f"Precision: {result['precision']:.4f}, "
                      f"Recall: {result['recall']:.4f}, "
                      f"F1: {result['f1_score']:.4f}")

                if result['roc_auc'] is not None:
                    print(f"  ROC-AUC: {result['roc_auc']:.4f}")

            except Exception as e:
                print(
                    f"  Error occurred during GridSearchCV optimization: {e}")
                print("  Proceeding with default parameters...")
                # Fallback to default parameters
                default_params = model_config["params"].copy()
                if model_name in ["Decision Tree", "Random Forest", "Logistic Regression"]:
                    default_params["random_state"] = seed

                clf = model_class(**default_params)
                clf.fit(train_X, train_y)

                result = evaluate_model_comprehensive(
                    clf, test_X, test_y, train_X, train_y,
                    f"{model_name}_Default", default_params
                )
                all_results.append(result)

        else:
            # Existing method: predefined parameter combinations
            params = model_config["params"]
            default_params = model_config["default_params"]
            model_results = []

            print(f"  Parameter combination: {params}")

            # Models that require a random seed
            if model_name in ["Decision Tree", "Random Forest", "Logistic Regression"]:
                params = params.copy()  # Keep original parameters
                params["random_state"] = seed

            # Create and train the model
            clf = model_class(**params, **default_params)
            clf.fit(train_X, train_y)

            # Comprehensive evaluation
            result = evaluate_model_comprehensive(
                clf, test_X, test_y, train_X, train_y,
                f"{model_name}", params
            )

            model_results.append(result)
            all_results.append(result)

            # Print results
            print(f"    Accuracy: {result['accuracy']:.4f}, "
                  f"Precision: {result['precision']:.4f}, "
                  f"Recall: {result['recall']:.4f}, "
                  f"F1: {result['f1_score']:.4f}")

            if result['roc_auc'] is not None:
                print(f"    ROC-AUC: {result['roc_auc']:.4f}")

            # Display the best performance of the model
            if model_results:
                best_model = max(model_results, key=lambda x: x['f1_score'])
                print(f"\n  [{model_name}] Highest F1-Score: {best_model['f1_score']:.4f} "
                      f"(parameters: {best_model['parameters']})")

    return all_results
