from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def get_parameter_grids():
    """
    Define parameter grids for each model for GridSearchCV.
    """
    parameter_grids = {
        GaussianNB: {
            'var_smoothing': np.logspace(-10, -6, 10)
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
            'metric': ['euclidean', 'manhattan', 'minkowski'],
            'p': [1, 2, 3]
        },
        LogisticRegression: {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'solver': ['lbfgs', 'liblinear', 'newton-cg'],
            'max_iter': [1000, 2000, 3000]
        },
        DecisionTreeClassifier: {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 2, 5, 10]
        },
        RandomForestClassifier: {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 15, 20],
            'max_features': ['sqrt', 'log2', None]
        }
    }
    return parameter_grids


def get_reduced_parameter_grids():
    """
    Define reduced parameter grids for faster computation.
    """
    reduced_grids = {
        GaussianNB: {
            'var_smoothing': np.logspace(-9, -7, 5)
        },
        KNeighborsClassifier: {
            'n_neighbors': [3, 5, 7, 11],
            'metric': ['euclidean', 'manhattan'],
            'p': [1, 2]
        },
        LogisticRegression: {
            'C': [0.01, 0.1, 1.0, 10.0],
            'solver': ['lbfgs', 'liblinear'],
            'max_iter': [1000, 2000]
        },
        DecisionTreeClassifier: {
            'max_depth': [3, 5, 10, None],
            'min_samples_split': [2, 5, 10],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 2, 5]
        },
        RandomForestClassifier: {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    }
    return reduced_grids


def perform_grid_search(model_class, train_X, train_y, cv=5, scoring='f1',
                        random_state=42, reduced=False, n_jobs=-1, **model_default_params):
    """
    Perform GridSearchCV to find optimal parameters.

    Parameters:
    -----------
    model_class : sklearn model class
        The model class to optimize
    train_X : array-like
        Training features
    train_y : array-like
        Training target
    cv : int, default=5
        Number of cross-validation folds
    scoring : str, default='f1'
        Scoring metric for optimization
    random_state : int, default=42
        Random state for reproducibility
    reduced : bool, default=False
        Whether to use reduced parameter grid for faster computation
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all processors)
    **model_default_params : dict
        Additional default parameters for the model

    Returns:
    --------
    GridSearchCV object with results
    """

    # Get parameter grids
    if reduced:
        param_grids = get_reduced_parameter_grids()
    else:
        param_grids = get_parameter_grids()

    if model_class not in param_grids:
        raise ValueError(f"Unsupported model class: {model_class}")

    param_grid = param_grids[model_class]

    # Handle random_state for models that support it
    if model_class in [DecisionTreeClassifier, RandomForestClassifier, LogisticRegression]:
        # Create model instance with random_state
        model = model_class(random_state=random_state, **model_default_params)
    else:
        # Create model instance without random_state
        model = model_class(**model_default_params)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True
    )

    # Fit the grid search
    grid_search.fit(train_X, train_y)

    return grid_search


def get_grid_search_results(grid_search, model_name):
    """
    Extract and format results from GridSearchCV.

    Parameters:
    -----------
    grid_search : GridSearchCV object
        Fitted GridSearchCV object
    model_name : str
        Name of the model

    Returns:
    --------
    dict : Dictionary containing grid search results
    """

    results = {
        'model_name': f"{model_name}_GridSearch",
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_,
        'n_splits': grid_search.n_splits_,
        'scoring': grid_search.scoring
    }

    return results


def print_grid_search_summary(grid_search_results):
    """
    Print a summary of grid search results.

    Parameters:
    -----------
    grid_search_results : dict
        Results dictionary from get_grid_search_results()
    """

    # Create panel for grid search results
    panel = Panel.fit(
        f"[bold white][{grid_search_results['model_name']}] Grid Search Results[/bold white]",
        style="bold blue"
    )
    console.print(panel)

    # Basic results
    console.print(
        f"  [bold]Best CV Score ({grid_search_results['scoring']}):[/bold] [bold green]{grid_search_results['best_score']:.4f}[/bold green]")
    console.print(
        f"  [bold]Best Parameters:[/bold] [yellow]{grid_search_results['best_params']}[/yellow]")
    console.print(
        f"  [bold]Number of CV folds:[/bold] {grid_search_results['n_splits']}")

    # Show top 5 parameter combinations
    cv_results = grid_search_results['cv_results']

    # Get indices sorted by mean test score
    sorted_indices = np.argsort(cv_results['mean_test_score'])[::-1]

    console.print(
        f"\n  [bold yellow]Top 5 parameter combinations:[/bold yellow]")

    # Create table for top combinations
    top_table = Table(show_header=True,
                      header_style="bold magenta", box=None, padding=(0, 1))
    top_table.add_column("Rank", style="cyan", width=4)
    top_table.add_column("Score", style="bold green", justify="right")
    top_table.add_column("Std", style="dim", justify="right")
    top_table.add_column("Parameters", style="white")

    for i, idx in enumerate(sorted_indices[:5]):
        score = cv_results['mean_test_score'][idx]
        std = cv_results['std_test_score'][idx]
        params = cv_results['params'][idx]

        # Format parameters string
        params_str = str(params)
        if len(params_str) > 60:
            params_str = params_str[:57] + "..."

        top_table.add_row(
            str(i+1),
            f"{score:.4f}",
            f"±{std:.4f}",
            params_str
        )

    console.print(top_table)
