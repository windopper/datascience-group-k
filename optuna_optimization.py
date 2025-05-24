from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
import optuna
from optuna.samplers import TPESampler


def create_optuna_objective(model_class, train_X, train_y, cv=5, random_state=42, **model_default_params):
    """
    Create an objective function for Optuna.
    """
    def objective(trial):
        # Model-specific parameter suggestions
        if model_class == GaussianNB:
            params = {
                'var_smoothing': trial.suggest_float('var_smoothing', 1e-10, 1e-6, log=True)
            }
        elif model_class == KNeighborsClassifier:
            params = {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 15),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
            }
            if params['metric'] == 'minkowski':
                params['p'] = trial.suggest_int('p', 1, 3)
        elif model_class == LogisticRegression:
            params = {
                'C': trial.suggest_float('C', 0.001, 100.0, log=True),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'newton-cg']),
                'max_iter': trial.suggest_int('max_iter', 1000, 3000),
                'random_state': random_state
            }
        elif model_class == DecisionTreeClassifier:
            max_depth = trial.suggest_int('max_depth', 3, 20)
            params = {
                'max_depth': None if max_depth == 20 else max_depth,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'random_state': random_state
            }
        elif model_class == RandomForestClassifier:
            max_depth = trial.suggest_int('max_depth', 3, 20)
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': None if max_depth == 20 else max_depth,
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': random_state
            }
        else:
            raise ValueError(f"Unsupported model class: {model_class}")

        # Create and cross-validate the model
        model = model_class(**params, **model_default_params)
        cv_scores = cross_val_score(
            model, train_X, train_y, cv=cv, scoring='f1')

        return cv_scores.mean()

    return objective


def perform_optuna_optimization(model_class, train_X, train_y, n_trials=10, cv=5, random_state=42, **model_default_params):
    """
    Use Optuna to find the optimal parameters.
    """
    # Create the Optuna study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=random_state)
    )

    # Create the objective function
    objective = create_optuna_objective(
        model_class, train_X, train_y, cv, random_state, **model_default_params)

    # Run the optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return study
