# DataScience Group K - Machine Learning Model Training and Evaluation Pipeline

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6.1%2B-orange.svg)](https://scikit-learn.org)
[![pandas](https://img.shields.io/badge/pandas-2.2.3%2B-green.svg)](https://pandas.pydata.org)
[![Optuna](https://img.shields.io/badge/optuna-4.3.0%2B-purple.svg)](https://optuna.org)

**DataScience Group K** is a comprehensive model training and evaluation pipeline that utilizes various machine learning algorithms and hyperparameter optimization techniques. It supports various classification algorithms from scikit-learn and Bayesian optimization using Optuna.

## Table of Contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Supported Models](#supported-models)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Data Preprocessing](#data-preprocessing)
- [Evaluation Metrics](#evaluation-metrics)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Developer Guide](#developer-guide)
- [License](#license)

## Key Features

### 🤖 Machine Learning Algorithms
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **K-Nearest Neighbors**: K-nearest neighbors algorithm
- **Logistic Regression**: Logistic regression classifier
- **Decision Tree**: Decision tree classifier
- **Random Forest**: Random forest ensemble

### 🔧 Hyperparameter Optimization
- **Optuna**: Automatic parameter search through Bayesian optimization
- **GridSearchCV**: Traditional grid search optimization
- **Efficient Data Sampling**: Data ratio adjustment for optimization (default: 20%)

### 📊 Data Preprocessing
- **Scaler Options**: StandardScaler, MinMaxScaler
- **Encoder Options**: OneHotEncoder, LabelEncoder
- **Preprocessing Combination Comparison**: Performance comparison of various preprocessing techniques

### 📈 Comprehensive Evaluation
- **Basic Metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced Metrics**: ROC-AUC, Confusion Matrix
- **Cross Validation**: Stable performance evaluation through 5-fold CV
- **Result Storage**: Experiment results saved in JSON format

## Installation

### Requirements
- Python 3.12 or higher
- Package manager: pip or uv

### Installation with pip
```bash
pip install -r requirements.txt
```

### Installation with uv (Recommended)
```bash
uv sync
```

### Development Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Usage
```python
from main import main
import sys

# Train and evaluate all models
sys.argv = ['main.py']
main()
```

### Running from Command Line
```bash
# Run entire pipeline
python main.py

# Train specific model
python main.py --model "Random Forest"

# Optimization using Optuna
python main.py --optuna --n-trials 50
```

## Usage

### Command Line Interface

#### Basic Execution
```bash
# Train and evaluate all models
python main.py

# Execute specific model only
python main.py --model "Random Forest"
python main.py --model "Logistic Regression"

# Save results to JSON file
python main.py --save-results
python main.py --save-results --output-file "my_results.json"
```

#### Hyperparameter Optimization
```bash
# Optuna Bayesian optimization (default: 10 trials, 20% data usage)
python main.py --optuna
python main.py --optuna --n-trials 50

# GridSearchCV optimization
python main.py --grid-search
python main.py --grid-search --reduced-grid

# Apply Optuna to specific model
python main.py --model "Random Forest" --optuna --n-trials 20
```

#### Data Preprocessing Settings
```bash
# Specify scaler and encoder
python main.py --scaler 'minmax' --encoder 'label'

# Preprocessing combination comparison experiment
python main.py --compare-preprocessing
python main.py --compare-preprocessing --save-results
```

#### Efficiency Optimization
```bash
# Adjust data ratio for optimization (default: 0.2)
python main.py --optuna --optimization-data-ratio 0.1
python main.py --grid-search --optimization-data-ratio 0.5
```

### Programming Interface

#### Direct Function Calls
```python
from preprocess import preprocessing
from train import train

# Data preprocessing
train_X, train_y, test_X, test_y = preprocessing(
    scaler='standard', 
    encoder='onehot'
)

# Model training
results = train(
    train_X, train_y, test_X, test_y,
    model="Random Forest",
    use_optuna=True,
    n_trials=20,
    seed=42
)
```

#### Individual Model Training
```python
from sklearn.ensemble import RandomForestClassifier
from evaluate import evaluate_model_comprehensive

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_X, train_y)

# Comprehensive evaluation
result = evaluate_model_comprehensive(
    model, test_X, test_y, train_X, train_y,
    "Random Forest", {"n_estimators": 100}
)
```

## API Reference

### `main.py`

#### `main()`
Main execution function.

```python
def main()
```

#### `save_results_to_file(results, filename=None, preprocessing_info=None)`
Saves experiment results to JSON file.

**Parameters:**
- `results` (list): List of model results
- `filename` (str, optional): Filename to save
- `preprocessing_info` (dict, optional): Preprocessing information

**Returns:**
- `str`: Saved filename

### `train.py`

#### `train(train_X, train_y, test_X, test_y, **kwargs)`
Performs model training and evaluation.

**Parameters:**
- `train_X` (array-like): Training features
- `train_y` (array-like): Training labels
- `test_X` (array-like): Test features  
- `test_y` (array-like): Test labels
- `model` (str, optional): Specific model name ("all" for all models)
- `seed` (int): Random seed (default: 42)
- `use_optuna` (bool): Whether to use Optuna
- `use_grid_search` (bool): Whether to use GridSearch
- `n_trials` (int): Number of Optuna trials (default: 10)
- `optimization_data_ratio` (float): Data ratio for optimization (default: 0.2)

**Returns:**
- `list`: List of model evaluation results

### `preprocess.py`

#### `preprocessing(scaler='standard', encoder='onehot')`
Performs data preprocessing.

**Parameters:**
- `scaler` (str): Scaler type ('standard', 'minmax')
- `encoder` (str): Encoder type ('onehot', 'label')

**Returns:**
- `tuple`: (train_X, train_y, test_X, test_y)

### `evaluate.py`

#### `evaluate_model_comprehensive(model, test_X, test_y, train_X, train_y, model_name, param_info)`
Performs comprehensive performance evaluation for models.

**Parameters:**
- `model`: Trained model object
- `test_X` (array-like): Test features
- `test_y` (array-like): Test labels
- `train_X` (array-like): Training features
- `train_y` (array-like): Training labels
- `model_name` (str): Model name
- `param_info` (dict): Model parameter information

**Returns:**
- `dict`: Evaluation result dictionary

## Supported Models

| Model Name | Class | Key Parameters |
|------------|-------|----------------|
| **Naive Bayes** | `GaussianNB` | `var_smoothing` |
| **K-Nearest Neighbors** | `KNeighborsClassifier` | `n_neighbors`, `metric`, `p` |
| **Logistic Regression** | `LogisticRegression` | `C`, `solver`, `max_iter` |
| **Decision Tree** | `DecisionTreeClassifier` | `max_depth`, `min_samples_split`, `criterion` |
| **Random Forest** | `RandomForestClassifier` | `n_estimators`, `max_depth`, `min_samples_split` |

## Hyperparameter Optimization

### Optuna Bayesian Optimization

**Advantages:**
- Tree-structured Parzen Estimator (TPE) algorithm
- Efficient parameter space exploration
- Early stopping functionality (Pruning)
- Support for various distributions

**Default Settings:**
- Number of trials: 10
- Optimization data ratio: 20% (of total training data)
- Cross validation: 5-fold CV

**Usage Examples:**
```bash
# Basic usage (10 trials, 20% data)
python main.py --optuna

# Set number of trials
python main.py --optuna --n-trials 50

# Apply to specific model only
python main.py --model "Random Forest" --optuna --n-trials 20
```

### GridSearchCV

**Features:**
- Exhaustive parameter combination search
- Reproducible results
- Reduced grid option available

**Default Settings:**
- Optimization data ratio: 20%
- Cross validation: 5-fold CV
- Evaluation metric: F1-Score

**Usage Examples:**
```bash
# Full grid search
python main.py --grid-search

# Reduced grid search (faster execution)
python main.py --grid-search --reduced-grid
```

## Data Preprocessing

### Supported Scalers

#### StandardScaler (Default)
```python
# Normalize to mean 0, standard deviation 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### MinMaxScaler
```python
# Scale to between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### Supported Encoders

#### OneHotEncoder (Default)
```python
# One-hot encode categorical variables
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X_categorical)
```

#### LabelEncoder
```python
# Encode categorical variables to integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_categorical)
```

### Preprocessing Combination Comparison

```bash
# Automatically compare 4 combinations: (standard/minmax) × (onehot/label)
python main.py --compare-preprocessing

# Save results
python main.py --compare-preprocessing --save-results
```

## Evaluation Metrics

### Basic Classification Metrics

The pipeline automatically calculates the following metrics through the `evaluate_model_comprehensive()` function:

#### **Accuracy**
```python
accuracy = accuracy_score(test_y, y_pred)
```
- **Definition**: Proportion of correct predictions among all predictions
- **Range**: 0.0 ~ 1.0 (higher is better)
- **Formula**: (TP + TN) / (TP + TN + FP + FN)

#### **Precision**
```python
precision = precision_score(test_y, y_pred, average='binary')
```
- **Definition**: Proportion of actual positives among positive predictions
- **Range**: 0.0 ~ 1.0 (higher is better)
- **Formula**: TP / (TP + FP)
- **Use case**: When reducing False Positives is important

#### **Recall**
```python
recall = recall_score(test_y, y_pred, average='binary')
```
- **Definition**: Proportion of correctly predicted positives among actual positives
- **Range**: 0.0 ~ 1.0 (higher is better)
- **Formula**: TP / (TP + FN)
- **Use case**: When reducing False Negatives is important

#### **F1-Score**
```python
f1 = f1_score(test_y, y_pred, average='binary')
```
- **Definition**: Harmonic mean of Precision and Recall
- **Range**: 0.0 ~ 1.0 (higher is better)
- **Formula**: 2 × (Precision × Recall) / (Precision + Recall)
- **Use case**: Metric considering balance between Precision and Recall

### Advanced Metrics

#### **ROC-AUC (Area Under ROC Curve)**
```python
roc_auc = roc_auc_score(test_y, y_pred_proba)
```
- **Definition**: Area under the ROC curve
- **Range**: 0.0 ~ 1.0 (0.5 is random, 1.0 is perfect)
- **Condition**: Only calculated for models with `predict_proba()` method
- **Use case**: Model performance evaluation independent of classification threshold

#### **Confusion Matrix**
```python
cm = confusion_matrix(test_y, y_pred)
tn, fp, fn, tp = cm.ravel()
```

Elements of confusion matrix:
- **True Positive (TP)**: Correctly predicted positive as positive
- **True Negative (TN)**: Correctly predicted negative as negative
- **False Positive (FP)**: Incorrectly predicted negative as positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted positive as negative (Type II error)

### Cross Validation

**5-fold Cross Validation** is used during hyperparameter optimization for stable performance evaluation:

```python
# CV score calculation in Optuna optimization
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
cv_score = cv_scores.mean()
```

### Result Output Format

#### Individual Model Results
```json
{
  "model_name": "Random Forest_Optuna",
  "parameters": {"n_estimators": 187, "max_depth": 12},
  "accuracy": 0.9200,
  "precision": 0.9150,
  "recall": 0.9350,
  "f1_score": 0.9250,
  "roc_auc": 0.9450,
  "confusion_matrix": [[85, 5], [3, 107]],
  "true_positive": 107,
  "true_negative": 85,
  "false_positive": 5,
  "false_negative": 3,
  "optuna_cv_score": 0.9250,
  "optuna_n_trials": 20
}
```

#### Overall Results Summary
The `print_evaluation_summary()` function outputs the following information:

1. **Top 5 models by F1-Score**
2. **Best performing model for each metric**
   - Best Accuracy
   - Best Precision  
   - Best Recall
   - Best F1-Score
   - Best ROC-AUC (when available)
3. **Preprocessing information** (when applicable)

#### Preprocessing Comparison Results
The `compare_preprocessing_results()` function provides:

1. **Performance ranking by preprocessing technique**
2. **Best preprocessing combination for each metric**
3. **Average performance vs. best performance comparison**

### Evaluation Metrics Selection Guide

| Scenario | Recommended Metrics | Reason |
|----------|-------------------|---------|
| **Balanced Dataset** | Accuracy, F1-Score | Suitable for overall performance evaluation |
| **Imbalanced Dataset** | Precision, Recall, F1-Score | Accuracy can be misleading |
| **Minimize False Positives** | Precision | Spam filtering, medical diagnosis, etc. |
| **Minimize False Negatives** | Recall | Fraud detection, disease screening, etc. |
| **Overall Classification Performance** | ROC-AUC | Threshold-independent evaluation |
| **Cost-Sensitive Classification** | Custom Cost Function | Reflects business costs |

## Examples

### Example 1: Basic Model Training

```bash
# Train all models and save results
python main.py --save-results
```

**Output:**
```
================================================================================
Starting Model Training and Evaluation
================================================================================

[Naive Bayes] Training model...
  Test Set - Accuracy: 0.8500, Precision: 0.8300, Recall: 0.8700, F1: 0.8500

[Random Forest] Training model...
  Test Set - Accuracy: 0.9100, Precision: 0.9000, Recall: 0.9200, F1: 0.9100
```

### Example 2: Optuna Optimization

```bash
# Apply Optuna to Random Forest
python main.py --model "Random Forest" --optuna --n-trials 50 --save-results
```

**Output:**
```
[Random Forest] Training model...
  Optuna searching for optimal parameters... (50 trials)
  Optimal parameters: {'n_estimators': 187, 'max_depth': 12, 'min_samples_split': 3}
  Optuna best CV F1-Score: 0.9250
  
  Test Set - Accuracy: 0.9200, Precision: 0.9150, Recall: 0.9350, F1: 0.9250
  ROC-AUC: 0.9450
```

### Example 3: Preprocessing Combination Comparison

```bash
# Compare preprocessing technique performance
python main.py --compare-preprocessing --model "Random Forest" --save-results
```

**Output:**
```
================================================================================
Preprocessing Technique Comparison Experiment
================================================================================
Testing 4 unique preprocessing combinations:
  1. Scaler: STANDARD, Encoder: ONEHOT
  2. Scaler: STANDARD, Encoder: LABEL
  3. Scaler: MINMAX, Encoder: ONEHOT
  4. Scaler: MINMAX, Encoder: LABEL

[1/4] Testing: Scaler=STANDARD, Encoder=ONEHOT
...
Best performing combination: STANDARD + ONEHOT (F1-Score: 0.9250)
```

### Example 4: Programmatic Usage

```python
from preprocess import preprocessing
from train import train
from evaluate import evaluate_model_comprehensive

# 1. Data preprocessing
print("Preprocessing data...")
train_X, train_y, test_X, test_y = preprocessing(
    scaler='standard',
    encoder='onehot'
)

# 2. Model training (using Optuna)
print("Training model...")
results = train(
    train_X, train_y, test_X, test_y,
    model="Random Forest",
    use_optuna=True,
    n_trials=20,
    seed=42
)

# 3. Output results
for result in results:
    print(f"Model: {result['model_name']}")
    print(f"F1-Score: {result['f1_score']:.4f}")
    print(f"Optimal parameters: {result['parameters']}")
```

## Project Structure

```
datascience-group-k/
├── main.py                 # Main execution file
├── train.py                # Model training and evaluation logic
├── evaluate.py             # Model performance evaluation functions
├── preprocess.py           # Data preprocessing module
├── grid_search.py          # GridSearchCV optimization
├── optuna_optimization.py  # Optuna Bayesian optimization
├── requirements.txt        # Package dependencies
├── pyproject.toml         # Project configuration
├── uv.lock                # uv lock file
├── README.md              # Project documentation
└── .gitignore            # Git ignore file
```

### Key Module Descriptions

- **`main.py`**: CLI interface and overall pipeline coordination
- **`train.py`**: Model training logic and parameter combination management
- **`evaluate.py`**: Comprehensive model evaluation and result output
- **`preprocess.py`**: Data preprocessing and scaling/encoding
- **`optuna_optimization.py`**: Optuna Bayesian optimization implementation
- **`grid_search.py`**: GridSearchCV grid search implementation

## Developer Guide

### Development Environment Setup

```bash
# Clone project
git clone <repository-url>
cd datascience-group-k

# Set up virtual environment (using uv)
uv venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync
```

### Adding New Models

1. Add model to `get_model_parameter_combinations()` function in `train.py`:

```python
"New Model": {
    "model": NewModelClass,
    "params": {"param1": value1, "param2": value2},
    "default_params": {}
}
```

2. Add parameter distributions to `optuna_optimization.py` and `grid_search.py` if needed

## Contributing

1. Register issues or propose features
2. Fork and create branch
3. Write code and test
4. Submit pull request

## Troubleshooting

### Frequently Asked Questions

**Q: When ModuleNotFoundError occurs**
```bash
# Check virtual environment activation
source .venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

**Q: Memory shortage error**
```bash
# Reduce optimization data ratio
python main.py --optimization-data-ratio 0.1
```

**Q: When Optuna optimization is too slow**
```bash
# Reduce number of trials and data sampling
python main.py --optuna --n-trials 5 --optimization-data-ratio 0.1
```

## License

This project is distributed under the MIT License. See the `LICENSE` file for more details.

