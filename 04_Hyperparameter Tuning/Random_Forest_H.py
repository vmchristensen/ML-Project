import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupKFold
from scipy.stats import randint
import random
from pathlib import Path

# Load data
script_dir = Path(__file__).parent
data_path = script_dir.parent / "01_Data" / "preprocessed_train.csv"
data = pd.read_csv(data_path)
categorical_features = ['language']
numerical_features = ['degree', 'closeness', 'betweenness', 'pagerank', 'harmonic', 'eigenvector', 'eccentricity']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('lang', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ])

X = preprocessor.fit_transform(data)
y = data['target']
groups = data['sentence_group']

# Custom evaluation for hyperparameter tuning
def evaluate_params(params, X, y, groups, n_splits=5):
    model = RandomForestClassifier(**params)
    group_kfold = GroupKFold(n_splits=n_splits)
    accuracies = []
    
    for train_idx, test_idx in group_kfold.split(X, y, groups=groups):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_test = groups.iloc[test_idx]
        
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        
        # Group predictions by sentence_id and select max probability
        test_data = data.iloc[test_idx].copy()
        test_data['proba'] = proba
        selected_roots = test_data.loc[test_data.groupby('sentence_id')['proba'].idxmax()]
        accuracy = (selected_roots['target'] == 1).mean()
        accuracies.append(accuracy)
    
    return np.mean(accuracies)

# Randomized hyperparameter search
n_iter = 30  
best_score = -np.inf
best_params = None

print(f"\nStarting randomized hyperparameter search with {n_iter} iterations...\n")

for i in range(n_iter):
    print(f"\nIteration {i+1}/{n_iter} - {((i+1)/n_iter)*100:.0f}% complete")
    print("----------------------------------------")
    params = {
        'n_estimators': random.choice([20, 100, 300, 500, 1000]),
        'max_depth': random.choice([None, 3, 5, 10, 20]),
        'min_samples_split': random.choice([2, 5, 10]),
        'min_samples_leaf': random.choice([1, 2, 5]),
        'max_features': random.choice(['sqrt', 'log2', None]),
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': 42
    }
    
    print("Testing parameters:", params)
    score = evaluate_params(params, X, y, groups)
    print(f"Iteration result: Score = {score:.3f}")
    
    if score > best_score:
        best_score = score
        best_params = params

print("\n=== Best Hyperparameters ===")
print(best_params)
print(f"Validation Accuracy: {best_score:.3f}")


"""
=== Best Hyperparameters ===
Testing parameters: {'n_estimators': 20, 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_features': 'log2', 'class_weight': 'balanced', 'n_jobs': -1, 'random_state': 42}
Iteration result: Score = 0.333
"""